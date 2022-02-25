#!/usr/bin/env python3

import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
import coffea  # noqa: E402
from collections import defaultdict  # noqa: E402
import json  # noqa: E402

import pepper  # noqa: E402
import pepper.plot  # noqa: E402


# Luminosities needed for CMS label at the top of plots
LUMIS = {
    "2016": "35.9",
    "2017": "41.5",
    "2018": "59.7",
    "ul2016pre": "19.5",
    "ul2016post": "16.8",
    "ul2017": "41.5",
    "ul2018": "59.9",
}


def get_syshists(histmap, histkey, dirname, ignore=None):
    if ignore is None:
        ignore = []
    sysmap = defaultdict(list)
    klen = len(histkey)
    for key, histname in histmap.items():
        if len(key) == klen + 1 and key[:klen] == histkey:
            sysname, direction = key[-1].rsplit("_", 1)
            if any(x in sysname for x in ignore):
                continue
            if not os.path.isabs(histname):
                histname = os.path.join(dirname, histname)
            sysmap[sysname].append(coffea.util.load(histname))
    return sysmap


def prepare(hist, dense_axis, chan):
    project_axes = [dense_axis]
    if "channel" in hist.fields:
        project_axes.append(hist.axis("channel"))
    if "proc" in hist.fields:
        project_axes.append(hist.axis("proc"))
    hist = hist.project(*project_axes)
    if "channel" in hist.fields:
        if chan == "all":
            chan = slice(None)
        hist = hist.integrate(hist.axis("channel"), int_range=chan)
    return hist


def compute_systematic(nominal_hist, syshists, scales=None):
    nominal_hist = nominal_hist.sum("proc")
    nom = nominal_hist.values()[()]
    uncerts = []
    for hists in syshists.values():
        values = np.stack([hist.values()[()] for hist in hists])
        diff = values - nom
        lo = np.minimum(np.min(diff, axis=0), 0)
        hi = np.maximum(np.max(diff, axis=0), 0)
        uncerts.append((lo, hi))
    if len(uncerts) == 0:
        return np.array([[0], [0]])
    uncerts = np.array(uncerts)
    if scales is not None:
        uncerts *= scales
    return np.sqrt((uncerts ** 2).sum(axis=0))


def set_limit_ratioplot(ax, data, center=1., points_out=0.8, minimum=0.25,
                        margin_factor=1.2):
    data = data[np.isfinite(data)]
    if len(data) == 0:
        q = minimum
    else:
        q = np.quantile(abs(data - center), points_out) * margin_factor
    if q < minimum:
        q = minimum
    ax.set_ylim(center - q, center + q)


def plot(data_hist, pred_hist, sys, namebase, colors={}, log=False,
         cmsyear=None, ext=".svg"):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    pepper.plot.plot1d(pred_hist,
                       ax=ax1,
                       overlay="proc",
                       clear=False,
                       stack=True,
                       fill_opts={}, error_opts={
                           "hatch": "////",
                           "facecolor": "#00000000",
                           "label": "Uncertainty"},
                       sys=sys)
    coffea.hist.plot1d(data_hist,
                       ax=ax1,
                       overlay="proc",
                       clear=False,
                       error_opts={
                           "color": "black",
                           "marker": "o",
                           "markersize": 4})
    pepper.plot.plotratio(data_hist.sum("proc"),
                          pred_hist.sum("proc"),
                          ax=ax2,
                          error_opts={"fmt": "ok", "markersize": 4},
                          denom_fill_opts={},
                          unc="num",
                          sys=sys)
    ax2.axhline(1, linestyle="--", color="black", linewidth=0.5)
    for handle, label in zip(*ax1.get_legend_handles_labels()):
        if label in colors:
            handle.set_color(colors[label])
    ax1.legend(ncol=2)
    ax1.set_xlabel("")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.set_ylabel("Data / Pred.")
    ax2.set_ylim(0.75, 1.25)
    fig.subplots_adjust(hspace=0)
    ax1.autoscale(axis="y")
    if cmsyear is not None:
        year = "".join(c for c in cmsyear if c.isdigit())
        ax1 = hep.cms.label(
            ax=ax1, llabel="Work in progress", year=year,
            lumi=LUMIS[cmsyear])
    if log:
        ax1.set_yscale("log")
    plt.tight_layout()
    fig.savefig(namebase + ext)
    plt.close()


parser = ArgumentParser(
    description="Plot histograms from previously created histograms")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "histfile", nargs="+", help="Coffea file with a single histogram or a "
    "JSON file containing histogram info. See output of select_events.py")
parser.add_argument(
    "--labels", help="Path to a JSON file mapping the MC dataset names to "
    "proper names for plotting")
parser.add_argument(
    "--outdir", help="Output directory. If not given, output to the directory "
    "where histfile is located")
parser.add_argument(
    "--ignoresys", action="append", help="Ignore a specific systematic. "
    "Can be specified multiple times.")
parser.add_argument(
    "--log", action="store_true", help="Make logarithmic plots")
parser.add_argument(
    "--ext", choices=["pdf", "svg", "png"], help="Output file format",
    default="svg")
args = parser.parse_args()

plt.set_loglevel("error")
config = pepper.Config(args.config)
mc_colors = {}
if args.labels:
    with open(args.labels) as f:
        labels_map = json.load(f)
        axis_labelmap = defaultdict(lambda: (list(),))
        for dsname, label in labels_map.items():
            axis_labelmap[label][0].append(dsname)
        for label in axis_labelmap.keys():
            mc_colors[label] = "C" + str(len(axis_labelmap)
                                         - len(mc_colors) - 1)
else:
    axis_labelmap = None
histfiles = []
for histfile in args.histfile:
    if histfile.endswith(".json"):
        dirname = os.path.dirname(histfile)
        with open(histfile) as f:
            for keys, histfile in zip(*json.load(f)):
                if len(keys) != 2:
                    continue
                histfiles.append(os.path.join(dirname, histfile))
    else:
        histfiles.append(histfile)
for histfilename in histfiles:
    print("Processing {}".format(histfilename))
    srcdir = os.path.dirname(histfilename)
    if os.path.exists(os.path.join(srcdir, "hists.json")):
        with open(os.path.join(srcdir, "hists.json")) as f:
            histmap = {tuple(k): v for k, v in zip(*json.load(f))}
        histmap_inv = dict(zip(histmap.values(), histmap.keys()))
        histkey = histmap_inv[os.path.relpath(histfilename, srcdir)]
    else:
        histmap = None
        histkey = None
    if args.outdir is None:
        outdir = srcdir
    else:
        outdir = args.outdir
    if histkey is not None:
        # Create subdirectiories by hist identifier
        obsname = histkey[1].replace("/", "")
        outdir = os.path.join(outdir, obsname)
    os.makedirs(outdir, exist_ok=True)
    namebase, fileext = os.path.splitext(os.path.basename(histfilename))
    hist = coffea.util.load(histfilename)
    dsaxis = hist.axis("dataset")
    if histmap is not None:
        syshists = get_syshists(histmap, histkey, srcdir, args.ignoresys)
    else:
        syshists = {}
    scales = np.ones(len(syshists))
    if "tmass" in syshists:
        # Scale down top mass uncertainty. This assumes that the 175p5 and
        # 169p5 datasets were used.
        scales[list(syshists.keys()).index("tmass")] = 1 / 6

    proc_axis = coffea.hist.Cat("proc", "Process", "placement")

    data_dsnames = list(config["exp_datasets"].keys())
    # Skip if this is an MC-only histogram
    if all(name not in [i.name for i in hist.identifiers(dsaxis)]
           for name in data_dsnames):
        continue
    data_hist = hist.group(dsaxis, proc_axis, {"Data": (data_dsnames,)})

    mc_dsnames = config["mc_datasets"].keys()
    if axis_labelmap is not None:
        pred_hist = hist.group(dsaxis, proc_axis, axis_labelmap)
        used_datasets = {d for d2 in axis_labelmap.values() for d in d2[0]}
        for key, syshist in syshists.items():
            grouped_hists = []
            for h in syshist:
                not_present = used_datasets - set(
                    idn.name for idn in syshist[0].identifiers("dataset"))
                grouped = h.integrate("dataset", int_range=list(
                    used_datasets - not_present))
                if len(not_present) > 0:
                    grouped.add(hist.integrate("dataset", int_range=list(
                        not_present)))
                grouped_hists.append(grouped)
            syshists[key] = grouped_hists
    else:
        mapping = {key: (key,) for key in mc_dsnames}
        pred_hist = hist.group(dsaxis, proc_axis, mapping)

    if "channel" in hist.fields:
        channels = list(idn.name for idn in hist.axis("channel").identifiers())
    else:
        channels = []
    channels.insert(0, "all")
    for dense in hist.dense_axes():
        for chan in channels:
            outdirchan = os.path.join(outdir, chan.replace("/", ""))
            os.makedirs(outdirchan, exist_ok=True)
            data_prepared = prepare(data_hist, dense, chan)
            pred_prepared = prepare(pred_hist, dense, chan)
            syshists_prep = {k: [prepare(vi, dense, chan) for vi in v]
                             for k, v in syshists.items()}
            sys = compute_systematic(pred_prepared,
                                     syshists_prep,
                                     scales[:, None, None])
            plot(data_prepared, pred_prepared, sys, os.path.join(
                 outdirchan, f"{namebase}_{chan}"), colors=mc_colors,
                 log=args.log, cmsyear=config["year"], ext="." + args.ext)
