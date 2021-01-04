#!/usr/bin/env python3

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from argparse import ArgumentParser
import coffea
from collections import defaultdict
import json
from cycler import cycler

from pepper import Config
import pepper.plot  # noqa: E402


mpl.use("Agg")

# Luminosities needed for CMS label at the top of plots
LUMIS = {
    "2016": "35.9",
    "2017": "41.5",
    "2018": "59.7",
}


def get_syshists(histmap, histkey, dirname, ignore=None):
    if ignore is None:
        ignore = []
    sysmap = defaultdict(lambda: [None, None])
    klen = len(histkey)
    for key, histname in histmap.items():
        if len(key) == klen + 1 and key[:klen] == histkey:
            sysname = key[klen]
            if any(x in sysname for x in ignore):
                continue
            if not os.path.isabs(histname):
                histname = os.path.join(dirname, histname)
            if sysname.endswith("_down"):
                sysidx = 1
                sysname = sysname[:-len("_down")]
            elif sysname.endswith("_up"):
                sysidx = 0
                sysname = sysname[:-len("_up")]
            else:
                raise RuntimeError(f"Invalid sysname {sysname}, expecting it "
                                   "to end with '_up' or '_down'.")
            sysmap[sysname][sysidx] = coffea.util.load(histname)
    for key, updown in sysmap.items():
        if None in updown:
            raise RuntimeError(f"Missing variation for systematic {key}")
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
    for up_hist, down_hist in syshists.values():
        up = up_hist.values()[()]
        down = down_hist.values()[()]
        diff = np.stack([down, up]) - nom
        lo = np.minimum(np.min(diff, axis=0), 0)
        hi = np.maximum(np.max(diff, axis=0), 0)
        uncerts.append((lo, hi))
    if len(uncerts) == 0:
        return np.array([[0], [0]])
    uncerts = np.array(uncerts)
    if scales is not None:
        uncerts *= scales
    return np.sqrt((uncerts ** 2).sum(axis=0))


def plot(data_hist, bkgd_hist, sig_hist, sys, namebase, bkgd_cols={},
         log=False, cmsyear=None, ext=".svg", sig_scaling=1, sig_cols={}):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    pepper.plot.plot1d(bkgd_hist,
                       ax=ax1,
                       overlay="proc",
                       clear=False,
                       stack=True,
                       fill_opts={}, error_opts={
                           "hatch": "////",
                           "facecolor": "#00000000",
                           "label": "Uncertainty"},
                       sys=sys)
    if sig_hist is not None:
        sig_hist.scale(sig_scaling)
        if len(list(sig_cols.values())) > 0:
            ax1.set_prop_cycle(cycler(color=list(sig_cols.values())[::-1]))
        pepper.plot.plot1d(sig_hist,
                           ax=ax1,
                           clear=False,
                           overlay="proc",
                           stack=False)
    try:
        pepper.plot.plot1d(data_hist,
                           ax=ax1,
                           overlay="proc",
                           clear=False,
                           error_opts={
                               "color": "black",
                               "marker": "o",
                               "markersize": 4})
        pepper.plot.plotratio(data_hist.sum("proc"),
                              bkgd_hist.sum("proc"),
                              ax=ax2,
                              error_opts={"fmt": "ok", "markersize": 4},
                              denom_fill_opts={},
                              unc="num",
                              sys=sys)
    except:
        pass
    ax2.axhline(1, linestyle="--", color="black", linewidth=0.5)
    for handle, label in zip(*ax1.get_legend_handles_labels()):
        if label in bkgd_cols:
            handle.set_color(bkgd_cols[label])
    ax1.legend(ncol=2)
    ax1.set_xlabel("")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.set_ylabel("Data / Pred.")
    ax2.set_ylim(0.75, 1.25)
    fig.subplots_adjust(hspace=0)
    if cmsyear is not None:
        ax1 = hep.cms.label(
            ax=ax1, data=True, paper=False, year=cmsyear, lumi=LUMIS[cmsyear])
    if log:
        ax1.autoscale(axis="y")
        ax1.set_yscale("log")
    plt.tight_layout()
    fig.savefig(namebase + ext)
    plt.close()


parser = ArgumentParser(
    description="Plot histograms from previously created histograms")
parser.add_argument(
    "plot_config", help="Path to a configuration file for plotting")
parser.add_argument(
    "histfile", nargs="+", help="Coffea file with a single histogram or a "
    "JSON file containing histogram info. See output of select_events.py")
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
parser.add_argument(
    "-s", "--signals", nargs='*', default=["None"], help="Set of signal "
    "points to plot. Can be All, None (default) or a name (or series of names)"
    " of a specific set defined in the config")
parser.add_argument(
    "-c", "--cut",  type=int, metavar="cut_num", help="If specified, only "
    "plot a given cut number. Negative numbers count from the last cut (as "
    "for numpy), so -1 is the final cut")
args = parser.parse_args()

plt.set_loglevel("error")
config = Config(args.plot_config)

histfiles = []
for histfile in args.histfile:
    if histfile.endswith(".json"):
        dirname = os.path.dirname(histfile)
        with open(histfile) as f:
            f = json.load(f)
            for keys, histfile in zip(*f):
                if args.cut:
                    cut_nums = [int(name.split(" ", 2)[1]) for name in f[1]]
                    if args.cut < 0:
                        cut_num = np.max(cut_nums) + 1 + args.cut
                    else:
                        cut_num = args.cut
                    if int(histfile.split(" ", 2)[1]) != cut_num:
                        continue
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
        syshists = []
    scales = np.ones(len(syshists))
    if "tmass" in syshists:
        scales[list(syshists.keys()).index("tmass")] = 0.5

    proc_axis = coffea.hist.Cat("proc", "Process", "integral")

    data_hist = hist.group(dsaxis, proc_axis, {"Data": (config["Data"],)})
    mc_hist = hist.group(dsaxis, proc_axis, config["Labels"])

    if "channel" in hist.fields:
        channels = list(idn.name for idn in hist.axis("channel").identifiers())
    else:
        channels = []
    channels.insert(0, "all")
    for dense in hist.dense_axes():
        for chan in channels:
            if args.signals[0] == "All":
                sigs = config["Signal_samples"].keys()
            else:
                sigs = args.signals
            for sig in sigs:
                if sig == "None":
                    outdirchan = os.path.join(outdir, chan.replace("/", ""))
                    os.makedirs(outdirchan, exist_ok=True)
                else:
                    outdirchan = os.path.join(outdir, chan.replace("/", "")
                                              + "_" + sig.replace("/", ""))
                    os.makedirs(outdirchan, exist_ok=True)
                data_prepared = prepare(data_hist, dense, chan)
                mc_prepared = prepare(mc_hist, dense, chan)
                mc_bkgd_hist = \
                    mc_prepared[list(config["MC_bkgd"].keys())].copy()
                syshists_prep = {k: [prepare(vi, dense, chan) for vi in v]
                                 for k, v in syshists.items()}
                sys = compute_systematic(mc_bkgd_hist,
                                         syshists_prep,
                                         scales[:, None, None])
                if sig != "None":
                    sig_hist = \
                        mc_prepared[list(config["Signal_samples"][sig].keys())]
                    plot(data_prepared, mc_bkgd_hist, sig_hist, sys,
                         os.path.join(outdirchan, f"{namebase}_{chan}"),
                         bkgd_cols=config["MC_bkgd"], log=args.log,
                         cmsyear=config["year"], ext="." + args.ext,
                         sig_scaling=1000,
                         sig_cols=config["Signal_samples"][sig])
                else:
                    plot(data_prepared, mc_bkgd_hist, None, sys,
                         os.path.join(outdirchan, f"{namebase}_{chan}"),
                         bkgd_cols=config["MC_bkgd"], log=args.log,
                         cmsyear=config["year"], ext="." + args.ext)
