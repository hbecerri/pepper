#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from argparse import ArgumentParser
import coffea
from collections import defaultdict
import json

from utils.config import Config


# Luminosities needed for CMS label at the top of plots
LUMIS = {
    "2016": "35.92",
    "2017": "41.53",
    "2018": "59.96",
}


def get_syshists(dirname, fname):
    keys, histnames = json.load(open(os.path.join(dirname, "hists.json")))
    try:
        histkey = keys[histnames.index(fname)]
    except ValueError:
        return []
    sysmap = defaultdict(lambda: [None, None])
    klen = len(histkey)
    for i, key in enumerate(keys):
        if len(key) == klen + 1 and key[:klen] == histkey:
            print("Processing " + histnames[i])
            if os.path.isabs(histnames[i]):
                histname = histnames[i]
            else:
                histname = os.path.join(dirname, histnames[i])
            sysname = key[klen]
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
    chan_axis = hist.axis("chan")
    if "proc" in hist.fields:
        proc_axis = hist.axis("proc")
        hist = hist.project(dense_axis, chan_axis, proc_axis)
    else:
        hist = hist.project(dense_axis, chan_axis)
    hist = hist.integrate(chan_axis, int_range=chan)
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


def plot(data_hist, pred_hist, sys, namebase, colors={}, cmsyear=None):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    coffea.hist.plot1d(pred_hist,
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
    coffea.hist.plotratio(data_hist.sum("proc"),
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
    if cmsyear is not None:
        ax1 = hep.cms.cmslabel(
            ax1, data=True, paper=False, year=cmsyear, lumi=LUMIS[cmsyear])
    ax1.autoscale(axis="y")
    plt.tight_layout()
    fig.savefig(namebase + ".svg")
    ax1.autoscale(axis="y")
    ax1.set_yscale("log")
    plt.tight_layout()
    fig.savefig(namebase + "_log.svg")
    plt.close()


parser = ArgumentParser(
    description="Plot histograms from previously created histograms")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "histfile", nargs="+", help="Coffea file with a single histogram")
parser.add_argument(
    "--labels", help="Path to a JSON file mapping the MC dataset names to "
    "proper names for plotting")
parser.add_argument(
    "--outdir", help="Output directory. If not given, output to the directory "
    "where histfile is located")
parser.add_argument(
    "--ignoresys", action="append", help="Ignore a specific systematic. "
    "Can be specified multiple times.")
args = parser.parse_args()

config = Config(args.config)
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
for histfilename in args.histfile:
    srcdir = os.path.dirname(histfilename)
    if args.outdir is None:
        outdir = srcdir
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    namebase, fileext = os.path.splitext(os.path.basename(histfilename))
    hist = coffea.util.load(histfilename)
    syshists = get_syshists(srcdir, os.path.basename(histfilename))
    scales = np.ones(len(syshists))
    if "tmass" in syshists:
        scales[list(syshists.keys()).index("tmass")] = 0.5

    proc_axis = coffea.hist.Cat("proc", "Process", "placement")

    data_dsnames = list(config["exp_datasets"].keys())
    data_hist = hist.group(
        hist.axis("dsname"), proc_axis, {"Data": (data_dsnames,)})

    mc_dsnames = config["mc_datasets"].keys()
    if axis_labelmap is not None:
        pred_hist = hist.group(hist.axis("dsname"), proc_axis, axis_labelmap)
    else:
        mapping = {key: (key,) for key in mc_dsnames}
        pred_hist = hist.group(hist.axis("dsname"), proc_axis, mapping)

    for dense in hist.dense_axes():
        for chan in (idn.name for idn in pred_hist.axis("chan").identifiers()):
            data_prepared = prepare(data_hist, dense, chan)
            pred_prepared = prepare(pred_hist, dense, chan)
            syshists_prep = {k: [prepare(vi, dense, chan) for vi in v]
                             for k, v in syshists.items()}
            sys = compute_systematic(pred_prepared,
                                     syshists_prep,
                                     scales[:, None, None])
            plot(data_prepared, pred_prepared, sys, os.path.join(
                outdir, f"{namebase}_{chan}"), colors=mc_colors)
