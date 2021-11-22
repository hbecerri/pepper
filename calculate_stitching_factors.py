#!/usr/bin/env python3

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
from argparse import ArgumentParser
import coffea.util
import hjson


mpl.use("Agg")

# Luminosities needed for CMS label at the top of plots
LUMIS = {
    "2016": "35.9",
    "2017": "41.5",
    "2018": "59.7",
}


def plot(data_hist, bkgd_hist, namebase, bkgd_cols={}, log=False, cmsyear=None,
         ext=".svg", yields=False, remove=False):
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    coffea.hist.plot1d(bkgd_hist,
                       ax=ax1,
                       overlay="proc",
                       clear=False,
                       stack=True,
                       fill_opts={}, error_opts={
                           "hatch": "////",
                           "facecolor": "#00000000",
                           "label": "Uncertainty"})
    coffea.hist.plot1d(data_hist,
                       ax=ax1,
                       overlay="proc",
                       clear=False,
                       error_opts={
                           "color": "black",
                           "marker": "o",
                           "markersize": 4})
    coffea.hist.plotratio(data_hist.sum("proc"),
                          bkgd_hist.sum("proc"),
                          ax=ax2,
                          error_opts={"fmt": "ok", "markersize": 4},
                          denom_fill_opts={},
                          unc="num")
    ax2.axhline(1, linestyle="--", color="black", linewidth=0.5)
    handles, labels = [], []
    for handle, label in zip(*ax1.get_legend_handles_labels()):
        if yields and label != "Uncertainty":
            if label in bkgd_cols:
                handle.set_color(bkgd_cols[label])
                int_hist = bkgd_hist.integrate("proc", [label])
            else:
                int_hist = data_hist
            int_hist = int_hist.sum(*[ax for ax in int_hist.fields])
            label = label + ": " + "{:.1f}".format(int_hist.values()[()])
        elif label in bkgd_cols:
            handle.set_color(bkgd_cols[label])
        handles.append(handle)
        labels.append(label)
    legend = ax1.legend(handles, labels, ncol=2, prop={'size': 6})
    if remove:
        legend.remove()
    ax1.set_xlabel("")
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.set_ylabel("Data / Pred.")
    ax2.set_ylim(0.8, 1.2)
    fig.subplots_adjust(hspace=0)
    if log:
        ax1.autoscale(axis="y")
        ax1.set_yscale("log")
    if cmsyear is not None:
        ax1 = hep.cms.label(ax=ax1, data=True, paper=True, year=cmsyear,
                            lumi=LUMIS[cmsyear], llabel="Work in progress")
    plt.tight_layout()
    fig.savefig(namebase + ext)
    plt.close()


def round_to_4(x):
    return round(x, 3-int(np.floor(np.log10(abs(x)))))


parser = ArgumentParser(
    description="Calculate factors to stitch MC samples from an already "
    "produced histogram (currently only 1D histograms supported)")
parser.add_argument(
    "plot_config", help="Path to a configuration file for plotting, where "
    "the 'MC_bkgd' contain the samples to be stitched (including the "
    "inclusive), and the 'Data' specifies the inclusive sample(s). Can "
    "also include a rebinning to be performed on histogram before "
    "calculating factors")
parser.add_argument(
    "histfile", help="Coffea file with a single histogram produced by "
    "select_events.py with the binning of the files to stitched")
parser.add_argument(
    "-p", "--plotdir", default=None, help="If specified, plot the stiched "
    "samples (including factors) and inclusive in the given directory")
parser.add_argument(
    "-l", "--log", action="store_true", help="Make logarithmic plots")
parser.add_argument(
    "-y", "--yields", action="store_true", help="Add integral of hists to "
    "legend")
args = parser.parse_args()

plt.set_loglevel("error")
with open(args.plot_config) as f:
    config = hjson.load(f)

print("Processing {}".format(args.histfile))
srcdir = os.path.dirname(args.histfile)
if os.path.exists(os.path.join(srcdir, "hists.json")):
    with open(os.path.join(srcdir, "hists.json")) as f:
        histmap = {tuple(k): v for k, v in zip(*hjson.load(f))}
    histmap_inv = dict(zip(histmap.values(), histmap.keys()))
    histkey = histmap_inv[os.path.relpath(args.histfile, srcdir)]
else:
    histmap = None
    histkey = None
if args.plotdir:
    os.makedirs(args.outdir, exist_ok=True)
    namebase, fileext = os.path.splitext(os.path.basename(args.histfile))
hist = coffea.util.load(args.histfile)
dsaxis = hist.axis("dataset")
proc_axis = coffea.hist.Cat("proc", "Process", "integral")

if "rebin" in config:
    for old_ax, new_ax in config["rebin"].items():
        if old_ax in hist.axes():
            hist = hist.rebin(old_ax, coffea.hist.Bin(**new_ax))
data_hist = hist.group(dsaxis, proc_axis, {"Data": (config["Data"],)})
mc_hist = hist.group(dsaxis, proc_axis, config["Labels"])

if len(hist.dense_axes()) != 1:
    raise ValueError("Can only calculate stitching factors for 1d histograms!")
dense = hist.dense_axes()[0]
mc_preped = mc_hist.project(dense, "proc")
data_preped = data_hist.project(dense, "proc")
sfs = (data_preped.integrate("proc").values(overflow="allnan")[()]
       / mc_preped.integrate("proc").values(overflow="allnan")[()])
print([round_to_4(sf) for sf in sfs.tolist()[1:-2]])
if args.plotdir:
    for k in mc_preped._sumw.keys():
        mc_preped._sumw[k] = mc_preped._sumw[k] * sfs
        mc_preped._sumw2[k] = mc_preped._sumw2[k] * sfs ** 2
    plot(data_preped, mc_preped, os.path.join(args.outdir, namebase),
         config["MC_bkgd"], log=args.log, cmsyear=config["year"],
         ext=".pdf", yields=args.yields)
