from collections import defaultdict
import os

import coffea
from coffea.analysis_objects import JaggedCandidateArray as JCA
from coffea.util import awkward, numpy
from coffea.util import numpy as np
from awkward import JaggedArray
import coffea.processor
from coffea.processor import parsl_executor
import uproot
import matplotlib
import matplotlib.pyplot as plt
from parsl import load, python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import CondorProvider
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel
import h5py
import mplhep
from cycler import cycler
import json

import pepper.config as config_utils
import pepper.datasets as dataset_utils
from pepper import Processor

matplotlib.interactive(False)


def plot_data_mc(hists, data, sigs=[], sig_scaling=1,
                 labels=None, colours=None, axis=None,
                 x_ax_name="x_ax", y_scale="linear"):
    if colours is not None:
        colours = colours.copy()
    if labels is not None:
        labelsset = list(set(labels.values()))
        labelmap = defaultdict(list)
        for key, val in labels.items():
            labelmap[val].append(key)

#        print((hists.integrate(x_ax_name)).values())
        print(x_ax_name)
        sortedlabels = sorted(labelsset, key=(
            lambda x: sum([(hists.integrate(x_ax_name)).values()[(y,)]
                           for y in labelmap[x]])))
        for key in sortedlabels:
            labelmap[key] = labelmap.pop(key)
            if colours is not None:
                colours[key] = colours.pop(key)

        labels_axis = coffea.hist.Cat("labels", "", sorting="placement")
        hists = hists.group("dataset", labels_axis, labelmap)
# Note hists are currently only ordered by integral if labels is specified
# -might want to change this
        axis = labels_axis
    if axis is None:
        raise ValueError("One of labels and axis must be specified")
    bkgd_hist = hists.remove((sigs+[data]), axis)
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": (3, 1)},
                                  sharex=True)
    ax = mplhep.cms.cmslabel(ax, data=False, paper=False, year='2018')
    fig.subplots_adjust(hspace=0)
    sig_colours = {}
    if colours is not None:
        for sig in sigs:
            sig_colours[sig] = colours.pop(sig)
        colours.pop(data)
        ax.set_prop_cycle(cycler(color=list(colours.values())[::-1]))
    fill_opts = {
        'edgecolor': (0, 0, 0, 0.3),
        'alpha': 0.8
    }
    err_opts = {
        'label': 'Stat. Unc.',
        'hatch': '///',
        'facecolor': 'none',
        'edgecolor': (0, 0, 0, 0.5),
        'linewidth':  0
    }
    coffea.hist.plot1d(bkgd_hist, overlay=axis, stack=True,
                       ax=ax, clear=False, fill_opts=fill_opts,
                       error_opts=err_opts)
    bkgdh = bkgd_hist.sum(axis)
    if len(list(sig_colours.values())) > 0:
        ax.set_prop_cycle(cycler(color=list(sig_colours.values())[::-1]))
    sig_hist = hists[sigs].copy()
    sig_hist.scale(sig_scaling)
    coffea.hist.plot1d(sig_hist,
                       ax=ax,
                       clear=False,
                       overlay=axis,
                       stack=False)
    data_err_opts = {
        'linestyle': 'none',
        'marker': '.',
        'markersize': 10.,
        'color': 'k',
        'elinewidth': 1,
    }
    coffea.hist.plot1d(
        hists[data],
        ax=ax,
        clear=False,
        error_opts=data_err_opts,
        overlay=axis)
    coffea.hist.plotratio(hists.integrate(axis, [data]),
                          bkgdh,
                          ax=rax,
                          error_opts=data_err_opts,
                          denom_fill_opts={},
                          guide_opts={},
                          unc='num')
    rax.set_ylabel('Ratio')
    rax.set_ylim(0, 2)
    ax.set_yscale(y_scale)
    if y_scale == "log":
        ax.set_ylim(10, 10**11)


def plot_cps(hist_set, plot_name,
             plot_kwargs, show=False, save_dir=None,
             channels=["ee", "emu", "mumu"], cuts="All"):
    def _plot(plot, plot_kwargs,
              show=False, save_dir=None, save_name=None):
        plot_data_mc(plot, **plot_kwargs)
        fig = plt.gcf()
        fig.suptitle(save_name)
        fig.set_size_inches(16, 12)
        if show:
            plt.show(block=True)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, save_name + ".pdf"))
        plt.clf()

    for key in hist_set.keys():
        if cuts == "All":
            plot_this_one = (len(key) == 2) & (key[1] == plot_name)
        elif type(cuts) == list:
            plot_this_one = ((len(key) == 2) & (key[1] == plot_name)
                             & (key[0] in cuts))
        if plot_this_one:
            hists = hist_set[key]
            ax_names = [ax.name for ax in hists.axes()]
            if type(channels) is list:
                if "channel" in ax_names:
                    for ch in channels:
                        plot = hists.integrate("channel", [ch])
                        _plot(plot, plot_kwargs,
                              show, save_dir,
                              key[0] + "_" + plot_name + "_" + ch)
            elif channels == "Sum":
                if "channel" in ax_names:
                    plot = hists.integrate("channel")
                else:
                    plot = hists
                _plot(plot, lumifactors, plot_kwargs,
                      show, save_dir,
                      key[0] + "_" + plot_name)


config = config_utils.Config("ttDM_config/config.json")
store = config["store"]
mc_fileset, _ = dataset_utils.expand_datasetdict(config["mc_datasets"], store)
data_fileset, _ = dataset_utils.expand_datasetdict(config["exp_datasets"],
                                                   store)
data_fileset.update(mc_fileset)
fileset = data_fileset
plot_config = config_utils.Config("example/plot_config.json")
labels = plot_config["labels"]
colours = plot_config["colours"]
xsecs = plot_config["cross-sections"]

output = coffea.util.load("out_hists/output.coffea")
cutvalues = dict((k, np.zeros(
    len(output["cutflows"]["all"]["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"])))
    for k in set(labels.values()))
cuteffs = dict((k, np.zeros(len(
    output["cutflows"]["all"]["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]) - 1))
    for k in set(labels.values()))
# currently assumes one always runs over a dilepton sample-
# might be nice to relax this
for dataset in xsecs.keys():
    cutf = output["cutflows"]["all"][dataset]
    if "Gen cut" in cutf.keys():
        cutf.pop("Gen cut")
    cutvals = np.array(list(cutf.values()))
    if len(cutvals) == 0:
        eff = 0
    else:
        eff = cutvals[-1]/cutvals[0]
    print(dataset, "efficiency:", eff*100)
    if len(cutvals > 0):
        cutvalues[labels[dataset]] += cutvals

labelsset = list(set(labels.values()))
nlabels = len(labelsset)
ax = plt.gca()
for n, label in enumerate(labelsset):
    cuteffs[label] = 100*cutvalues[label][1:]/cutvalues[label][:-1]
    ax.bar(np.arange(len(cuteffs[label])) + (2*n-nlabels)*0.4/nlabels,
           cuteffs[label], 0.8/nlabels, label=label, color=colours[label])

ax.set_xticks(np.arange(len(
    cuteffs[labels["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]])))
ax.set_xticklabels(np.array(list(
    (output["cutflows"]["all"]
        ["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]).keys()))[1:])
ax.set_ylabel("Efficiency")

handles, labs = ax.get_legend_handles_labels()
# https://stackoverflow.com/questions/43348348/pyplot-legend-index-error-tuple-index-out-of-range
leghandles = []
leglabs = []
for i, h in enumerate(handles):
    if len(h):
        leghandles.append(h)
        leglabs.append(labs[i])
ax.legend(leghandles, leglabs)

plt.show(block=True)

plt.style.use(mplhep.cms.style.ROOT)

plot_kwargs = {"data": "Data",
               "sigs": plot_config["signals"],
               "sig_scaling": 1000,
               "labels": labels,
               "colours": colours,
               "x_ax_name": "MET"}

hist2plot = {"MET": "MET",
             "Mll": "Mll",
             "dilep_pt": "dilep_pt",
             "jet_mult": "jet_mult",
             "1st_lep_pt": "pt",
             "2nd_lep_pt": "pt",
             "1st_lep_eta": "eta",
             "2nd_lep_eta": "eta"}

for hist, x_ax in hist2plot.items():
    plot_kwargs["x_ax_name"] = x_ax
    plot_cps(output["sel_hists"],
             hist,
             plot_kwargs,
             False,
             plot_config["hist_dir"],
             ["is_ee", "is_em", "is_mm"],
             ["MET > 40 GeV"])

hist2plot.pop("jet_mult")

for hist, x_ax in hist2plot.items():
    plot_kwargs["x_ax_name"] = x_ax
    plot_cps(output["sel_hists"],
             hist,
             plot_kwargs,
             False,
             plot_config["hist_dir"],
             ["is_ee", "is_em", "is_mm"],
             ["M_ll"])

hist = output["sel_hists"][("MET > 40 GeV", '1st_lep_eta_phi')]
hist.scale(unblinded_LFs, "dataset")
mc_hist = hist.remove(plot_config["process_names"]["Data"]
                      + ["DM Chi1 S100", "DM Chi1 PS100"], "dataset")
mc_hist = mc_hist.integrate("channel")
mc_hist = mc_hist.integrate("dataset")
coffea.hist.plot2d(mc_hist, "eta")
plt.show(block=True)

data_hist = hist.integrate("channel")
data_hist = data_hist.integrate("dataset",
                                plot_config["process_names"]["Data"])
coffea.hist.plot2d(data_hist, "eta")
plt.show(block=True)

data_hist.scale(-1)
mc_hist.add(data_hist)
coffea.hist.plot2d(mc_hist, "eta")
plt.show(block=True)
