#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from argparse import ArgumentParser
import coffea
import json
from functools import partial
import itertools

from utils.config import Config
from utils.misc import get_event_files, montecarlo_iterate, expdata_iterate


LUMIS = {
    "2016": "35.92",
    "2017": "41.53",
    "2018": "59.96",
}


class ComparisonHistogram(object):
    def __init__(self, data_hist, pred_hist):
        self.data_hist = data_hist
        self.pred_hist = pred_hist

    @classmethod
    def frombin(cls, cofbin, ylabel):
        data_hist = coffea.hist.Hist(
            ylabel,
            coffea.hist.Cat("proc", "Process", "integral"),
            coffea.hist.Cat("chan", "Channel"),
            cofbin)
        pred_hist = data_hist.copy(content=False)
        return cls(data_hist, pred_hist)

    def fill(self, proc, chan, **kwargs):
        if proc.lower() == "data":
            self.data_hist.fill(chan=chan, proc=proc, **kwargs)
        else:
            self.pred_hist.fill(chan=chan, proc=proc, **kwargs)

    def plot1d(self, chan, fname_data, fname_pred):
        for hist, fname in zip([self.data_hist, self.pred_hist],
                               [fname_data, fname_pred]):
            hist = hist.integrate("chan", int_range=chan)
            fig, ax, p = coffea.hist.plot1d(hist, overlay="proc", stack=True)
            fig.savefig(fname)
            plt.close()

    def plotratio(self, chan, namebase, colors={}, cmsyear=None):
        data_hist = self.data_hist.integrate("chan", int_range=chan)
        pred_hist = self.pred_hist.integrate("chan", int_range=chan)
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        if cmsyear is not None:
            ax1 = hep.cms.cmslabel(
                ax1, data=True, paper=False, year=cmsyear, lumi=LUMIS[cmsyear])
        coffea.hist.plot1d(pred_hist,
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
                              pred_hist.sum("proc"),
                              ax=ax2,
                              error_opts={"fmt": "ok", "markersize": 4},
                              denom_fill_opts={},
                              unc="num")
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
        fig.savefig(namebase + ".svg")
        ax1.autoscale(axis="y")
        ax1.set_yscale("log")
        fig.savefig(namebase + "_log.svg")
        plt.close()

    def save(self, fname_data, fname_pred):
        coffea.util.save(self.data_hist, fname_data)
        coffea.util.save(self.pred_hist, fname_pred)


class ParticleComparisonHistograms(object):
    def __init__(self, particle_name, data_hist, pred_hist):
        self.particle_name = particle_name
        self.data_hist = data_hist
        self.pred_hist = pred_hist

    @classmethod
    def create_empty(cls, particle_name):
        bins = (
            coffea.hist.Cat("proc", "Process", "integral"),
            coffea.hist.Cat("chan", "Channel"),
            coffea.hist.Bin(
                "pt", "{} $p_{{T}}$ in GeV".format(particle_name), 20, 0, 200),
            coffea.hist.Bin(
                "eta", r"{} $\eta$".format(particle_name), 26, -2.6, 2.6),
            coffea.hist.Bin(
                "phi", r"{} $\varphi$".format(particle_name), 38, -3.8, 3.8)
        )
        data_hist = coffea.hist.Hist("Counts", *bins)
        pred_hist = coffea.hist.Hist("Counts", *bins)
        return cls(particle_name, data_hist, pred_hist)

    def fill(self, proc, chan, pt, eta, phi, weight=1):
        if proc.lower() == "data":
            hist = self.data_hist
        else:
            hist = self.pred_hist
        hist.fill(proc=proc, chan=chan, pt=pt, eta=eta, phi=phi, weight=weight)

    def fill_from_data(self, proc, chan, data, weight=1):
        pt = data[self.particle_name + "_pt"].flatten()
        eta = data[self.particle_name + "_eta"].flatten()
        phi = data[self.particle_name + "_phi"].flatten()
        if isinstance(weight, np.ndarray):
            counts = data[self.particle_name + "_pt"].counts
            weight = np.repeat(weight, counts)
        self.fill(proc, chan, pt, eta, phi, weight)

    def plotratio(self, chan, namebase, colors={}, cmsyear=None):
        axes = ["pt", "eta", "phi"]
        for sumax in itertools.combinations(axes, len(axes) - 1):
            data_hist = self.data_hist.sum(*sumax, overflow="all")
            pred_hist = self.pred_hist.sum(*sumax, overflow="all")
            cmphist = ComparisonHistogram(data_hist, pred_hist)
            new_namebase = namebase + "_" + next(iter(set(axes) - set(sumax)))
            cmphist.plotratio(chan, new_namebase, colors, cmsyear)

    def plot2ddiff(self, outaxis, chan, namebase):
        data_hist = (self.data_hist.integrate("chan", int_range=chan)
                                   .sum("proc", outaxis, overflow="all"))
        pred_hist = (self.pred_hist.integrate("chan", int_range=chan)
                                   .sum("proc", outaxis, overflow="all"))
        pred_count, pred_err = pred_hist.values(
            sumw2=True, overflow="allnan")[()]
        data_count, data_err = data_hist.values(
            sumw2=True, overflow="allnan")[()]
        err = np.sqrt(data_err + pred_err)
        err[err == 0] = 1

        sig = (data_count - pred_count) / err
        sig_hist = pred_hist.copy(content=False)
        sig_hist._sumw[()] = sig
        sig_hist._sumw2 = None
        sig_hist.label = "(Data - Pred) / Err"

        display_max = abs(sig_hist.values()[()]).max()
        patch_opts = {
            "cmap": "RdBu",
            "vmin": -display_max,
            "vmax": display_max,
        }
        coffea.hist.plot2d(sig_hist, sig_hist.axes()[0], patch_opts=patch_opts)
        plt.savefig(namebase + ".svg")
        plt.close()

        coffea.hist.plot2d(data_hist, sig_hist.axes()[0])
        plt.savefig(namebase + "_data.svg")
        plt.close()
        coffea.hist.plot2d(pred_hist, sig_hist.axes()[0])
        plt.savefig(namebase + "_pred.svg")
        plt.close()

    def save(self, fname_data, fname_pred):
        coffea.util.save(self.data_hist, fname_data)
        coffea.util.save(self.pred_hist, fname_pred)


def get_channel_masks(data):
    p0 = abs(data["Lepton_pdgId"][:, 0])
    p1 = abs(data["Lepton_pdgId"][:, 1])
    return {
        "ee": (p0 == 11) & (p1 == 11),
        "mm": (p0 == 13) & (p1 == 13),
        "em": p0 != p1,
    }


parser = ArgumentParser()
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "eventdir", help="Path to the directory with the events. Event files need "
    "to be in subdirectories named by their dataset")
parser.add_argument(
    "--eventext", default=".hdf5", help="File extension of the event files. "
    "Defaults to \".hdf5\"")
parser.add_argument(
    "--outdir", default="plots", help="Path where to output the plots. By "
    "default a subdirectiory \"plots\"")
parser.add_argument(
    "--labels", help="Path to a JSON file mapping the MC dataset names to "
    "proper names for plotting")
parser.add_argument(
    "--cuts", type=int, help="Plot only events that pass the cuts "
    "corresponding to this number")
parser.add_argument(
    "--negcuts", type=int, help="Plot only events that do not pass the cuts "
    "corresponding to this number")
parser.add_argument(
    "--dyscale", type=float, nargs=3,
    metavar=("ee_scale", "em_scale", "mm_scale"), help="Scale DY samples "
    "according to a channel-dependent factor")
parser.add_argument(
    "--debug", nargs=2, metavar=("data_hist", "pred_hist"))
args = parser.parse_args()

if args.debug is not None:
    data_hist = coffea.util.load(args.debug[0])
    pred_hist = coffea.util.load(args.debug[1])
    lep_hists = ParticleComparisonHistograms("Lepton", data_hist, pred_hist)
    sys.exit(0)

config = Config(args.config)

exp_datasets = get_event_files(
    args.eventdir, args.eventext, config["exp_datasets"])
mc_datasets = get_event_files(
    args.eventdir, args.eventext, config["mc_datasets"])
montecarlo_factors = config["mc_lumifactors"]

if args.labels:
    with open(args.labels) as f:
        labels_map = json.load(f)
else:
    labels_map = None

lep_hists = ParticleComparisonHistograms.create_empty("Lepton")
jet_hists = ParticleComparisonHistograms.create_empty("Jet")
njets_hist = ComparisonHistogram.frombin(
    coffea.hist.Bin("njets", "Number of jets", np.arange(10)), "Counts")
met_hist = ComparisonHistogram.frombin(
    coffea.hist.Bin(
        "met", "Missing transverse momentum", 20, 0, 200), "Counts")
mll_hist = ComparisonHistogram.frombin(
    coffea.hist.Bin(
        "mll", "Invariant mass of the lepton system", 20, 0, 200), "Counts")
branches = ["Lepton_pt",
            "Lepton_eta",
            "Lepton_phi",
            "Lepton_pdgId",
            "Jet_pt",
            "Jet_eta",
            "Jet_phi",
            "MET_pt",
            "mll",
            "cutflags"]

mc_colors = {}
for name, weight, data in montecarlo_iterate(mc_datasets,
                                             montecarlo_factors,
                                             branches):
    cutsel = None
    if args.cuts is not None:
        cutsel = (data["cutflags"] & args.cuts) == args.cuts
    if args.negcuts is not None:
        if cutsel is None:
            cutsel = np.full(data["cutflags"].size, True)
        cutsel &= (~data["cutflags"] & args.negcuts) == args.negcuts
    if cutsel is not None:
        print("{}: {} events passing cuts, {} total".format(name,
                                                            cutsel.sum(),
                                                            data.size))
        data = data[cutsel]
        weight = weight[cutsel]
    else:
        print("{}: {} events".format(name, data.size))

    is_dy = name.startswith("DYJets")
    if labels_map:
        if name in labels_map:
            name = labels_map[name]
        else:
            print("No label given for {}".format(name))
    if name not in mc_colors:
        mc_colors[name] = "C" + str(len(mc_colors))
    for chan_name, sel in get_channel_masks(data).items():
        if args.dyscale is not None and is_dy:
            weight[sel] *= args.dyscale[["ee", "em", "mm"].index(chan_name)]

        lep_hists.fill_from_data(name, chan_name, data[sel], weight[sel])
        jet_hists.fill_from_data(name, chan_name, data[sel], weight[sel])
        njets_hist.fill(proc=name,
                        chan=chan_name,
                        njets=data[sel]["Jet_pt"].counts,
                        weight=weight[sel])
        met_hist.fill(proc=name,
                      chan=chan_name,
                      met=data[sel]["MET_pt"],
                      weight=weight[sel])
        mll_hist.fill(proc=name,
                      chan=chan_name,
                      mll=data[sel]["mll"],
                      weight=weight[sel])

for name, data in expdata_iterate(exp_datasets, branches):
    cutsel = None
    if args.cuts is not None:
        cutsel = (data["cutflags"] & args.cuts) == args.cuts
    if args.negcuts is not None:
        if cutsel is None:
            cutsel = np.full(data["cutflags"].size, True)
        cutsel &= (~data["cutflags"] & args.negcuts) == args.negcuts
    if cutsel is not None:
        print("{}: {} events passing cuts, {} total".format(name,
                                                            cutsel.sum(),
                                                            data.size))
        data = data[cutsel]
    else:
        print("{}: {} events".format(name, data.size))

    for chan_name, sel in get_channel_masks(data).items():
        lep_hists.fill_from_data("Data", chan_name, data[sel])
        jet_hists.fill_from_data("Data", chan_name, data[sel])
        njets_hist.fill("Data",
                        chan=chan_name,
                        njets=data[sel]["Jet_pt"].counts)
        met_hist.fill("Data",
                      chan=chan_name,
                      met=data[sel]["MET_pt"])
        mll_hist.fill("Data",
                      chan=chan_name,
                      mll=data[sel]["mll"])

os.makedirs(args.outdir, exist_ok=True)
getout = partial(os.path.join, args.outdir)
lep_hists.save(getout("hist_l_data.coffea"), getout("hist_l_mc.coffea"))
for chan in ("ee", "mm", "em"):
    lep_hists.plotratio(chan, getout(f"{chan}_lepton"), mc_colors)
    lep_hists.plot2ddiff("pt", chan, getout(f"diff_{chan}_lep"))
    jet_hists.plotratio(chan, getout(f"{chan}_jet"), mc_colors)
    jet_hists.plot2ddiff("pt", chan, getout(f"diff_{chan}_jet"))
    njets_hist.plotratio(chan, getout(f"{chan}_jet_n"), mc_colors)
    met_hist.plotratio(chan, getout(f"{chan}_met"), mc_colors)
    mll_hist.plotratio(chan, getout(f"{chan}_mll"), mc_colors)
