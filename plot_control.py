#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import coffea
import json
from functools import partial
import itertools

from utils.config import Config
from utils.misc import get_event_files, montecarlo_iterate, expdata_iterate


class ComparisonHistogram1D(object):
    def __init__(self, data_hist, pred_hist):
        self.data_hist = data_hist
        self.pred_hist = pred_hist

    @classmethod
    def frombin(cls, cofbin, ylabel):
        data_hist = coffea.hist.Hist(
            ylabel, coffea.hist.Cat("proc", "Process", "integral"), cofbin)
        pred_hist = coffea.hist.Hist(
            ylabel, coffea.hist.Cat("proc", "Process", "integral"), cofbin)
        return ComparisonHistogram1D(data_hist, pred_hist)

    def fill(self, proc, **kwargs):
        if proc.lower() == "data":
            self.data_hist.fill(proc=proc, **kwargs)
        else:
            self.pred_hist.fill(proc=proc, **kwargs)

    def plot1d(self, fname_data, fname_pred):
        for hist, fname in zip([self.data_hist, self.pred_hist],
                               [fname_data, fname_pred]):
            fig, ax, p = coffea.hist.plot1d(
                self.data_hist, overlay="proc", stack=True)
            fig.savefig(fname)
            plt.close()

    def plotratio(self, namebase):
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        coffea.hist.plot1d(self.pred_hist,
                           ax=ax1,
                           overlay="proc",
                           stack=True,
                           fill_opts={}, error_opts={
                               "hatch": "////",
                               "facecolor": "#00000000",
                               "label": "Uncertainty"})
        coffea.hist.plot1d(self.data_hist,
                           ax=ax1,
                           overlay="proc",
                           clear=False,
                           error_opts={
                               "color": "black",
                               "marker": "o",
                               "markersize": 4})
        coffea.hist.plotratio(self.data_hist.sum("proc"),
                              self.pred_hist.sum("proc"),
                              ax=ax2,
                              error_opts={"fmt": "ok", "markersize": 4},
                              denom_fill_opts={},
                              unc="num")
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
        self._particle_name = particle_name
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

    def plotratio(self, chan, namebase):
        axes = ["pt", "eta", "phi"]
        for sumax in itertools.combinations(axes, len(axes) - 1):
            data_hist = (self.data_hist.integrate("chan", int_range=chan)
                                       .sum(*sumax, overflow="all"))
            pred_hist = (self.pred_hist.integrate("chan", int_range=chan)
                                       .sum(*sumax, overflow="all"))
            cmphist = ComparisonHistogram1D(data_hist, pred_hist)
            new_namebase = namebase + "_" + next(iter(set(axes) - set(sumax)))
            cmphist.plotratio(new_namebase)

    def plot2ddiff(self, outaxis, chan, namebase):
        data_hist = (self.data_hist.integrate("chan", int_range=chan)
                                   .sum("proc", outaxis, overflow="all"))
        pred_hist = (self.pred_hist.integrate("chan", int_range=chan)
                                   .sum("proc", outaxis, overflow="all"))
        for key in pred_hist._sumw.keys():
            pred_hist._sumw[key] = -pred_hist._sumw[key]
        data_hist.add(pred_hist)
        coffea.hist.plot2d(data_hist, data_hist.axes()[0])
        plt.savefig(namebase + ".svg")
        plt.close()

    def save(self, fname_data, fname_pred):
        coffea.util.save(self.data_hist, fname_data)
        coffea.util.save(self.pred_hist, fname_pred)


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
    "--cuts", type=int, help="Plot only events that have the given number as "
    "their entry in the cut_arrays")
args = parser.parse_args()

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
branches = ["Lepton_pt",
            "Lepton_eta",
            "Lepton_phi",
            "Lepton_pdgId",
            "cutflags"]

for name, weight, data in montecarlo_iterate(mc_datasets,
                                             montecarlo_factors,
                                             branches):
    if args.cuts is not None:
        cutsel = ((data["cutflags"] & args.cuts) == args.cuts).astype(bool)
        print("{}: {} events passing cuts, {} total".format(name,
                                                            cutsel.sum(),
                                                            data.size))
        data = data[cutsel]
        weight = weight[cutsel]
    else:
        print("{}: {} events".format(name, data.size))

    if labels_map:
        if name in labels_map:
            name = labels_map[name]
        else:
            print("No label given for {}".format(name))
    p0 = abs(data["Lepton_pdgId"][:, 0])
    p1 = abs(data["Lepton_pdgId"][:, 1])
    channels = {
        "ee": (p0 == 11) & (p1 == 11),
        "mm": (p0 == 13) & (p1 == 13),
        "em": p0 != p1,
    }
    for chan_name, sel in channels.items():
        lep_hists.fill(name,
                       chan_name,
                       data["Lepton_pt"][sel].flatten(),
                       data["Lepton_eta"][sel].flatten(),
                       data["Lepton_phi"][sel].flatten(),
                       np.repeat(weight[sel], 2))

for dsname, data in expdata_iterate(exp_datasets, branches):
    if args.cuts is not None:
        cutsel = ((data["cutflags"] & args.cuts) == args.cuts).astype(bool)
        print("Data: {} events passing cuts, {} total".format(cutsel.sum(),
                                                              data.size))
        data = data[cutsel]
    else:
        print("{}: {} events".format(dsname, data.size))

    p0 = abs(data["Lepton_pdgId"][:, 0])
    p1 = abs(data["Lepton_pdgId"][:, 1])
    channels = {
        "ee": (p0 == 11) & (p1 == 11),
        "mm": (p0 == 13) & (p1 == 13),
        "em": p0 != p1,
    }
    for chan_name, sel in channels.items():
        lep_hists.fill("Data",
                       chan_name,
                       data["Lepton_pt"][sel].flatten(),
                       data["Lepton_eta"][sel].flatten(),
                       data["Lepton_phi"][sel].flatten())

os.makedirs(args.outdir, exist_ok=True)
lep_hists.save(os.path.join(args.outdir, "hist_l_data.coffea"),
               os.path.join(args.outdir, "hist_l_mc.coffea"))
lep_hists.plotratio("ee", os.path.join(args.outdir, "ratio_ee"))
lep_hists.plotratio("mm", os.path.join(args.outdir, "ratio_ee"))
lep_hists.plotratio("em", os.path.join(args.outdir, "ratio_em"))
lep_hists.plot2ddiff("pt", "ee", os.path.join(args.outdir, "diff_ee"))
lep_hists.plot2ddiff("pt", "mm", os.path.join(args.outdir, "diff_mm"))
lep_hists.plot2ddiff("pt", "em", os.path.join(args.outdir, "diff_em"))
