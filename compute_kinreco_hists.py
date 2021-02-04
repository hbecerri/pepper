#!/usr/bin/env python3

import os
import sys
import numpy as np
import awkward as ak
import coffea
from coffea.nanoevents import NanoAODSchema
import parsl
from argparse import ArgumentParser

import pepper
from pepper.misc import export
from pepper.datasets import expand_datasetdict

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import uproot3  # noqa: E402


class Processor(pepper.ProcessorTTbarLL):
    def __init__(self, config):
        config["reco_algorithm"] = None
        if "reco_info_file" in config:
            del config["reco_info_file"]
        config["blinding_denom"] = None
        config["compute_systematics"] = False
        super().__init__(config, None)

    @property
    def accumulator(self):
        mlb_axis = coffea.hist.Bin(
            "mlb", r"$m_{\mathrm{lb}}$ (GeV)", 200, 0, 200)
        mlb = coffea.hist.Hist("Counts", mlb_axis)
        # W mass resolution in NanoAOD seems to be 0.25
        mw_axis = coffea.hist.Bin(
            "mw", r"$m_{\mathrm{W}}$ (GeV)", 160, 40, 120)
        mw = coffea.hist.Hist("Counts", mw_axis)
        mt_axis = coffea.hist.Bin(
            "mt", r"$m_{\mathrm{t}}$ (GeV)", 180, 160, 190)
        mt = coffea.hist.Hist("Counts", mt_axis)
        alphal_axis = coffea.hist.Bin("alpha", r"$\alpha$ (rad)", 20, 0, 0.02)
        alphal = coffea.hist.Hist("Counts", alphal_axis)
        alphaj_axis = coffea.hist.Bin("alpha", r"$\alpha$ (rad)", 100, 0, 0.2)
        alphaj = coffea.hist.Hist("Counts", alphaj_axis)
        energyfl_axis = coffea.hist.Bin(
            "energyf", r"$E_{\mathrm{gen}} / E_{\mathrm{reco}}", 200, 0.5, 1.5)
        energyfl = coffea.hist.Hist("Counts", energyfl_axis)
        energyfj_axis = coffea.hist.Bin(
            "energyf", r"$E_{\mathrm{gen}} / E_{\mathrm{reco}}", 200, 0, 3)
        energyfj = coffea.hist.Hist("Counts", energyfj_axis)
        return coffea.processor.dict_accumulator(
            {"mlb": mlb, "mw": mw, "mt": mt, "alphal": alphal,
             "alphaj": alphaj, "energyfl": energyfl, "energyfj": energyfj})

    def setup_outputfiller(self, data, dsname):
        return pepper.DummyOutputFiller(self.accumulator.identity())

    def setup_selection(self, data, dsname, is_mc, filler):
        return pepper.Selector(data, data["genWeight"])

    def process_selection(self, selector, dsname, is_mc, filler):
        selector.set_multiple_columns(self.build_gen_columns)
        selector.add_cut("Has gen particles", self.has_gen_particles)

        self.fill_before_selection(
            selector.data, selector.systematics, filler.output)

        super().process_selection(selector, dsname, is_mc, filler)

        self.fill_after_selection(
            selector.final, selector.final_systematics, filler.output)

    @staticmethod
    def sortby(data, field):
        return data[ak.argsort(data["pdgId"], ascending=False)]

    def build_gen_columns(self, data):
        part = data["GenPart"]
        mass = np.asarray(ak.flatten(part.mass))
        # mass is only stored for masses greater than 10 GeV
        # this is a fudge to set the b mass to the right
        # value if not stored
        mass[(mass == 0) & (abs(ak.flatten(part["pdgId"])) == 5)] = 4.18
        mass[(mass == 0) & (abs(ak.flatten(part["pdgId"])) == 13)] = 0.102
        part["mass"] = ak.unflatten(mass, ak.num(part))
        # Fill none in order to not enter masked arrays regime
        part["motherid"] = ak.fill_none(part.parent.pdgId, 0)
        part = part[part["motherid"] != 0]
        part = part[part.hasFlags("isFirstCopy", "isHardProcess")]
        abspdg = abs(part.pdgId)
        sgn = np.sign(part.pdgId)

        cols = {}
        cols["genlepton"] = part[((abspdg == 11) | (abspdg == 13))
                                 & (part.motherid == sgn * -24)]
        cols["genlepton"] = self.sortby(cols["genlepton"], "pdgId")

        cols["genb"] = part[(abspdg == 5) & (part.motherid == sgn * 6)]
        cols["genb"] = self.sortby(cols["genb"], "pdgId")

        cols["genw"] = part[(abspdg == 24) & (part.motherid == sgn * 6)]
        cols["genw"] = self.sortby(cols["genw"], "pdgId")

        cols["gent"] = part[(abspdg == 6)]
        cols["gent"] = self.sortby(cols["gent"], "pdgId")

        return cols

    def has_gen_particles(self, data):
        return ((ak.num(data["genlepton"]) == 2)
                & (ak.num(data["genb"]) == 2)
                & (ak.num(data["genw"]) == 2)
                & (ak.num(data["gent"]) == 2))

    def fill_before_selection(self, data, sys, output):
        lep = data["genlepton"][:, 0]
        antilep = data["genlepton"][:, 1]
        b = data["genb"][:, 0]
        antib = data["genb"][:, 1]
        w = data["genw"]
        t = data["gent"]
        weight = np.asarray(sys["weight"])

        mlbarb = (antilep + b).mass
        mlbbar = (lep + antib).mass
        output["mlb"].fill(mlb=mlbarb, weight=weight)
        output["mlb"].fill(mlb=mlbbar, weight=weight)
        output["mw"].fill(mw=ak.flatten(w.mass), weight=np.repeat(weight, 2))
        output["mt"].fill(mt=ak.flatten(t.mass), weight=np.repeat(weight, 2))

    def match_leptons(self, data):
        recolep = self.sortby(data["Lepton"][:, :2], "pdgId")
        genlep = data["genlepton"]
        is_same_flavor = recolep.pdgId == genlep.pdgId
        is_close = recolep.delta_r(genlep) < 0.3
        is_matched = is_same_flavor & is_close

        return genlep[is_matched], recolep[is_matched]

    def match_jets(self, data):
        recojet = ak.with_name(data["Jet"][["pt", "eta", "phi", "mass"]],
                               "PtEtaPhiMLorentzVector")
        genb = ak.with_name(data["genb"][["pt", "eta", "phi", "mass"]],
                            "PtEtaPhiMLorentzVector")
        genbc, recojetc = ak.unzip(ak.cartesian([genb, recojet], nested=True))
        is_close = genbc.delta_r(recojetc) < 0.3
        is_matched = is_close & (ak.sum(is_close, axis=2) == 1)

        mrecojet = [recojet[is_matched[:, i]] for i in range(2)]
        mrecojet = ak.concatenate(mrecojet, axis=1)
        return genb[ak.any(is_matched, axis=2)], mrecojet

    @staticmethod
    def fill_alpha_energyf(gen, reco, weight, alphahist, energyfhist):
        deltaphi = gen.delta_phi(reco)
        energyf = gen.energy / reco.energy
        # axis=None to remove eventual masking
        alphahist.fill(alpha=ak.flatten(deltaphi, axis=None),
                       weight=np.repeat(weight, ak.num(deltaphi)))
        energyfhist.fill(energyf=ak.flatten(energyf, axis=None),
                         weight=np.repeat(weight, ak.num(energyf)))

    def fill_after_selection(self, data, sys, output):
        weight = np.asarray(sys["weight"])

        genlep, recolep = self.match_leptons(data)
        self.fill_alpha_energyf(
            genlep, recolep, weight, output["alphal"], output["energyfl"])

        genjet, recojet = self.match_jets(data)
        self.fill_alpha_energyf(
            genjet, recojet, weight, output["alphaj"], output["energyfj"])

    def postprocess(self, accumulator):
        return accumulator


parser = ArgumentParser(
    description="Create histograms needed for kinematic reconstruction")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "-o", "--output", help="Name of the output file. Defaults to kinreco.root",
    default="kinreco.root")
parser.add_argument(
    "-c", "--condor", type=int, const=10, nargs="?", metavar="simul_jobs",
    help="Split and submit to HTCondor. By default 10 condor jobs are "
    "submitted. The number can be changed by supplying it to this option")
parser.add_argument(
    "--chunksize", type=int, default=500000, help="Number of events to "
    "process at once. Defaults to 5*10^5")
parser.add_argument(
    "-d", "--debug", action="store_true", help="Only process a small amount "
    "of files to make debugging feasible")
args = parser.parse_args()

if os.path.exists(args.output):
    a = input(f"Overwrite {args.output}? y/n ")
    if a != "y":
        sys.exit(1)

config = pepper.ConfigTTbarLL(args.config)
store = config["store"]


datasets = {
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8":
        config["mc_datasets"]["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]
}
datasets, paths2dsname = expand_datasetdict(datasets, store)
if args.debug:
    print("Processing only one file because of --debug")
    key = next(iter(datasets.keys()))
    datasets = {key: datasets[key][:1]}

os.makedirs(os.path.dirname(os.path.realpath(args.output)), exist_ok=True)

processor = Processor(config)
executor_args = {"schema": NanoAODSchema}
if args.condor is not None:
    executor = coffea.processor.parsl_executor
    # Load parsl config immediately instead of putting it into executor_args
    # to be able to use the same jobs for preprocessing and processing
    print("Spawning jobs. This can take a while")
    parsl.load(pepper.misc.get_parsl_config(args.condor))
else:
    executor = coffea.processor.iterative_executor

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args,
    chunksize=args.chunksize)

with uproot3.recreate(args.output) as f:
    items = ("mlb", "mw", "mt", "alphal", "energyfl", "alphaj", "energyfj")
    for key in items:
        f[key] = export(output[key])
