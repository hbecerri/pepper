import os
import sys
import numpy as np
import awkward
import uproot
import coffea
from coffea.analysis_objects import JaggedCandidateArray as Jca
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict, namedtuple
import shutil
import parsl
import json
import logging
from argparse import ArgumentParser

import pepper
from pepper.misc import jcafromjagged, sortby
from pepper.datasets import expand_datasetdict

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 7):
    VariationArg = namedtuple(
        "VariationArgs", ["name", "junc", "jer", "met"],
        defaults=(None, "central", "central"))
else:
    # defaults in nampedtuple were introduced in python 3.7
    # As soon as CMSSW offers 3.7 or newer, remove this
    class VariationArg(
            namedtuple("VariationArg", ["name", "junc", "jer", "met"])):
        def __new__(cls, name, junc=None, jer="central", met="central"):
            return cls.__bases__[0].__new__(cls, name, junc, jer, met)

class DY_processor(pepper.Processor):

    @property
    def accumulator(self):
        self._accumulator = coffea.processor.dict_accumulator({
            "hists": coffea.processor.dict_accumulator(),
            "cutflows": coffea.processor.dict_accumulator(),
            "LO_DY_numbers": coffea.processor.defaultdict_accumulator(int),
            "LO_DY_errs": coffea.processor.defaultdict_accumulator(int),
            "NLO_DY_numbers": coffea.processor.defaultdict_accumulator(int),
            "NLO_DY_errs": coffea.processor.defaultdict_accumulator(int)
        })
        return self._accumulator

    def process_selection(self, selector, dsname, is_mc, filler):
        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)
        if is_mc:
            self.add_crosssection_scale(selector, dsname)

        selector.add_cut(partial(self.good_lumimask, is_mc), "Lumi")

        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.trigger_paths, self.trigger_order)
        selector.add_cut(partial(
            self.passing_trigger, pos_triggers, neg_triggers), "Trigger")

        selector.add_cut(partial(self.met_filters, is_mc), "MET filters")

        selector.set_column(self.build_lepton_column, "Lepton")
        # Wait with hists filling after channel masks are available
        selector.add_cut(partial(self.lepton_pair, is_mc), "At least 2 leps",
                         no_callback=True)
        filler.channels = ("is_ee", "is_em", "is_mm")
        selector.set_multiple_columns(self.channel_masks)
        selector.set_column(self.mll, "mll")
        selector.set_column(self.dilep_pt, "dilep_pt")

        selector.freeze_selection()

        selector.add_cut(self.opposite_sign_lepton_pair, "Opposite sign")
        selector.add_cut(partial(self.no_additional_leptons, is_mc),
                         "No add. leps")
        selector.add_cut(self.channel_trigger_matching, "Chn. trig. match")
        selector.add_cut(self.lep_pt_requirement, "Req lep pT")
        selector.add_cut(self.good_mll, "M_ll")

        selector.set_column(self.build_jet_column, "Jet")
        selector.set_column(partial(self.build_met_column,
                                    variation=VariationArg(None).met), "MET")
        selector.add_cut(self.has_jets, "#Jets >= %d"
                         % self.config["num_jets_atleast"])
        if (self.config["hem_cut_if_ele"] or self.config["hem_cut_if_muon"]
                or self.config["hem_cut_if_jet"]):
            selector.add_cut(self.hem_cut, "HEM cut")
        selector.add_cut(self.jet_pt_requirement, "Jet pt req")

        mll = selector.final["mll"]
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        Z_window = (mll >= m_min) & (mll <= m_max)
        output = filler.output
        if dsname.startswith("DY"):
            weights = selector.final_systematics["weight"].flatten()
            self.fill_DY_nums(output, weights[selector.final["is_ee"] & Z_window], "Zee_in", dsname)
            self.fill_DY_nums(output, weights[selector.final["is_mm"] & Z_window], "Zmm_in", dsname)
            self.fill_DY_nums(output, weights[selector.final["is_ee"] & ~Z_window], "Zee_out", dsname)
            self.fill_DY_nums(output, weights[selector.final["is_mm"] & ~Z_window], "Zmm_out", dsname)
        elif not is_mc:
            self.fill_DY_nums(output, np.ones((selector.final["is_ee"] & Z_window).sum()), "Nee_in", dsname, True)
            self.fill_DY_nums(output, np.ones((selector.final["is_em"] & Z_window).sum()), "Nem_in", dsname, True)
            self.fill_DY_nums(output, np.ones((selector.final["is_mm"] & Z_window).sum()), "Nmm_in", dsname, True)

        logger.debug("Selection done")

    def fill_DY_nums(self, output, weights, name, dsname, data=False):
        LO_DY_ds = ["DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8"]
        NLO_DY_ds = ["DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]
        print(dsname)
        if data or dsname in LO_DY_ds:
            print(output)
            output["LO_DY_numbers"][name] += weights.sum()
            output["LO_DY_errs"][name] += (weights**2).sum()
        if data or dsname in NLO_DY_ds:
            output["NLO_DY_numbers"][name] += weights.sum()
            output["NLO_DY_errs"][name] += (weights**2).sum()

parser = ArgumentParser(description="Select events from nanoAODs")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "--eventdir", help="Path to event destination output directory. If not "
    "specified, no events will be saved")
parser.add_argument(
    "--histdir", help="Path to the histogram destination output directory. By "
    "default, ./hists will be used.", default="./hists")
parser.add_argument(
    "--dataset", nargs=2, action="append", metavar=("name", "path"),
    help="Can be specified multiple times. Ignore datasets given in "
    "config and instead process these")
parser.add_argument(
    "-c", "--condor", type=int, const=10, nargs="?", metavar="simul_jobs",
    help="Split and submit to HTCondor. By default 10 condor jobs are "
    "submitted. The number can be changed by supplying it to this option"
)
parser.add_argument(
    "--chunksize", type=int, default=500000, help="Number of events to "
    "process at once. Defaults to 5*10^5")
parser.add_argument(
    "--mc", action="store_true", help="Only process MC files")
parser.add_argument(
    "-d", "--debug", action="store_true", help="Enable debug messages and "
    "only process a small amount of files to make debugging feasible")
parser.add_argument(
    "-p", "--parsl_config", help="Path to a json specifying the condor_init "
    "and condor_configs to be used with parsl. If not specified, the default "
    "settings in misc.py will be used")
parser.add_argument(
    "--DY", action="store_true", help="Run on the processor for the DY CR")
args = parser.parse_args()

logger = logging.getLogger("pepper")
logger.addHandler(logging.StreamHandler())
if args.debug:
    logger.setLevel(logging.DEBUG)

config = pepper.Config(args.config)
store = config["store"]


datasets = {}
if args.dataset is None:
    datasets = {}
    if not args.mc:
        datasets.update(config["exp_datasets"])
    duplicate = set(datasets.keys()) & set(config["mc_datasets"])
    if len(duplicate) > 0:
        print("Got duplicate dataset names: {}".format(", ".join(duplicate)))
        exit(1)
    DY_ds = ["DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
             "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",
             "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]
    mc_ds = {key: val for key, val in config["mc_datasets"].items() if key in DY_ds}
    datasets.update(mc_ds)
else:
    datasets = {}
    for dataset in args.dataset:
        if dataset[0] in datasets:
            datasets[dataset[0]].append(dataset[1])
        else:
            datasets[dataset[0]] = [dataset[1]]
if not config["compute_systematics"]:
    for sysds in config["dataset_for_systematics"].keys():
        if sysds in datasets:
            del datasets[sysds]

requested_datasets = datasets.keys()
datasets, paths2dsname = pepper.datasets.expand_datasetdict(datasets, store)
if args.dataset is None:
    missing_datasets = requested_datasets - datasets.keys()
    if len(missing_datasets) > 0:
        print("Could not find files for: " + ", ".join(missing_datasets))
        exit(1)
    num_files = len(paths2dsname)
    num_mc_files = sum(len(datasets[dsname])
                       for dsname in config["mc_datasets"].keys()
                       if dsname in requested_datasets)

    print("Got a total of {} files of which {} are MC".format(num_files,
                                                              num_mc_files))

if args.debug:
    print("Processing only one file per dataset because of --debug")
    datasets = {key: [val[0]] for key, val in datasets.items()}

if len(datasets) == 0:
    print("No datasets found")
    exit(1)

# Plotting hists is pointless since we're not running over the full mc:
config["hists"] = {}

processor = DY_processor(config, args.eventdir)

if args.condor is not None:
    executor = coffea.processor.parsl_executor
    # Load parsl config immediately instead of putting it into executor_args
    # to be able to use the same jobs for preprocessing and processing
    print("Spawning jobs. This can take a while")
    if args.parsl_config is not None:
        with open(args.parsl_config) as f:
            parsl_config = json.load(f)
        parsl_config = pepper.misc.get_parsl_config(
            args.condor,
            condor_submit=parsl_config["condor_config"],
            condor_init=parsl_config["condor_init"])
    else:
        parsl_config = pepper.misc.get_parsl_config(args.condor)
    parsl.load(parsl_config)
    executor_args = {}
else:
    if args.parsl_config is not None:
        print("Ignoring parsl_config because condor is not specified")
    executor = coffea.processor.iterative_executor
    executor_args = {}

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args,
    chunksize=args.chunksize)

nums = {"LO_DY_numbers": output["LO_DY_numbers"],
        "LO_DY_errs": output["LO_DY_errs"],
        "LO_SFs": {},
        "NLO_DY_numbers": output["NLO_DY_numbers"],
        "NLO_DY_errs": output["NLO_DY_errs"],
        "NLO_SFs": {}}

Nee_in = output["LO_DY_numbers"]["Nee_in"]
Nem_in = output["LO_DY_numbers"]["Nem_in"]
Nmm_in = output["LO_DY_numbers"]["Nmm_in"]
Zee_in = output["LO_DY_numbers"]["Zee_in"]
Zmm_in = output["LO_DY_numbers"]["Zmm_in"]
kee = np.sqrt(Nee_in/Nmm_in)

nums["LO_SFs"]["is_ee"] = (Nee_in - 0.5 * kee * Nem_in) / Zee_in
nums["LO_SFs"]["is_mm"] = (Nmm_in - 0.5 * Nem_in / kee) / Zmm_in
nums["LO_SFs"]["is_em"] = np.sqrt(nums["LO_SFs"]["is_ee"] * nums["LO_SFs"]["is_mm"])

Nee_in = output["NLO_DY_numbers"]["Nee_in"]
Nem_in = output["NLO_DY_numbers"]["Nem_in"]
Nmm_in = output["NLO_DY_numbers"]["Nmm_in"]
Zee_in = output["NLO_DY_numbers"]["Zee_in"]
Zmm_in = output["NLO_DY_numbers"]["Zmm_in"]
kee = np.sqrt(Nee_in/Nmm_in)

nums["NLO_SFs"]["is_ee"] = (Nee_in - 0.5 * kee * Nem_in) / Zee_in
nums["NLO_SFs"]["is_mm"] = (Nmm_in - 0.5 * Nem_in / kee) / Zmm_in
nums["NLO_SFs"]["is_em"] = np.sqrt(nums["NLO_SFs"]["is_ee"] * nums["NLO_SFs"]["is_mm"])
with open("DY_sfs.json", "w") as f:
    json.dump(nums, f, indent=4)
