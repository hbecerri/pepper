import sys
from collections import namedtuple
import json
import logging
from argparse import ArgumentParser
import os
from functools import partial

import numpy as np
import parsl
import awkward as ak
import coffea
from coffea.nanoevents import NanoAODSchema

import pepper


MET_bins = ["0_to_40", "40_to_70", "70_to_100", "100_to_150", "150_to_inf"]

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


class DYOutputFiller(pepper.OutputFiller):
    def fill_cutflows(self, data, systematics, cut, done_steps):
        if systematics is not None:
            weight = systematics["weight"]
        else:
            weight = ak.Array(np.ones(len(data)))
        logger.info("Filling cutflow. Current event count: "
                    + str(ak.sum(weight)))
        self.fill_accumulator(self.output["cutflows"], cut, data, weight)
        if systematics is not None:
            weight = systematics["weight"] ** 2
        else:
            weight = ak.Array(np.ones(len(data)))
        self.fill_accumulator(self.output["cutflow_errs"], cut, data, weight)

    def fill_accumulator(self, accumulator, cut, data, weight):
        if "all" not in accumulator:
            accumulator["all"] = coffea.processor.defaultdict_accumulator(
                partial(coffea.processor.defaultdict_accumulator, int))
        if cut not in accumulator["all"][self.dsname]:
            accumulator["all"][self.dsname][cut] = ak.sum(weight)
        for ch in self.channels:
            if ch not in accumulator:
                accumulator[ch] = coffea.processor.defaultdict_accumulator(
                    partial(coffea.processor.defaultdict_accumulator, int))
            if cut not in accumulator[ch][self.dsname]:
                accumulator[ch][self.dsname][cut] = ak.sum(weight[data[ch]])


class DYprocessor(pepper.ProcessorTTbarLL):
    @property
    def accumulator(self):
        self._accumulator = coffea.processor.dict_accumulator({
            "hists": coffea.processor.dict_accumulator(),
            "cutflows": coffea.processor.dict_accumulator(),
            "cutflow_errs": coffea.processor.dict_accumulator()
        })
        return self._accumulator

    def setup_outputfiller(self, dsname, is_mc):
        output = self.accumulator.identity()
        sys_enabled = self.config["compute_systematics"]

        if dsname in self.config["dataset_for_systematics"]:
            dsname_in_hist = self.config["dataset_for_systematics"][dsname][0]
            sys_overwrite = self.config["dataset_for_systematics"][dsname][1]
        else:
            dsname_in_hist = dsname
            sys_overwrite = None

        if "cuts_to_histogram" in self.config:
            cuts_to_histogram = self.config["cuts_to_histogram"]
        else:
            cuts_to_histogram = None

        filler = DYOutputFiller(
            output, self.hists, is_mc, dsname, dsname_in_hist, sys_enabled,
            sys_overwrite=sys_overwrite, copy_nominal=self.copy_nominal,
            cuts_to_histogram=cuts_to_histogram)

        return filler

    def z_window(self, data):
        # Don't apply Z window cut, as we'll add columns inside and
        # outside of it later
        return np.full(len(data), True)

    def drellyan_sf_columns(self, filler, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        Z_window = (data["mll"] >= m_min) & (data["mll"] <= m_max)
        MET = data["MET"].pt
        channel = {"ee": data["is_ee"],
                   "em": data["is_em"],
                   "mm": data["is_mm"]}
        Z_w = {"in": Z_window, "out": ~Z_window}
        btags = {"0b": (ak.sum(data["Jet"]["btagged"]) == 0),
                 "1b": (ak.sum(data["Jet"]["btagged"]) > 0)}
        MET_bins = {"0_to_40": (MET < 40),
                    "40_to_70": (MET > 40) & (MET < 70),
                    "70_to_100": (MET > 70) & (MET < 100),
                    "100_to_160": (MET > 100) & (MET < 160),
                    "160_to_inf": (MET > 160)}
        new_chs = {}
        for ch in channel.items():
            for Zw in Z_w.items():
                for btag in btags.items():
                    for MET_bin in MET_bins.items():
                        new_chs[ch[0]+"_"+Zw[0]+"_"+btag[0]+"_"+MET_bin[0]] = (
                            ch[1] & Zw[1] & btag[1] & MET_bin[1])
        filler.channels = new_chs.keys()
        return new_chs

    def btag_cut(self, is_mc, data):
        return np.full(len(data), True)


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
    "submitted. The number can be changed by supplying it to this option")
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
    "-r", "--retries", type=int, help="Number of times to retry if there is "
    "exception in an HTCondor job. If not given, retry infinitely.")
parser.add_argument(
    "-t", "--test", action="store_true", help="Run over all histograms and "
    "produce plots inside and outside the z window (otherwise will only run "
    "over data and DY samples)")
args = parser.parse_args()

logger = logging.getLogger("pepper")
logger.addHandler(logging.StreamHandler())
if args.debug:
    logger.setLevel(logging.DEBUG)

config = pepper.ConfigTTbarLL(args.config)
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
    if args.test:
        mc_ds = config["mc_datasets"]
    else:
        mc_ds = {key: val for key, val in config["mc_datasets"].items()
                 if key in DY_ds}
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

# Create histdir and in case of errors, raise them now (before processing)
os.makedirs(args.histdir, exist_ok=True)

if args.test:
    for hist in config["hists"].values():
        if hasattr(hist, "cats"):
            hist.cats.extend([coffea.hist.Cat("Z_window", "Z_window"),
                              coffea.hist.Cat("MET_bin", "MET_bin"),
                              coffea.hist.Cat("At least 1 btag",
                                              "At least 1 btag")])
            hist.cat_fill_methods["Z_window"] = {"in": ["Z window",
                                                        {"function": "not"}],
                                                 "out": ["Z window"]}
            hist.cat_fill_methods["MET_bin"] = {MET_bin: [MET_bin]
                                                for MET_bin in MET_bins}
            hist.cat_fill_methods["At least 1 btag"] = \
                {"0b": ["At least 1 btag", {"function": "not"}],
                 "1b": ["At least 1 btag"]}
        else:
            hist.cats = [coffea.hist.Cat("Z_window", "Z_window"),
                         coffea.hist.Cat("MET_bin", "MET_bin"),
                         coffea.hist.Cat("At least 1 btag", "At least 1 btag")]
            hist.cat_fill_methods = \
                {"Z_window": {"in": ["Z window"],
                              "out": ["Z window", {"function": "not"}]},
                 "MET_bin": {MET_bin: [MET_bin] for MET_bin in MET_bins},
                 "At least 1 btag": {"0b": ["At least 1 btag",
                                            {"function": "not"}],
                                     "1b": ["At least 1 btag"]}}
else:
    # Plotting hists is pointless if we're not running over the full mc:
    config["hists"] = {}

processor = DYprocessor(config, args.eventdir)
executor_args = {"schema": NanoAODSchema, "align_clusters": True}
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
            condor_init=parsl_config["condor_init"],
            retries=args.retries)
    else:
        parsl_config = pepper.misc.get_parsl_config(
            args.condor, retries=args.retries)
    parsl.load(parsl_config)
else:
    if args.parsl_config is not None:
        print("Ignoring parsl_config because condor is not specified")
    executor = coffea.processor.iterative_executor

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args,
    chunksize=args.chunksize)


if args.test:
    # Save histograms with a hist.json describing the hist files
    hists = output["hists"]
    jsonname = "hists.json"
    hists_forjson = {}
    for key, hist in hists.items():
        if hist.values() == {}:
            continue
        cuts = next(iter(output["cutflows"]["all"].values())).keys()
        cutnum = list(cuts).index(key[0])
        fname = "Cut {:03} {}.coffea".format(cutnum, "_".join(key))
        fname = fname.replace("/", "")
        coffea.util.save(hist, os.path.join(args.histdir, fname))
        hists_forjson[key] = fname
    with open(os.path.join(args.histdir, jsonname), "a+") as f:
        try:
            hists_injson = {tuple(k): v for k, v in zip(*json.load(f))}
        except json.decoder.JSONDecodeError:
            hists_injson = {}
    hists_injson.update(hists_forjson)
    with open(os.path.join(args.histdir, jsonname), "w") as f:
        json.dump([[tuple(k) for k in hists_injson.keys()],
                   list(hists_injson.values())], f, indent=4)

    # Save cutflows
    coffea.util.save({"cutflow": output["cutflows"],
                      "errs": output["cutflow_errs"]},
                     os.path.join(args.histdir, "cutflows.coffea"))
else:
    coffea.util.save({"cutflow": output["cutflows"],
                      "errs": output["cutflow_errs"]},
                     "DY_SF_cutflows.coffea")
print("Done!")
