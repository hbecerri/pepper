import sys
import numpy as np
import coffea
from functools import partial
from collections import namedtuple
import parsl
import json
import logging
from argparse import ArgumentParser
import os

import sympy

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


class DYprocessor(pepper.Processor):

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
        if dsname.startswith("TTTo"):
            selector.set_column(self.gentop, "gent_lc")
            if self.topptweighter is not None:
                self.do_top_pt_reweighting(selector)
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
        selector.set_column(self.z_window, "Z window")

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
        MET = selector.final["MET"].pt.flatten()
        at_least_1btag = self.btag_cut(False, selector.final)

        output = filler.output
        channel = {"ee": selector.final["is_ee"],
                   "em": selector.final["is_em"],
                   "mm": selector.final["is_mm"]}
        Z_w = {"in": Z_window, "out": ~Z_window}
        btags = {"0b": ~at_least_1btag, "1b": at_least_1btag}
        MET_bins = {"0_to_40": (MET < 40),
                    "40_to_70": (MET > 40) & (MET < 70),
                    "70_to_100": (MET > 70) & (MET < 100),
                    "100_to_150": (MET > 100) & (MET < 150),
                    "150_to_inf": (MET > 150)}
        selector.set_column(self.met_0_to_40, "0_to_40",
                            no_callback=True)
        selector.set_column(self.met_40_to_70, "40_to_70",
                            no_callback=True)
        selector.set_column(self.met_70_to_100, "70_to_100",
                            no_callback=True)
        selector.set_column(self.met_100_to_150, "100_to_150",
                            no_callback=True)
        selector.set_column(self.met_150_to_inf, "150_to_inf",
                            no_callback=True)
        if dsname.startswith("DY"):
            weights = selector.final_systematics["weight"].flatten()
            for ch in channel.items():
                for Zw in Z_w.items():
                    for btag in btags.items():
                        for MET_bin in MET_bins.items():
                            self.fill_dy_nums(output, weights, ch, Zw, btag,
                                              MET_bin, dsname)
        elif not is_mc:
            for ch in channel.items():
                for Zw in Z_w.items():
                    for btag in btags.items():
                        for MET_bin in MET_bins.items():
                            self.fill_dy_nums(output,
                                              np.ones(selector.final.size),
                                              ch, Zw, btag, MET_bin, dsname,
                                              True)
        selector.set_column(partial(self.btag_cut, False), "At least 1 btag")
        logger.debug("Selection done")

    def met_0_to_40(self, data):
        ret_arr = np.full(data.size, False)
        MET = data["MET"].pt.flatten()
        ret_arr[data["MET"].counts > 0] = (MET < 40)
        return ret_arr

    def met_40_to_70(self, data):
        ret_arr = np.full(data.size, False)
        MET = data["MET"].pt.flatten()
        ret_arr[data["MET"].counts > 0] = ((MET > 40) & (MET < 70))
        return ret_arr

    def met_70_to_100(self, data):
        ret_arr = np.full(data.size, False)
        MET = data["MET"].pt.flatten()
        ret_arr[data["MET"].counts > 0] = ((MET > 70) & (MET < 100))
        return ret_arr

    def met_100_to_150(self, data):
        ret_arr = np.full(data.size, False)
        MET = data["MET"].pt.flatten()
        ret_arr[data["MET"].counts > 0] = ((MET > 100) & (MET < 150))
        return ret_arr

    def met_150_to_inf(self, data):
        ret_arr = np.full(data.size, False)
        MET = data["MET"].pt.flatten()
        ret_arr[data["MET"].counts > 0] = (MET > 150)
        return ret_arr

    def fill_dy_nums(self, output, weights, ch, z_win, btag,
                     met_bin, dsname, data=False):
        LO_DY_ds = ["DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
                    "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8"]
        NLO_DY_ds = ["DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
                     "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]
        if data:
            name = ("N" + ch[0] + "_" + z_win[0] + "_"
                    + btag[0] + "_" + met_bin[0])
        else:
            name = ("Z" + ch[0] + "_" + z_win[0] + "_"
                    + btag[0] + "_" + met_bin[0])
        weights = weights[ch[1] & z_win[1] & btag[1] & met_bin[1]]
        if data or dsname in LO_DY_ds:
            output["LO_DY_numbers"][name] += weights.sum()
            output["LO_DY_errs"][name] += (weights**2).sum()
        if data or dsname in NLO_DY_ds:
            output["NLO_DY_numbers"][name] += weights.sum()
            output["NLO_DY_errs"][name] += (weights**2).sum()


class SFEquations():
    def __init__(self, met_bins):
        self.regions = ["in_0b_", "out_0b_", "in_1b_"]
        self.MET_bins = met_bins
        data_MC_chs = ["Nee_", "Nem_", "Nmm_", "Zee_", "Zmm_"]
        self.vals_to_sub = [ch + reg + MET for ch in data_MC_chs
                            for reg in self.regions for MET in MET_bins]
        Nee_inc = ""
        Nmm_inc = ""
        for reg in self.regions:
            for MET in MET_bins:
                Nee_inc += " + Nee_" + reg + MET
                Nmm_inc += " + Nmm_" + reg + MET
        Nee_inc = sympy.sympify(Nee_inc)
        Nmm_inc = sympy.sympify(Nmm_inc)
        kee = (Nee_inc / Nmm_inc) ** 0.5
        self.ee_SFs = {}
        self.em_SFs = {}
        self.mm_SFs = {}
        for MET in met_bins:
            Nee_0b_in = sympy.symbols("Nee_in_0b_" + MET)
            Nem_0b_in = sympy.symbols("Nem_in_0b_" + MET)
            Nmm_0b_in = sympy.symbols("Nmm_in_0b_" + MET)
            Nee_0b_out = sympy.symbols("Nee_out_0b_" + MET)
            Nem_0b_out = sympy.symbols("Nem_out_0b_" + MET)
            Nmm_0b_out = sympy.symbols("Nmm_out_0b_" + MET)
            Nee_1b_in = sympy.symbols("Nee_in_1b_" + MET)
            Nem_1b_in = sympy.symbols("Nem_in_1b_" + MET)
            Nmm_1b_in = sympy.symbols("Nmm_in_1b_" + MET)

            Zee_0b_in = sympy.symbols("Zee_in_0b_" + MET)
            Zmm_0b_in = sympy.symbols("Zmm_in_0b_" + MET)
            Zee_0b_out = sympy.symbols("Zee_out_0b_" + MET)
            Zmm_0b_out = sympy.symbols("Zmm_out_0b_" + MET)
            Zee_1b_in = sympy.symbols("Zee_in_1b_" + MET)
            Zmm_1b_in = sympy.symbols("Zmm_in_1b_" + MET)

            Ree_0b_data = ((Nee_0b_in - 0.5 * kee * Nem_0b_in)
                           / (Nee_0b_out - 0.5 * kee * Nem_0b_out))
            Ree_0b_MC = Zee_0b_in / Zee_0b_out
            Rmm_0b_data = ((Nmm_0b_in - 0.5 * Nem_0b_in / kee)
                           / (Nmm_0b_out - 0.5 * Nem_0b_out / kee))
            Rmm_0b_MC = Zmm_0b_in / Zmm_0b_out
            self.ee_SFs[MET] = ((Nee_1b_in - 0.5 * kee * Nem_1b_in) / Zee_1b_in
                                * Ree_0b_MC / Ree_0b_data)
            self.mm_SFs[MET] = ((Nmm_1b_in - 0.5 * Nem_1b_in / kee) / Zmm_1b_in
                                * Rmm_0b_MC / Rmm_0b_data)
            self.em_SFs[MET] = (self.ee_SFs[MET] * self.mm_SFs[MET]) ** 0.5

    def evaluate(self, values):
        out_dict = {}
        subs = {val: values[val] for val in self.vals_to_sub}
        for MET in self.MET_bins:
            out_dict["is_ee" + MET] = sympy.lambdify(list(subs.keys()),
                                                     self.ee_SFs[MET])(**subs)
            out_dict["is_em" + MET] = sympy.lambdify(list(subs.keys()),
                                                     self.em_SFs[MET])(**subs)
            out_dict["is_mm" + MET] = sympy.lambdify(list(subs.keys()),
                                                     self.mm_SFs[MET])(**subs)
        return out_dict

    def calculate_errs(self):
        self.ee_SF_errs = {MET: 0 for MET in self.MET_bins}
        self.em_SF_errs = {MET: 0 for MET in self.MET_bins}
        self.mm_SF_errs = {MET: 0 for MET in self.MET_bins}
        err_dicts = [(self.ee_SF_errs, self.ee_SFs),
                     (self.em_SF_errs, self.em_SFs),
                     (self.mm_SF_errs, self.mm_SFs)]
        for MET in self.MET_bins:
            for err_dict, SFs in err_dicts:
                for val in self.vals_to_sub:
                    err_dict[MET] += ((sympy.diff(SFs[MET], val)) ** 2
                                      * sympy.symbols(val+"_err"))
                err_dict[MET] = err_dict[MET] ** 0.5

    def evaluate_errs(self, values, errs):
        out_dict = {}
        subs = {val: values[val] for val in self.vals_to_sub}
        subs.update({val + "_err": errs[val] for val in self.vals_to_sub})
        for MET in self.MET_bins:
            out_dict["is_ee" + MET] = sympy.lambdify(
                list(subs.keys()), self.ee_SF_errs[MET])(**subs)
            out_dict["is_em" + MET] = sympy.lambdify(
                list(subs.keys()), self.em_SF_errs[MET])(**subs)
            out_dict["is_mm" + MET] = sympy.lambdify(
                list(subs.keys()), self.mm_SF_errs[MET])(**subs)
        return out_dict


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
    "-r", "--rerun", action="store_true", help="Rerun over all the datasets: "
    "otherwise will just load numbers in from json")
parser.add_argument(
    "-t", "--test", action="store_true", help="Run over all histograms and "
    "produce plots inside and outside the z window")
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

if args.rerun or args.test:
    output = coffea.processor.run_uproot_job(
        datasets, "Events", processor, executor, executor_args,
        chunksize=args.chunksize)
else:
    with open("DY_sfs.json", "r") as f:
        output = json.load(f)

print(output)
nums = {"LO_DY_numbers": output["LO_DY_numbers"],
        "LO_DY_errs": output["LO_DY_errs"],
        "LO_SFs": {},
        "LO_SF_errs": {},
        "NLO_DY_numbers": output["NLO_DY_numbers"],
        "NLO_DY_errs": output["NLO_DY_errs"],
        "NLO_SFs": {},
        "NLO_SF_errs": {}}

sf_eq = SFEquations(MET_bins)
nums["LO_SFs"] = sf_eq.evaluate(output["LO_DY_numbers"])
nums["NLO_SFs"] = sf_eq.evaluate(output["NLO_DY_numbers"])
sf_eq.calculate_errs()
nums["LO_SF_errs"] = sf_eq.evaluate_errs(output["LO_DY_numbers"],
                                         output["LO_DY_errs"])
nums["NLO_SF_errs"] = sf_eq.evaluate_errs(output["NLO_DY_numbers"],
                                          output["NLO_DY_errs"])

with open("DY_sfs.json", "w") as f:
    json.dump(nums, f, indent=4)

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
    coffea.util.save(output["cutflows"], "cutflows.coffea")
