import numpy as np
import coffea
import json
import logging
import argparse
from copy import copy

import sympy


logger = logging.getLogger(__name__)


class SFEquations():
    def __init__(self, met_bins, config):
        self.config = config
        self.regions = ["in_0b_", "out_0b_", "in_1b_"]
        self.MET_bins = met_bins
        chs = ["ee_", "em_", "mm_"]
        self.symbols = [ch + reg + MET for ch in chs
                        for reg in self.regions for MET in met_bins]
        Nee_inc = ""
        Nmm_inc = ""
        for reg in self.regions:
            for MET in met_bins:
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

    def evaluate(self, values, cut):
        ret_vals = [[], [], []]
        subs = {"N" + sym: sum([values[sym][ds][cut]
                                for ds in self.config["Data"]])
                for sym in self.symbols}
        subs.update({"Z" + sym: sum([values[sym][ds][cut]
                                     for ds in self.config["DY_MC"]])
                     for sym in self.symbols})
        for MET in self.MET_bins:
            ret_vals[0].append(sympy.lambdify(list(subs.keys()),
                                              self.ee_SFs[MET])(**subs))
            ret_vals[1].append(sympy.lambdify(list(subs.keys()),
                                              self.em_SFs[MET])(**subs))
            ret_vals[2].append(sympy.lambdify(list(subs.keys()),
                                              self.mm_SFs[MET])(**subs))
        return ret_vals

    def calculate_errs(self):
        self.ee_SF_errs = {MET: 0 for MET in self.MET_bins}
        self.em_SF_errs = {MET: 0 for MET in self.MET_bins}
        self.mm_SF_errs = {MET: 0 for MET in self.MET_bins}
        err_dicts = [(self.ee_SF_errs, self.ee_SFs),
                     (self.em_SF_errs, self.em_SFs),
                     (self.mm_SF_errs, self.mm_SFs)]
        for MET in self.MET_bins:
            for err_dict, SFs in err_dicts:
                for sym in self.symbols:
                    err_dict[MET] += ((sympy.diff(SFs[MET], "N" + sym)) ** 2
                                      * sympy.symbols("N" + sym+"_err"))
                    err_dict[MET] += ((sympy.diff(SFs[MET], "Z" + sym)) ** 2
                                      * sympy.symbols("Z" + sym+"_err"))
                err_dict[MET] = err_dict[MET] ** 0.5

    def evaluate_errs(self, values, errs, cut):
        ret_vals = [[], [], []]
        subs = {"N" + sym: sum([values[sym][ds][cut]
                                for ds in self.config["Data"]])
                for sym in self.symbols}
        subs.update({"Z" + sym: sum([values[sym][ds][cut]
                                     for ds in self.config["DY_MC"]])
                     for sym in self.symbols})
        subs.update({"N" + sym + "_err":
                     sum([errs[sym][ds][cut] for ds in self.config["Data"]])
                     for sym in self.symbols})
        subs.update({"Z" + sym + "_err":
                     sum([errs[sym][ds][cut] for ds in self.config["DY_MC"]])
                     for sym in self.symbols})
        for MET in self.MET_bins:
            ret_vals[0].append(sympy.lambdify(
                list(subs.keys()), self.ee_SF_errs[MET])(**subs))
            ret_vals[1].append(sympy.lambdify(
                list(subs.keys()), self.em_SF_errs[MET])(**subs))
            ret_vals[2].append(sympy.lambdify(
                list(subs.keys()), self.mm_SF_errs[MET])(**subs))
        return ret_vals


def rebin_met(cutflows, rebin_dict):
    cutflow = cutflows["cutflow"]
    errors = cutflows["errs"]
    met_bins = [k[9:] for k in cutflow if k.startswith("ee_in_0b_")]
    if rebin_dict == "Inclusive":
        rebin_dict = {"Inclusive": copy(met_bins)}
    regions = ["in_0b_", "out_0b_", "in_1b_"]
    chs = ["ee_", "em_", "mm_"]
    for new_bin, old_bins in rebin_dict.items():
        for old_bin in old_bins:
            if old_bin not in met_bins:
                raise ValueError(f"Bin {old_bin} not present in met_bins: "
                                 f"{met_bins}")
            met_bins.remove(old_bin)
        met_bins.append(new_bin)
        for ch in chs:
            for reg in regions:
                cutflow[ch+reg+new_bin] = sum(
                    [cutflow[ch+reg+old_bin] for old_bin in old_bins], {})
                errors[ch+reg+new_bin] = sum(
                    [errors[ch+reg+old_bin] for old_bin in old_bins], {})
    return met_bins, cutflow, errors


class MultiInputOptArg(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values:
            setattr(namespace, self.dest, values)
        else:
            setattr(namespace, self.dest, self.const)


parser = argparse.ArgumentParser(description="Calculate DY scale factors from "
                                 "DY yields produced by produce_DY_numbers.py")
parser.add_argument(
    "config", help="Path to a config containing labels for which datsets "
    "correspond to DY, and which to data, as well as information about the "
    "binning to use, which should be either 'Inclusive' or a dict with bins "
    "(with names in the format <lower_lim>_to_<upper_lim> as keys, and lists "
    "of the corresponding MET bins as values")
parser.add_argument("cutflow", help="Path to a cutflow containing the numbers "
                    "required to calculate the SFs")
parser.add_argument("output", help="Path to the output file")
parser.add_argument(
    "-c", "--cutname", default="Jet pt req", help="Cut for which scale "
    "factors should be calculated- default 'Jet pt req'")
parser.add_argument(
    "-v", "--variation", action=MultiInputOptArg, nargs="*", const=["Reco"],
    help="Alternative working point at which to calculate scale factors to "
    "estimate systematic error. May give either one argument (which will "
    "be interpreted as another cut in the same cutflow), or two, specifying "
    "a different cutflow file, and a cut in that cutflow. Default: 'Reco'")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)
cutflows = coffea.util.load(args.cutflow)
met_bins, cutflow, errors = rebin_met(cutflows, config["rebin"])

if config["rebin"] == "Inclusive":
    met_edges = [0, 10000]  # Could altenatively use inf as upper limit, but
    # if we've got events with more than 10TeV MET, we've got bigger problems
else:
    met_edges = [int(k.split("_")[0]) for k in config["rebin"].keys()]
    met_edges.append(10000)
out_dict = {"bins": {"met": met_edges, "channel": [0, 1, 2]}}

sf_eq = SFEquations(met_bins, config)
sfs = sf_eq.evaluate(cutflow, args.cutname)
sf_eq.calculate_errs()
stat_errs = sf_eq.evaluate_errs(cutflow, errors, args.cutname)

if args.variation:
    if len(args.variation) > 2:
        raise ValueError("Too many arguments supplied to variation (max 2)")
    elif len(args.variation) == 2:
        with open(args.variation[0], "r") as f:
            var_cutflows = json.load(f)
        _, var_cutflow, _ = rebin_met(var_cutflows, config["rebin"])
        cutname = args.variation[1]
    else:
        var_cutflow = cutflow
        cutname = args.variation[0]
    var_sfs = sf_eq.evaluate(var_cutflow, cutname)
    sys_errs = [[np.abs(sfs[ch_i][met_i] - var_sfs[ch_i][met_i])
                 for met_i in range(len(sfs[ch_i]))] for ch_i in range(3)]
    tot_errs = [
        [np.sqrt(sys_errs[ch_i][met_i] ** 2 + stat_errs[ch_i][met_i] ** 2)
         for met_i in range(len(sfs[ch_i]))] for ch_i in range(3)]
else:
    tot_errs = stat_errs

out_dict["factors"] = sfs
out_dict["factors_up"] = [
    [sfs[ch_i][met_i] + tot_errs[ch_i][met_i]
     for met_i in range(len(sfs[ch_i]))] for ch_i in range(3)]
out_dict["factors_down"] = [
    [sfs[ch_i][met_i] - tot_errs[ch_i][met_i]
     for met_i in range(len(sfs[ch_i]))] for ch_i in range(3)]

with open(args.output, "w+") as f:
    json.dump(out_dict, f, indent=4)
