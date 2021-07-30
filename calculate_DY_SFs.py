import numpy as np
import hjson
import logging
import argparse
import os

import sympy


logger = logging.getLogger(__name__)


class SFEquations():
    def __init__(self, config):
        self.config = config
        self.regions = ["in_0b", "out_0b", "in_1b"]
        self.bins = config["bins"]
        if config["bins"] is None:
            self.bins = [""]
        else:
            self.bins = [
                "_" + str(config["bins"][i]) + "_" + str(config["bins"][i+1])
                for i in range(len(config["bins"]) - 1)]
        chs = ["ee_", "em_", "mm_"]
        self.symbols = [ch + reg + b for ch in chs
                        for reg in self.regions for b in self.bins]
        Nee_inc = ""
        Nmm_inc = ""
        for reg in self.regions:
            for b in self.bins:
                Nee_inc += " + Nee_" + reg + b
                Nmm_inc += " + Nmm_" + reg + b
        Nee_inc = sympy.sympify(Nee_inc)
        Nmm_inc = sympy.sympify(Nmm_inc)
        kee = (Nee_inc / Nmm_inc) ** 0.5
        self.ee_SFs = {}
        self.em_SFs = {}
        self.mm_SFs = {}
        for b in self.bins:
            Nee_0b_in = sympy.symbols("Nee_in_0b" + b)
            Nem_0b_in = sympy.symbols("Nem_in_0b" + b)
            Nmm_0b_in = sympy.symbols("Nmm_in_0b" + b)
            Nee_0b_out = sympy.symbols("Nee_out_0b" + b)
            Nem_0b_out = sympy.symbols("Nem_out_0b" + b)
            Nmm_0b_out = sympy.symbols("Nmm_out_0b" + b)
            Nee_1b_in = sympy.symbols("Nee_in_1b" + b)
            Nem_1b_in = sympy.symbols("Nem_in_1b" + b)
            Nmm_1b_in = sympy.symbols("Nmm_in_1b" + b)

            Zee_0b_in = sympy.symbols("Zee_in_0b" + b)
            Zmm_0b_in = sympy.symbols("Zmm_in_0b" + b)
            Zee_0b_out = sympy.symbols("Zee_out_0b" + b)
            Zmm_0b_out = sympy.symbols("Zmm_out_0b" + b)
            Zee_1b_in = sympy.symbols("Zee_in_1b" + b)
            Zmm_1b_in = sympy.symbols("Zmm_in_1b" + b)

            Ree_0b_data = ((Nee_0b_in - 0.5 * kee * Nem_0b_in)
                           / (Nee_0b_out - 0.5 * kee * Nem_0b_out))
            Ree_0b_MC = Zee_0b_in / Zee_0b_out
            Rmm_0b_data = ((Nmm_0b_in - 0.5 * Nem_0b_in / kee)
                           / (Nmm_0b_out - 0.5 * Nem_0b_out / kee))
            Rmm_0b_MC = Zmm_0b_in / Zmm_0b_out
            self.ee_SFs[b] = ((Nee_1b_in - 0.5 * kee * Nem_1b_in) / Zee_1b_in
                              * Ree_0b_MC / Ree_0b_data)
            self.mm_SFs[b] = ((Nmm_1b_in - 0.5 * Nem_1b_in / kee) / Zmm_1b_in
                              * Rmm_0b_MC / Rmm_0b_data)
            self.em_SFs[b] = (self.ee_SFs[b] * self.mm_SFs[b]) ** 0.5

    def evaluate(self, values, cut):
        ret_vals = [[], [], []]
        subs = {"N" + sym: sum([values[sym][ds][cut]
                                for ds in self.config["Data"]])
                for sym in self.symbols}
        subs.update({"Z" + sym: sum([values[sym][ds][cut]
                                     for ds in self.config["DY_MC"]])
                     for sym in self.symbols})
        for b in self.bins:
            ret_vals[0].append(sympy.lambdify(list(subs.keys()),
                                              self.ee_SFs[b])(**subs))
            ret_vals[1].append(sympy.lambdify(list(subs.keys()),
                                              self.em_SFs[b])(**subs))
            ret_vals[2].append(sympy.lambdify(list(subs.keys()),
                                              self.mm_SFs[b])(**subs))
        return ret_vals

    def calculate_errs(self):
        self.ee_SF_errs = {b: 0 for b in self.bins}
        self.em_SF_errs = {b: 0 for b in self.bins}
        self.mm_SF_errs = {b: 0 for b in self.bins}
        err_dicts = [(self.ee_SF_errs, self.ee_SFs),
                     (self.em_SF_errs, self.em_SFs),
                     (self.mm_SF_errs, self.mm_SFs)]
        for b in self.bins:
            for err_dict, SFs in err_dicts:
                for sym in self.symbols:
                    err_dict[b] += ((sympy.diff(SFs[b], "N" + sym)) ** 2
                                    * sympy.symbols("N" + sym+"_err"))
                    err_dict[b] += ((sympy.diff(SFs[b], "Z" + sym)) ** 2
                                    * sympy.symbols("Z" + sym+"_err"))
                err_dict[b] = err_dict[b] ** 0.5

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
        for b in self.bins:
            ret_vals[0].append(sympy.lambdify(
                list(subs.keys()), self.ee_SF_errs[b])(**subs))
            ret_vals[1].append(sympy.lambdify(
                list(subs.keys()), self.em_SF_errs[b])(**subs))
            ret_vals[2].append(sympy.lambdify(
                list(subs.keys()), self.mm_SF_errs[b])(**subs))
        return ret_vals


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
    "binning to use, which can be either 'null' or a list of the bin edges")
parser.add_argument("cutflows", help="Path to a output directory from "
                    "produce_DY_numbers.py, containing the cutflows for "
                    "this calculation")
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
    config = hjson.load(f)
with open(os.path.join(args.cutflows, "cutflows.json"), "r") as f:
    cutflow = hjson.load(f)
with open(os.path.join(args.cutflows, "cutflow_errs.json"), "r") as f:
    errors = hjson.load(f)

if config["bins"] is None:
    bins = {"channel": [0, 1, 2, 3]}
else:
    bin_edges = [int(k.split("_")[0]) for k in config["rebin"].keys()]
    bins = {"axis": config["bins"], "channel": [0, 1, 2, 3]}
out_dict = {"bins": bins}

sf_eq = SFEquations(config)
sfs = sf_eq.evaluate(cutflow, args.cutname)
sf_eq.calculate_errs()
stat_errs = sf_eq.evaluate_errs(cutflow, errors, args.cutname)

if args.variation:
    if len(args.variation) > 2:
        raise ValueError("Too many arguments supplied to variation (max 2)")
    elif len(args.variation) == 2:
        with open(args.variation[0], "r") as f:
            var_cutflow = hjson.load(f)
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

if config["bins"] is None:
    out_dict["factors"] = [sf[0] for sf in out_dict["factors"]]
    out_dict["factors_up"] = [sf[0] for sf in out_dict["factors_up"]]
    out_dict["factors_down"] = [sf[0] for sf in out_dict["factors_down"]]

with open(args.output, "w+") as f:
    hjson.dump(out_dict, f, indent=4)
