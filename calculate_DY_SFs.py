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


logger = logging.getLogger(__name__)


class SFEquations():
    def __init__(self, met_bins):
        self.regions = ["in_0b_", "out_0b_", "in_1b_"]
        self.MET_bins = met_bins
        data_MC_chs = ["Nee_", "Nem_", "Nmm_", "Zee_", "Zmm_"]
        self.vals_to_sub = [ch + reg + MET for ch in data_MC_chs
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


def rebin_met(cutflow, errors, rebin_dict):
    regions = ["in_0b_", "out_0b_", "in_1b_"]
    chs = ["ee_", "em_", "mm_"]
    for new_bin, old_bins in rebin_dict.items():
        for entry in entries:
            for ch in data_MC_chs:
                for reg in regions:
                    nums[entry][ch+reg+new_bin] = sum(
                        [nums[entry][ch+reg+old_bin] for old_bin in old_bins])


parser = ArgumentParser(description="Select events from nanoAODs")
parser.add_argument("labels", help="Path to a config containing labels for which "
                    "datsets correspond to DY, and which to data")
parser.add_argument("cutflow", help="Path to a cutflow containing the numbers "
                    "required to calculate the SFs")
parser.add_argument("-c", "--cutname", default="Jet pt req", help="Cut for which "
                    "scale factors should be calculated- default 'Jet pt req'")
parser.add_argument("-v", "--variation", nargs=?, default="Reco", help="Alternative"
                    " working point at which to calculate scale factors to estimate"
                    " systematic error. May give either one argument (which will be"
                    " interpreted as another cut in the same cutflow), or two, "
                    "specifying a different cutflow file, and a cut in that cutflow."
                    " Default: 'Reco'")
args = parser.parse_args()


with open(args.cutflow, "r") as f:
    cutflows = json.load(f)
cutflow = cutflows["cutflow"]
errors = cutflows["errs"]

met_bins = [k[9:] for k in cutflow if k.startswith("ee_in_0b_")]

#rebin_met(nums, {"100_to_inf": ["100_to_160", "160_to_inf"]})
sf_eq = SFEquations(met_bins)
nums["LO_SFs"].update(sf_eq.evaluate(nums["LO_DY_numbers"]))
sf_eq.calculate_errs()
nums["NLO_SF_errs"].update(sf_eq.evaluate_errs(nums["NLO_DY_numbers"],
                                               nums["NLO_DY_errs"]))

rebin_met(nums, {"Inclusive": MET_bins})
sfs_inclusive = SFEquations(["Inclusive"])
nums["LO_SFs"].update(sfs_inclusive.evaluate(nums["LO_DY_numbers"]))
nums["NLO_SFs"].update(sfs_inclusive.evaluate(nums["NLO_DY_numbers"]))
sfs_inclusive.calculate_errs()
nums["LO_SF_errs"].update(sfs_inclusive.evaluate_errs(nums["LO_DY_numbers"],
                                                      nums["LO_DY_errs"]))
nums["NLO_SF_errs"].update(sfs_inclusive.evaluate_errs(nums["NLO_DY_numbers"],
                                                       nums["NLO_DY_errs"]))

