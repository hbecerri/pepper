import logging
import argparse
import itertools

import hjson
import numpy as np
import sympy
import coffea.util


logger = logging.getLogger(__name__)


class SFEquations():
    def __init__(self, config, bins):
        self.config = config
        self.Z_win_regs = ["in", "out"]
        self.btags = ["0b", "1b"]
        if bins is None:
            self.bins = [""]
        else:
            self.bins = bins
        self.chs = ["ee", "em", "mm"]
        Nee_inc = ""
        Nmm_inc = ""
        for reg in ["in_0b", "out_0b", "in_1b"]:
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

    def get_subs(self, n, hist, bins=False, err=False):
        if err:
            err_n = "_err"
            err_idx = 1
        else:
            err_n = ""
            err_idx = 0
        ret_dict = {}
        for ch, Zinout, btag in itertools.product(
                self.chs, self.Z_win_regs, self.btags):
            h = hist.integrate("channel", "is_" + ch)
            h = h.integrate("in_Z_window", Zinout + "_Z_win")
            h = h.integrate("btags", btag)
            if () in h.values():
                vals = h.values(True)[()][err_idx]
            else:
                print(f"Warning! {ch}, {Zinout}, {btag} not found in hist, "
                      f"assuming zero events")
                if bins:
                    vals = np.zeros(len(bins))
                else:
                    vals = 0
            if bins:
                for i, b in enumerate(self.bins):
                    ret_dict[f"{n}{ch}_{Zinout}_{btag}{b}{err_n}"] = vals[i]
            else:
                ret_dict[n + ch + "_" + Zinout + "_" + btag + err_n] = vals
        return ret_dict

    def get_dy_datasets(self):
        if "DY_datasets" in self.config:
            return self.config["DY_datasets"]
        else:
            return [k for k in self.config["mc_datasets"]
                    if k.startswith("DY")]

    def evaluate(self, hist):
        ret_vals = [[], [], []]
        data_hist = hist.integrate("dataset",
                                   list(self.config["exp_datasets"].keys()))
        dy_hist = hist.integrate("dataset", self.get_dy_datasets())
        if self.bins != [""]:
            subs = self.get_subs("N", data_hist, True)
            subs.update(self.get_subs("Z", dy_hist, True))
        else:
            subs = self.get_subs("N", data_hist)
            subs.update(self.get_subs("Z", dy_hist))
        for b in self.bins:
            ret_vals[0].append(sympy.lambdify(list(subs.keys()),
                                              self.ee_SFs[b])(**subs))
            ret_vals[1].append(sympy.lambdify(list(subs.keys()),
                                              self.em_SFs[b])(**subs))
            ret_vals[2].append(sympy.lambdify(list(subs.keys()),
                                              self.mm_SFs[b])(**subs))
        return ret_vals

    def get_err(self, sfs, n, ch, z_inout, btag, b):
        return (sympy.diff(sfs, n + ch + "_" + z_inout + "_" + btag + b) ** 2
                * sympy.symbols(
                    n + ch + "_" + z_inout + "_" + btag + b + "_err"))

    def calculate_errs(self):
        self.ee_SF_errs = {b: 0 for b in self.bins}
        self.em_SF_errs = {b: 0 for b in self.bins}
        self.mm_SF_errs = {b: 0 for b in self.bins}
        err_dicts = [(self.ee_SF_errs, self.ee_SFs),
                     (self.em_SF_errs, self.em_SFs),
                     (self.mm_SF_errs, self.mm_SFs)]
        for b in self.bins:
            for err_dict, SFs in err_dicts:
                for ch, Zinout, btag, diff_b in itertools.product(
                        self.chs, self.Z_win_regs, self.btags, self.bins):
                    err_dict[b] += self.get_err(
                        SFs[b], "N", ch, Zinout, btag, diff_b)
                    err_dict[b] += self.get_err(
                        SFs[b], "Z", ch, Zinout, btag, diff_b)
                err_dict[b] = err_dict[b] ** 0.5

    def evaluate_errs(self, values):
        ret_vals = [[], [], []]
        data_hist = hist.integrate("dataset",
                                   list(self.config["exp_datasets"].keys()))
        dy_hist = hist.integrate("dataset", self.get_dy_datasets())
        if self.bins != [""]:
            subs = self.get_subs("N", data_hist, True)
            subs.update(self.get_subs("Z", dy_hist, True))
            subs.update(self.get_subs("N", data_hist, True, True))
            subs.update(self.get_subs("Z", dy_hist, True, True))
        else:
            subs = self.get_subs("N", data_hist)
            subs.update(self.get_subs("Z", dy_hist))
            subs.update(self.get_subs("N", data_hist, err=True))
            subs.update(self.get_subs("Z", dy_hist, err=True))
        for b in self.bins:
            ret_vals[0].append(sympy.lambdify(
                list(subs.keys()), self.ee_SF_errs[b])(**subs))
            ret_vals[1].append(sympy.lambdify(
                list(subs.keys()), self.em_SF_errs[b])(**subs))
            ret_vals[2].append(sympy.lambdify(
                list(subs.keys()), self.mm_SF_errs[b])(**subs))
        return ret_vals


parser = argparse.ArgumentParser(
    description="Calculate DY scale factors from DY yields produced by "
    "ttbarll_dy_sf_produce.py. (Currently only works for 1d histograms)")
parser.add_argument("config", help="JSON configuration file (same as used "
                    "by ttbarll_dy_sf_produce.py)")
parser.add_argument(
    "hist", help="Path to a histogram filled by ttbarll_dy_sf_produce.py, "
    "containing the cutflows for this calculation, e.g. after the jet cut")
parser.add_argument("output", help="Path to the output file")
parser.add_argument(
    "-v", "--variation", help="Histogram for an alternate working point at "
    "which to calculate scale factors to  estimate systematic error, e.g. "
    "after the reco cut")
parser.add_argument(
    "-i", "--integrate", action="store_true", help="Integrate this histogram "
    "(including overflow bins) to get inclusive SF")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = hjson.load(f)

hist = coffea.util.load(args.hist)

# ToDo: get this from hist
if args.integrate:
    bins = {"channel": [0, 1, 2, 3]}
    bin_names = None
    hist = hist.project("dataset", "channel", "in_Z_window", "btags",
                        overflow="allnan")
else:
    if len(hist.dense_axes()) != 1:
        raise NotImplementedError("Can only compute 1d SFs!")
    edges = hist.dense_axes()[0].edges()
    bins = {"axis": list(edges), "channel": [0, 1, 2, 3]}
    bin_names = [str(edges[i]).replace(".", "p") + "_to_"
                 + str(edges[i + 1]).replace(".", "p")
                 for i in range(len(edges) - 1)]
out_dict = {"bins": bins}

sf_eq = SFEquations(config, bin_names)
sfs = sf_eq.evaluate(hist)
sf_eq.calculate_errs()
stat_errs = sf_eq.evaluate_errs(hist)

if args.variation:
    var_hist = coffea.util.load(args.variation)
    if args.integrate:
        var_hist = var_hist.project("dataset", "channel", "in_Z_window",
                                    "btags", overflow="allnan")
    var_sfs = sf_eq.evaluate(var_hist)
    sys_errs = [[np.abs(sfs[ch_i][bin_i] - var_sfs[ch_i][bin_i])
                 for bin_i in range(len(sfs[ch_i]))] for ch_i in range(3)]
    tot_errs = [
        [np.sqrt(sys_errs[ch_i][bin_i] ** 2 + stat_errs[ch_i][bin_i] ** 2)
         for bin_i in range(len(sfs[ch_i]))] for ch_i in range(3)]
else:
    tot_errs = stat_errs

out_dict["factors"] = sfs
out_dict["factors_up"] = [
    [sfs[ch_i][bin_i] + tot_errs[ch_i][bin_i]
     for bin_i in range(len(sfs[ch_i]))] for ch_i in range(3)]
out_dict["factors_down"] = [
    [sfs[ch_i][bin_i] - tot_errs[ch_i][bin_i]
     for bin_i in range(len(sfs[ch_i]))] for ch_i in range(3)]

if args.integrate:
    out_dict["factors"] = [sf[0] for sf in out_dict["factors"]]
    out_dict["factors_up"] = [sf[0] for sf in out_dict["factors_up"]]
    out_dict["factors_down"] = [sf[0] for sf in out_dict["factors_down"]]

with open(args.output, "w+") as f:
    hjson.dumpJSON(out_dict, f, indent=4)
