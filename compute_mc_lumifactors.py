#!/usr/bin/env python3

import os
import numpy as np
from argparse import ArgumentParser
import json
from collections import defaultdict
import uproot
from itertools import chain
from tqdm import tqdm

import pepper


def update_counts(f, counts, process_name, geskey, lhesskey, lhepdfskey):
    gen_event_sumw = f["Runs"][geskey].array()[0]
    has_lhe = f["Runs"][lhesskey].array()[0].size != 0
    if has_lhe:
        lhe_scale_sumw = f["Runs"][lhesskey].array()[0]
        lhe_scale_sumw /= lhe_scale_sumw[4]
        lhe_scale_sumw *= gen_event_sumw
        lhe_pdf_sumw = f["Runs"][lhepdfskey].array()[0] * gen_event_sumw

    counts[process_name] += gen_event_sumw
    if has_lhe:
        counts[process_name + "_LHEScaleSumw"] += lhe_scale_sumw
        counts[process_name + "_LHEPdfSumw"] += lhe_pdf_sumw
    return counts


parser = ArgumentParser(description="Compute factors from luminosity and "
                                    "cross sections to scale MC")
parser.add_argument("lumi", type=float, help="The luminosity in 1/fb")
parser.add_argument("crosssections", help="Path to the JSON file containing "
                                          "cross section in fb")
parser.add_argument("config", help="Path to the JSON config file containing "
                                   "the MC dataset names")
parser.add_argument("out", help="Path to the output file")
args = parser.parse_args()

config = pepper.Config(args.config)
store, datasets = config[["store", "mc_datasets"]]
datasets = {k: v for k, v in datasets.items() if v is not None}
datasets = pepper.datasets.expand_datasetdict(datasets, store)[0]
if "dataset_for_systematics" in config:
    dsforsys = config["dataset_for_systematics"]
else:
    dsforsys = {}
if "randomised_parameter_scan_datasets" in config:
    rps_datasets = config["randomised_parameter_scan_datasets"]
else:
    rps_datasets = {}
crosssections = json.load(open(args.crosssections))
counts = defaultdict(int)
num_files = len(list(chain(*datasets.values())))
i = 0
for process_name, proc_datasets in tqdm(datasets.items(), desc="Total"):
    if (process_name not in crosssections and process_name not in dsforsys
            and process_name not in rps_datasets):
        print("Could not find crosssection for {}".format(process_name))
        continue
    for path in tqdm(proc_datasets, desc="Per dataset"):
        f = uproot.open(path)
        if process_name in rps_datasets:
            scanpoints = [key.decode("utf-8").split("_", 1)[1]
                          for key in f["Runs"].iterkeys()
                          if key.decode("utf-8").startswith("genEventSumw_")]
            for sp in scanpoints:
                geskey = "genEventSumw_" + sp
                lhesskey = "LHEScaleSumw_" + sp
                lhepdfskey = "LHEPdfSumw_" + sp
                counts = update_counts(f, counts, sp,
                                       geskey, lhesskey, lhepdfskey)
        else:
            if "genEventSumw_" in f["Runs"]:
                # inconsistent naming in NanoAODv6
                geskey = "genEventSumw_"
                lhesskey = "LHEScaleSumw_"
                lhepdfskey = "LHEPdfSumw_"
            else:
                geskey = "genEventSumw"
                lhesskey = "LHEScaleSumw"
                lhepdfskey = "LHEPdfSumw"
            counts = update_counts(f, counts, process_name,
                                   geskey, lhesskey, lhepdfskey)
        i += 1
    print("\033[F\033[F")  # Workaround for tqdm adding a new line
print("")
factors = {}
for key in counts.keys():
    if key.endswith("_LHEScaleSumw") or key.endswith("_LHEPdfSumw"):
        dsname = key.rsplit("_", 1)[0]
    else:
        dsname = key
    if dsname in dsforsys:
        xs = crosssections[dsforsys[dsname][0]]
    else:
        xs = crosssections[dsname]
    factor = xs * args.lumi / counts[key]
    if key.endswith("_LHEScaleSumw") or key.endswith("_LHEPdfSumw"):
        factor = counts[dsname] / counts[key]
    if isinstance(factor, np.ndarray):
        factor = list(factor)
    factors[key] = factor
    if key == dsname:
        print("{}: {} fb, {} events, factor of {:.3e}".format(key,
                                                              xs,
                                                              counts[key],
                                                              factors[key]))

if os.path.exists(args.out):
    try:
        with open(args.out) as f:
            factors_old = json.load(f)
    except json.decoder.JSONDecodeError:
        pass
    else:
        factors_old.update(factors)
        factors = factors_old

with open(args.out, "w") as f:
    json.dump(factors, f, indent=4)
