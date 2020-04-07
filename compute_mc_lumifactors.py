#!/usr/bin/env python3

import os
import sys
import numpy as np
from argparse import ArgumentParser
import json
from glob import glob
from collections import defaultdict
import uproot
from itertools import chain

import pepper


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
datasets = pepper.datasets.expand_datasetdict(datasets, store)[0]
if "dataset_for_systematics" in config:
    dsforsys = config["dataset_for_systematics"]
else:
    dsforsys = {}
crosssections = json.load(open(args.crosssections))
counts = defaultdict(int)
num_files = len(list(chain(*datasets.values())))
i = 0
for process_name, proc_datasets in datasets.items():
    if process_name not in crosssections and process_name not in dsforsys:
        print("Could not find crosssection for {}".format(process_name))
        continue
    for path in proc_datasets:
        f = uproot.open(path)
        gen_event_sumw = f["Runs"]["genEventSumw"].array()[0]
        has_lhe = f["Runs"]["LHEScaleSumw"].array()[0].size != 0
        if has_lhe:
            lhe_scale_sumw = f["Runs"]["LHEScaleSumw"].array()[0]
            lhe_scale_sumw /= lhe_scale_sumw[4]
            lhe_scale_sumw *= gen_event_sumw
            lhe_pdf_sumw = f["Runs"]["LHEPdfSumw"].array()[0] * gen_event_sumw

        counts[process_name] += gen_event_sumw
        if has_lhe:
            counts[process_name + "_LHEScaleSumw"] += lhe_scale_sumw
            counts[process_name + "_LHEPdfSumw"] += lhe_pdf_sumw
        print("[{}/{}] Processed {}".format(i + 1, num_files, path))
        i += 1
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

json.dump(factors, open(args.out, "w"), indent=4)
