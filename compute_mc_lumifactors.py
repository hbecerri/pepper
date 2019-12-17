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

import utils


parser = ArgumentParser(description="Compute factors from luminosity and "
                                    "cross sections to scale MC")
parser.add_argument("lumi", type=float, help="The luminosity in 1/fb")
parser.add_argument("crosssections", help="Path to the JSON file containing "
                                          "cross section in fb")
parser.add_argument("config", help="Path to the JSON config file containing "
                                   "the MC dataset names")
parser.add_argument("out", help="Path to the output file")
args = parser.parse_args()

config = utils.Config(args.config)
store, datasets = config[["store", "mc_datasets"]]
datasets = utils.expand_datasetdict(datasets, store)[0]
crosssections = json.load(open(args.crosssections))
counts = defaultdict(int)
num_files = len(list(chain(*datasets.values())))
i = 0
for process_name, proc_datasets in datasets.items():
    if process_name not in crosssections:
        print("Could not find crosssection for {}".format(process_name))
        continue
    for path in proc_datasets:
        f = uproot.open(path)
        counts[process_name] += f["Runs"]["genEventSumw"].array()[0]
        print("[{}/{}] Processed {}".format(i + 1, num_files, path))
        i += 1
factors = {}
for key in crosssections.keys():
    factors[key] = crosssections[key] * args.lumi / counts[key]
    print("{}: {} fb, {} events, factor of {:.3e}".format(key,
                                                          crosssections[key],
                                                          counts[key],
                                                          factors[key]))

json.dump(factors, open(args.out, "w"), indent=4)
