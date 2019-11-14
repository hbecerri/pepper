#!/usr/bin/env python3

import os
import sys
import numpy as np
from argparse import ArgumentParser
import json
from glob import glob
from collections import defaultdict
import uproot

import utils


def dataset_to_paths(dataset, store, ext=".root"):
    t, cv, tier = dataset.split("/")[1:]
    campaign, version = cv.split("-", 1)
    isMc = "SIM" in tier
    pat = "{}/{}/{}/{}/{}/{}/*/*{}".format(
        store, "mc" if isMc else "data", campaign, t, tier, version, ext)
    return glob(pat)


def read_paths(source, store, ext=".root"):
    paths = []
    if source.endswith(ext):
        paths = glob(source)
    else:
        for line in open(source):
            line = line.strip()
            if line.startswith(store):
                paths_from_line = glob(line)
            else:
                paths_from_line = dataset_to_paths(line, store, ext)
            num_files = len(paths_from_line)
            if num_files == 0:
                print("No files found for \"{}\"".format(line))
            else:
                print("Found {} file{} for \"{}\"".format(
                    num_files, "s" if num_files > 1 else "", line))
                paths.extend(paths_from_line)

    return [os.path.realpath(path) for path in paths]


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
store, datasets = config[["store", "datasets"]]
datasets = utils.expand_datasetdict(datasets, store)[0]
crosssections = json.load(open(args.crosssections))
counts = defaultdict(int)
for i, path in enumerate(datasets["MC"]):
    f = uproot.open(path)
    tree = f["Runs"]
    keyfound = False
    for key in crosssections.keys():
        if key in path:
            counts[key] += tree["genEventSumw"].array()[0]
            keyfound = True
            break
    if not keyfound:
        print("Could not find crosssection for {}".format(path))
        sys.exit(1)
    else:
        print("[{}/{}] Processed {}".format(i + 1, len(datasets["MC"]), path))
factors = {}
for key in crosssections.keys():
    factors[key] = crosssections[key] * args.lumi / counts[key]
    print("{}: {} fb, {} events".format(key, crosssections[key], counts[key]))

json.dump(factors, open(args.out, "w"), indent=4)
