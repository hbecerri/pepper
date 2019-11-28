#!/usr/bin/env python3

import os
import sys
import numpy as np
import awkward
import h5py
import uproot
import coffea
from coffea.analysis_objects import JaggedCandidateArray as CandArray
from time import time
import random
from functools import partial
from collections import OrderedDict

import utils
from processor import Processor
from argparse import ArgumentParser

def get_outpath(path, dest):
    if path.startswith("/pnfs/desy.de/cms/tier2/store/"):
        outpath = path.replace("/pnfs/desy.de/cms/tier2/store/", "")
    else:
        outpath = path
    outpath = os.path.splitext(outpath)[0] + ".hdf5"
    outpath = os.path.join(dest, outpath)
    return outpath

parser = ArgumentParser(description="Select events from nanoAODs")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument("dest", help="Path to destination output directory")
parser.add_argument(
    "--dataset", nargs=2, action="append", metavar=("name", "path"),
    help="Can be specified multiple times. Ignore datasets given in "
    "config and instead process these")
parser.add_argument(
    "--condor", type=int, const=10, nargs="?", metavar="files_per_job",
    help="Split and submit to HTCondor. By default for every 10 files a "
    "job is created. The number can be changed by supplying it to this "
    "option")
parser.add_argument(
   "--skip_existing", action="store_true", help="Skip already existing "
   "files")
parser.add_argument(
    "--mc", action="store_true", help="Only process MC files")
args = parser.parse_args()

config = utils.Config(args.config)
store = config["store"]

datasets = {}
if args.dataset is None:
    datasets = config["exp_datasets"]
    if "MC" in datasets:
        raise utils.ConfigError(
            "MC must not be specified in config exp_datasets")
    datasets["MC"] = []
    for mc_datasets in config["mc_datasets"].values():
        datasets["MC"].extend(mc_datasets)
else:
    datasets = {}
    for dataset in args.dataset:
        if dataset[0] in datasets:
            datasets[dataset[0]].append(dataset[1])
        else:
            datasets[dataset[0]] = [dataset[1]]

if args.skip_existing:
    datasets, paths2dsname = utils.expand_datasetdict(
        datasets, store, partial(skip_existing, args.dest))
else:
    datasets, paths2dsname = utils.expand_datasetdict(datasets, store)
if args.mc:
    paths2dsname = {path: dsname for path, dsname in paths2dsname.items()
                    if dsname == "MC"}
    datasets = {"MC": datasets["MC"]}
num_files = len(paths2dsname)
num_mc_files = len(datasets["MC"]) if "MC" in datasets else 0

print("Got a total of {} files of which {} are MC".format(num_files,
                                                          num_mc_files))

#key = next(iter(datasets.keys()))
#datasets_debug = {key: datasets[key]}
output = coffea.processor.run_uproot_job(
    datasets, treename="Events",
    processor_instance=Processor(config),
    executor=coffea.processor.iterative_executor,
    executor_args={"workers": 4, "flatten": False, "processor_compression": None}, chunksize=100000)

for dataset in datasets.keys():
    if len(output["flat cols"][dataset])+len(output["jagged cols"][dataset])>0:
        outpath = get_outpath(dataset, args.dest)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with h5py.File(outpath, "w") as f:
            out = awkward.hdf5(f)
            for key in output["flat cols"][dataset].keys():
                out[key]=output["flat cols"][dataset][key]
            for key in output["jagged cols"][dataset].keys():
                out[key]=output["jagged cols"][dataset][key]