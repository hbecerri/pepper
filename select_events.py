#!/usr/bin/env python3

import os
import sys
import numpy as np
import awkward
import uproot
import coffea
from coffea.analysis_objects import JaggedCandidateArray as CandArray
from time import time
import random
from functools import partial
from collections import OrderedDict
import shutil
import parsl
from parsl.addresses import address_by_hostname

import config_utils
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


def skip_existing(dest, path):
    return os.path.exists(get_outpath(path, dest))


parser = ArgumentParser(description="Select events from nanoAODs")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument("dest", help="Path to destination output directory")
parser.add_argument(
    "--dataset", nargs=2, action="append", metavar=("name", "path"),
    help="Can be specified multiple times. Ignore datasets given in "
    "config and instead process these")
parser.add_argument(
    "--condor", type=int, const=10, nargs="?", metavar="simul_jobs",
    help="Split and submit to HTCondor. By default 10 condor jobs are "
    "submitted. The number can be changed by supplying it to this option"
)
parser.add_argument(
   "--skip_existing", action="store_true", help="Skip already existing "
   "files")
parser.add_argument(
    "--mc", action="store_true", help="Only process MC files")
parser.add_argument(
    "--debug", action="store_true", help="Only process a small amount of files"
    "to make debugging feasible")
args = parser.parse_args()


config = config_utils.Config(args.config)
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
    datasets.update(config["mc_datasets"])
else:
    datasets = {}
    for dataset in args.dataset:
        if dataset[0] in datasets:
            datasets[dataset[0]].append(dataset[1])
        else:
            datasets[dataset[0]] = [dataset[1]]

if args.skip_existing:
    datasets, paths2dsname = config_utils.expand_datasetdict(
        datasets, store, partial(skip_existing, args.dest))
else:
    datasets, paths2dsname = config_utils.expand_datasetdict(datasets, store)
num_files = len(paths2dsname)
num_mc_files = sum(len(datasets[dsname])
                   for dsname in config["mc_datasets"].keys())

print("Got a total of {} files of which {} are MC".format(num_files,
                                                          num_mc_files))

if args.debug:
    print("Processing only one file because of --debug")
    key = next(iter(datasets.keys()))
    datasets = {key: datasets[key][:1]}

nonempty = []
for dsname in datasets.keys():
    try:
        next(os.scandir(os.path.join(args.dest, dsname)))
    except (FileNotFoundError, StopIteration):
        pass
    else:
        nonempty.append(dsname)
if len(nonempty) != 0:
    print("Non-empty output directories: {}".format(", ".join(nonempty)))
    while True:
        answer = input("Delete? y/n ")
        if answer == "y":
            for dsname in nonempty:
                shutil.rmtree(os.path.join(args.dest, dsname))
            break
        elif answer == "n":
            break

processor = Processor(config, os.path.realpath(args.dest))
if args.condor is not None:
    executor = coffea.processor.parsl_executor
    conor_config = ""
    # Need to unset PYTHONPATH because of cmssw setting it incorrectly
    # Need to put own directory into PYTHONPATH for unpickling processor to
    # work. Should be unncessecary, one this we have correct module structure
    # Need to extend PATH to be able to execute the main parsle script
    condor_init = """
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
eval `scramv1 runtime -sh`
cd -
export PYTHONPATH={}
export PATH=~/.local/bin:$PATH
""".format(os.path.dirname(os.path.abspath(__file__)))
    provider = parsl.providers.CondorProvider(
        channel=parsl.channels.LocalChannel(),
        init_blocks=args.condor,
        max_blocks=args.condor,
        nodes_per_block=1,
        parallelism=1,
        scheduler_options=conor_config,
        worker_init=condor_init
    )
    parsl_executor = parsl.executors.HighThroughputExecutor(
        label="HTCondor",
        address=address_by_hostname(),
        prefetch_capacity=0,
        cores_per_worker=1,
        max_workers=1,
        provider=provider,
        worker_debug=True,
    )
    parsl_config = parsl.config.Config(
        executors=[parsl_executor],
        lazy_errors=False
    )

    executor_args = {"config": parsl_config}
else:
    executor = coffea.processor.iterative_executor
    executor_args = {}
output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args)
