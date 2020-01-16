#!/usr/bin/env python3

import os
import coffea
from functools import partial
import shutil
import parsl
from parsl.addresses import address_by_hostname
from argparse import ArgumentParser

import utils.config
from utils.datasets import expand_datasetdict
from processor import Processor


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
    "--chunksize", type=int, default=500000, help="Number of events to "
    "process at once. Defaults to 5*10^5")
parser.add_argument(
    "--mc", action="store_true", help="Only process MC files")
parser.add_argument(
    "--debug", action="store_true", help="Only process a small amount of files"
    "to make debugging feasible")
args = parser.parse_args()


config = utils.config.Config(args.config)
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

datasets, paths2dsname = expand_datasetdict(datasets, store)
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
    conor_config = ("requirements = (OpSysAndVer == \"SL6\" || OpSysAndVer =="
                    " \"CentOS7\")")
    # Need to unset PYTHONPATH because of cmssw setting it incorrectly
    # Need to put own directory into PYTHONPATH for unpickling processor to
    # work. Should be unncessecary, one this we have correct module structure
    # Need to extend PATH to be able to execute the main parsle script
    condor_init = """
source /cvmfs/cms.cern.ch/cmsset_default.sh
if lsb_release -r | grep -q 7\\.; then
cd /cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
else
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
fi
eval `scramv1 runtime -sh`
cd -
export PYTHONPATH={}
export PATH=~/.local/bin:$PATH
""".format(os.path.dirname(os.path.abspath(__file__)))
    provider = parsl.providers.CondorProvider(
        init_blocks=args.condor,
        max_blocks=args.condor,
        scheduler_options=conor_config,
        worker_init=condor_init
        )
    parsl_executor = parsl.executors.HighThroughputExecutor(
        label="HTCondor",
        address=address_by_hostname(),
        max_workers=1,
        provider=provider,
    )
    parsl_config = parsl.config.Config(
        executors=[parsl_executor],
        retries=100,
    )

    # Load config now instead of putting it into executor_args to be able to
    # use the same jobs for preprocessing and processing
    parsl.load(parsl_config)
    executor_args = {"tailtimeout": None}
else:
    executor = coffea.processor.iterative_executor
    executor_args = {}

if args.condor is not None:
    print("Spawning jobs. This can take a while")
output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args,
    chunksize=args.chunksize)
