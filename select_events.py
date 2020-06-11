#!/usr/bin/env python3

import os
import coffea
import shutil
import parsl
import json
from argparse import ArgumentParser
import logging

import pepper


parser = ArgumentParser(description="Select events from nanoAODs")
parser.add_argument("config", help="JSON configuration file")
parser.add_argument(
    "--eventdir", help="Event destination output directory. If not "
    "specified, no events will be saved")
parser.add_argument(
    "--histdir", help="Histogram destination output directory. By default, "
    "./hists will be used.", default="./hists")
parser.add_argument(
    "--dataset", nargs=2, action="append", metavar=("name", "path"),
    help="Can be specified multiple times. Ignore datasets given in "
    "config and instead process these")
parser.add_argument(
    "-c", "--condor", type=int, const=10, nargs="?", metavar="simul_jobs",
    help="Split and submit to HTCondor. By default a maximum of 10 condor "
    "jobs are submitted. The number can be changed by supplying it to this "
    "option."
)
parser.add_argument(
    "--chunksize", type=int, default=500000, help="Number of events to "
    "process at once. Defaults to 5*10^5")
parser.add_argument(
    "--mc", action="store_true", help="Only process MC files")
parser.add_argument(
    "-d", "--debug", action="store_true", help="Enable debug messages and "
    "only process a small amount of files to make debugging feasible")
parser.add_argument(
    "-p", "--parsl_config", help="JSON file holding a dictionary with the "
    "keys condor_init and condor_config. Former overwrites the enviroment "
    "script that is executed at the start of a Condor job. Latter is appended "
    "to the job submit file.")
args = parser.parse_args()

logger = logging.getLogger("pepper")
logger.addHandler(logging.StreamHandler())
if args.debug:
    logger.setLevel(logging.DEBUG)

config = pepper.Config(args.config)
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
if not config["compute_systematics"]:
    for sysds in config["dataset_for_systematics"].keys():
        if sysds in datasets:
            del datasets[sysds]

requested_datasets = datasets.keys()
datasets, paths2dsname = pepper.datasets.expand_datasetdict(datasets, store)
if args.dataset is None:
    missing_datasets = requested_datasets - datasets.keys()
    if len(missing_datasets) > 0:
        print("Could not find files for: " + ", ".join(missing_datasets))
        exit(1)
    num_files = len(paths2dsname)
    num_mc_files = sum(len(datasets[dsname])
                       for dsname in config["mc_datasets"].keys()
                       if dsname in requested_datasets)

    print("Got a total of {} files of which {} are MC".format(num_files,
                                                              num_mc_files))

if args.debug:
    print("Processing only one file per dataset because of --debug")
    datasets = {key: [val[0]] for key, val in datasets.items()}

if len(datasets) == 0:
    print("No datasets found")
    exit(1)

if args.eventdir is not None:
    # Check for non-empty subdirectories and remove them if wanted
    nonempty = []
    for dsname in datasets.keys():
        try:
            next(os.scandir(os.path.join(args.eventdir, dsname)))
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
                    shutil.rmtree(os.path.join(args.eventdir, dsname))
                break
            elif answer == "n":
                break

# Create histdir and in case of errors, raise them now (before processing)
os.makedirs(args.histdir, exist_ok=True)

processor = pepper.Processor(config, args.eventdir)
if args.condor is not None:
    executor = coffea.processor.parsl_executor
    # Load parsl config immediately instead of putting it into executor_args
    # to be able to use the same jobs for preprocessing and processing
    print("Spawning jobs. This can take a while")
    if args.parsl_config is not None:
        with open(args.parsl_config) as f:
            parsl_config = json.load(f)
        parsl_config = pepper.misc.get_parsl_config(
            args.condor,
            condor_submit=parsl_config["condor_config"],
            condor_init=parsl_config["condor_init"])
    else:
        parsl_config = pepper.misc.get_parsl_config(args.condor)
    parsl.load(parsl_config)
    executor_args = {}
else:
    if args.parsl_config is not None:
        print("Ignoring parsl_config because condor is not specified")
    executor = coffea.processor.iterative_executor
    executor_args = {}

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args,
    chunksize=args.chunksize)

hists = output["hists"]
jsonname = "hists.json"
hists_forjson = {}
for key, hist in hists.items():
    if hist.values() == {}:
        continue
    cuts = next(iter(output["cutflows"]["all"].values())).keys()
    cutnum = list(cuts).index(key[0])
    fname = "Cut {:03} {}.coffea".format(cutnum, "_".join(key))
    fname = fname.replace("/", "")
    coffea.util.save(hist, os.path.join(args.histdir, fname))
    hists_forjson[key] = fname
with open(os.path.join(args.histdir, jsonname), "a+") as f:
    try:
        hists_injson = {tuple(k): v for k, v in zip(*json.load(f))}
    except json.decoder.JSONDecodeError:
        hists_injson = {}
hists_injson.update(hists_forjson)
with open(os.path.join(args.histdir, jsonname), "w") as f:
    json.dump([[tuple(k) for k in hists_injson.keys()],
               list(hists_injson.values())], f, indent=4)
