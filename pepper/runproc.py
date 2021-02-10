#!/usr/bin/env python3

import os
import sys
import importlib
import coffea
import shutil
import parsl
import json
from argparse import ArgumentParser
import logging

import pepper


def run_processor(processor_class=None, description=None, mconly=False):
    if description is None:
        description = ("Run a processor on files given in configuration and "
                       "save the output.")
    parser = ArgumentParser(description=description)
    if processor_class is None:
        parser.add_argument(
            "processor", help="Python source code file of the processor to "
            "run")
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument(
        "--eventdir", help="Event destination output directory. If not "
        "specified, no events will be saved")
    parser.add_argument(
        "-o", "--output", help="Directory to save final output to. This "
        "usual are objects like cutflow and histograms. Defaults to '.'",
        default=".")
    parser.add_argument(
        "--file", nargs=2, action="append", metavar=("dataset_name", "path"),
        help="Can be specified multiple times. Ignore datasets given in "
        "config and instead process these. Can be specified multiple times.")
    parser.add_argument(
        "--dataset", action="append", help="Only process this dataset. Can be "
        "specified multiple times.")
    if not mconly:
        parser.add_argument(
            "--mc", action="store_true", help="Only process MC files")
    parser.add_argument(
        "-c", "--condor", type=int, const=10, nargs="?", metavar="simul_jobs",
        help="Split and submit to HTCondor. By default a maximum of 10 condor "
        "simultaneous jobs are submitted. The number can be changed by "
        "supplying it to this option.")
    parser.add_argument(
        "-r", "--retries", type=int, help="Number of times to retry if there "
        "is exception in an HTCondor job. If not given, retry infinitely.")
    parser.add_argument(
        "--chunksize", type=int, default=500000, help="Number of events to "
        "process at once. A smaller value means less memory usage. Defaults "
        "to 5*10^5")
    parser.add_argument(
        "--force_chunksize", action="store_true", help="If present, makes the "
        "processor process exactly the number of events given in chunksize, "
        "unless there aren't enough events in the file. This will make "
        "reading slower.")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug messages and "
        "only process a small amount of files to make debugging feasible")
    parser.add_argument(
        "-p", "--parsl_config", help="JSON file holding a dictionary with the "
        "keys condor_init and condor_config. Former overwrites the enviroment "
        "script that is executed at the start of a Condor job. Latter is "
        "appended to the job submit file.")
    args = parser.parse_args()

    logger = logging.getLogger("pepper")
    logger.addHandler(logging.StreamHandler())
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if processor_class is None:
        spec = importlib.util.spec_from_file_location("pepper", args.processor)
        proc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proc_module)
        try:
            Processor = proc_module.Processor
        except AttributeError:
            print("Could not find class with name 'Processor' in "
                  f"{args.processor}")
            sys.exit(1)
    else:
        Processor = processor_class

    config = Processor.config_class(args.config)
    store = config["store"]
    if not os.path.exists(store):
        raise pepper.config.ConfigError("store directory does not exist")

    datasets = {}
    if args.file is None:
        if not config["compute_systematics"]:
            exclude = config["dataset_for_systematics"].keys()
        else:
            exclude = None
        datasets = config.get_datasets(
            args.dataset, exclude, "mc" if mconly or args.mc else "any")
    else:
        datasets = {}
        for customfile in args.file:
            if customfile[0] in datasets:
                datasets[customfile[0]].append(customfile[1])
            else:
                datasets[customfile[0]] = [customfile[1]]

    if args.file is None:
        num_files = sum(len(dsfiles) for dsfiles in datasets.values())
        num_mc_files = sum(len(datasets[dsname])
                           for dsname in config["mc_datasets"].keys()
                           if dsname in datasets)

        print(
            f"Got a total of {num_files} files of which {num_mc_files} are MC")

    if args.debug:
        print("Processing only one file per dataset because of --debug")
        datasets = {key: [val[0]] for key, val in datasets.items()}

    if len(datasets) == 0:
        print("No datasets found")
        sys.exit(1)

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
            print(
                "Non-empty output directories: {}".format(", ".join(nonempty)))
            while True:
                answer = input("Delete? y/n ")
                if answer == "y":
                    for dsname in nonempty:
                        shutil.rmtree(os.path.join(args.eventdir, dsname))
                    break
                elif answer == "n":
                    break

    # Create histdir and in case of errors, raise them now (before processing)
    os.makedirs(args.output, exist_ok=True)

    processor = Processor(config, args.eventdir)
    datasets = processor.preprocess(datasets)
    executor_args = {
        "schema": processor.schema_class,
        "align_clusters": not args.force_chunksize}
    if args.condor is not None:
        executor = coffea.processor.parsl_executor
        # Load parsl config immediately instead of putting it into
        # executor_args to be able to use the same jobs for preprocessing and
        # processing
        print("Spawning jobs. This can take a while")
        if args.parsl_config is not None:
            with open(args.parsl_config) as f:
                parsl_config = json.load(f)
            parsl_config = pepper.misc.get_parsl_config(
                args.condor,
                condor_submit=parsl_config["condor_config"],
                condor_init=parsl_config["condor_init"])
        else:
            parsl_config = pepper.misc.get_parsl_config(
                args.condor, retries=args.retries)
        parsl.load(parsl_config)
    else:
        if args.parsl_config is not None:
            print("Ignoring parsl_config because condor is not specified")
        executor = coffea.processor.iterative_executor

    output = coffea.processor.run_uproot_job(
        datasets, "Events", processor, executor, executor_args,
        chunksize=args.chunksize)
    processor.save_output(output, args.output)

    return output


if __name__ == "__main__":
    run_processor()
