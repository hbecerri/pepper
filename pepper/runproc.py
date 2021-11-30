#!/usr/bin/env python3

import os
import sys
import importlib
import coffea
import shutil
import parsl
import argparse
import logging
from datetime import datetime

import pepper
import pepper.executor


BUILTIN_PROCESSORS = {
    "ttbarll": ("pepper.processor_ttbarll", "Processor")
}


class _ListShortcutsAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest,
                         default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        shortcuts = ", ".join(BUILTIN_PROCESSORS.keys())
        print(f"Available shortcuts: {shortcuts}")
        parser.exit()


def run_processor(processor_class=None, description=None, mconly=False):
    if description is None:
        description = ("Run a processor on files given in configuration and "
                       "save the output.")
    parser = argparse.ArgumentParser(description=description)
    if processor_class is None:
        parser.add_argument(
            "processor", help="Python source code file of the processor to "
            "run or a shortcut for a processor")
        parser.add_argument(
            "--listshortcuts", action=_ListShortcutsAction, help="List the "
            "available  shortcuts for processors to use in the processor "
            "argument and exit")
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
        "-l", "--loglevel",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Set log level. Overwrites what is set by --debug. Default is "
        "'warning'")
    parser.add_argument(
        "-i", "--condorinit",
        help="Shell script that will be sourced by an HTCondor job after "
        "starting. This can be used to setup environment variables, if using "
        "for example CMSSW. If not provided, the local content of the "
        "environment variable PEPPER_CONDOR_ENV will be used as path to the "
        "script instead.")
    parser.add_argument(
        "--condorsubmit",
        help="Text file containing additional parameters to put into the "
        "HTCondor job submission file that is used for condor_submit"
    )
    parser.add_argument(
        "--metadata", help="File to cache metadata in. This allows speeding "
        "up or skipping the preprocessing step. Default is "
        "'pepper_metadata.coffea'", default="pepper_metadata.coffea")
    parser.add_argument(
        "--statedata", help="File to write and load processing state to/from. "
        "This allows resuming the processor after an interruption that made "
        "the process quit. States produced from different configurations "
        "should not be loaded, as this can lead to bogus results. Default is "
        "'pepper_state.coffea'", default="pepper_state.coffea"
    )
    args = parser.parse_args()

    logger = logging.getLogger("pepper")
    logger.addHandler(logging.StreamHandler())
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.loglevel is not None:
        if args.loglevel == "critical":
            logger.setLevel(logging.CRITICAL)
        elif args.loglevel == "error":
            logger.setLevel(logging.ERROR)
        elif args.loglevel == "warning":
            logger.setLevel(logging.WARNING)
        elif args.loglevel == "info":
            logger.setLevel(logging.INFO)
        elif args.loglevel == "debug":
            logger.setLevel(logging.DEBUG)

    if processor_class is None and args.processor in BUILTIN_PROCESSORS:
        name, class_name = BUILTIN_PROCESSORS[args.processor]
        module = importlib.import_module(name)
        Processor = getattr(module, class_name)
        logger.debug(f"Using processor from {module.__file__}")
    elif processor_class is None:
        spec = importlib.util.spec_from_file_location("pepper", args.processor)
        if spec is None:
            sys.exit(f"No such processor shortcut or file: {args.processor}")
        proc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proc_module)
        try:
            Processor = proc_module.Processor
        except AttributeError:
            sys.exit("Could not find class with name 'Processor' in "
                     f"{args.processor}")
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
        sys.exit("No datasets found")

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

    if os.path.realpath(args.metadata) == os.path.realpath(args.statedata):
        print("--metadata and --statedate can not be the same")
        sys.exit(1)
    if os.path.exists(args.statedata) and os.path.getsize(args.statedata) > 0:
        mtime = datetime.fromtimestamp(os.stat(args.statedata).st_mtime)
        while True:
            answer = input(
                f"Found old processor state from {mtime}. Load? y/n ")
            if answer == "n":
                print(f"Please delete/rename the old file {args.statedata} or "
                      "use --statedata")
                sys.exit(1)
            elif answer == "y":
                break

    processor = Processor(config, args.eventdir)
    datasets = processor.preprocess(datasets)
    if args.condor is not None:
        executor_class = pepper.executor.ParslExecutor
        if args.condorinit is not None:
            with open(args.condorinit) as f:
                condorinit = f.read()
        else:
            condorinit = None
        if args.condorsubmit is not None:
            with open(args.condorsubmit) as f:
                condorsubmit = f.read()
        else:
            condorsubmit = None
        print("Spawning jobs. This can take a while")
        parsl_config = pepper.misc.get_parsl_config(
            args.condor,
            condor_submit=condorsubmit,
            condor_init=condorinit)
        parsl.load(parsl_config)
    else:
        if args.condorinit is not None or args.condorsubmit is not None:
            print(
                "Ignoring condor parameters because --condor is not specified")
        executor_class = pepper.executor.IterativeExecutor
    pre_executor = executor_class(state_file_name=args.metadata)
    executor = executor_class(state_file_name=args.statedata)

    try:
        pre_executor.load_state()
    except (FileNotFoundError, EOFError):
        pass
    try:
        executor.load_state()
    except (FileNotFoundError, EOFError):
        pass
    userdata = executor.state["userdata"]
    if "chunksize" in userdata:
        if userdata["chunksize"] != args.chunksize:
            sys.exit(
                f"'{args.statedata}' got different chunksize: "
                f"{userdata['chunksize']}. Delete it or change --chunksize")
    userdata["chunksize"] = args.chunksize

    runner = coffea.processor.Runner(
        executor, pre_executor, chunksize=args.chunksize,
        align_clusters=not args.force_chunksize, schema=processor.schema_class)
    output = runner(datasets, "Events", processor)
    processor.save_output(output, args.output)

    return output


if __name__ == "__main__":
    run_processor()
