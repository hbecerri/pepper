#!/usr/bin/env python3
from parsl import python_app


def process(directory, titles, mergesys, ignore, maskval=999,
            max_size_bytes=400000000):
    import os
    from collections import defaultdict
    import numpy as np
    import awkward as ak
    import uproot
    from pepper import HDF5File
    from tqdm import tqdm

    def create_or_extend_tree(f, treename, tree):
        if treename in f:
            f[treename].extend(tree)
        else:
            f[treename] = tree

    def read_input(fpath, tree, systree=None):
        if systree is None:
            systree = tree
        with HDF5File(fpath, "r") as data:
            cutmask = np.full(len(data["events"]), True)
            for cut in ak.fields(data["cutflags"]):
                cutmask = cutmask & ak.values_astype(
                    data["cutflags"][cut], bool)
            if not ak.any(cutmask):
                return 0
            events = data["events"]
            events["cuts"] = cutmask
            for column in ak.fields(events):
                if column in ignore:
                    continue
                observable = events[column]

                for i in range(observable.ndim):
                    if i == 0:
                        fill = maskval
                    else:
                        idx = (0,) * (observable.ndim - i)
                        fill = ak.full_like(observable[idx], maskval)
                    observable = ak.fill_none(observable, fill, axis=-i - 1)
                tree[column].append(observable)
            if "weight" in data.keys():
                systree["weight"].append(data["weight"])
            elif "systematics" in data.keys():
                systematics = data["systematics"]
                for column in ak.fields(systematics):
                    if column in ignore:
                        continue
                    systree[column].append(np.asarray(systematics[column]))

        return len(events)

    def save_output(f, tree, systree=None):
        tree = {k: ak.concatenate(v) for k, v in tree.items()}
        if systree is not None:
            systree = {k: ak.concatenate(v) for k, v in systree.items()}

        create_or_extend_tree(f, "Events", tree)
        if systree is not None:
            create_or_extend_tree(f, "Systematics", systree)

    rootpath = os.path.join(
        os.path.dirname(directory), os.path.basename(directory) + ".root.temp")
    f = uproot.recreate(rootpath)
    tree = defaultdict(list)
    systree = None if mergesys else defaultdict(list)
    numevents = 0
    for fname in tqdm(os.listdir(directory)):
        if not any(fname.endswith(ext) for ext in [".hdf5", ".h5"]):
            continue
        fpath = os.path.join(directory, fname)
        numevents += read_input(fpath, tree, systree)

        size = sum(np.asarray(v[0]).dtype.itemsize
                   * numevents for v in tree.values())
        if systree is not None:
            size += sum(np.asarray(v[0]).dtype.itemsize
                        * numevents for v in systree.values())
        if size > max_size_bytes:
            save_output(f, tree, systree)

            tree = defaultdict(list)
            systree = None if mergesys else {}
            numevents = 0

    if numevents != 0:
        save_output(f, tree, systree if not mergesys else None)
    f.close()
    os.rename(rootpath, rootpath.rsplit(".", 1)[0])


if __name__ == "__main__":
    import os
    from argparse import ArgumentParser
    from tqdm import tqdm
    import pepper
    import parsl
    import concurrent.futures

    parser = ArgumentParser(
        description="Merge and convert Pepper HDF5 files to Root files "
        "containing TTrees")
    parser.add_argument(
        "directory", help="Directory specified for eventdir in runproc, which "
        "contains subdirectories with .h5 files")
    parser.add_argument(
        "-c", "--condor", type=int, default=0, help="Number of HTCondor jobs "
        "to launch")
    parser.add_argument(
        "-r", "--retries", type=int, default=999999999, help="Number of "
        "retries on HTCondor if there is an error")
    parser.add_argument(
        "--silent", action="store_true", help="Do not print progress")
    parser.add_argument(
        "--title", nargs=2, metavar=("column", "title"), action="append",
        help="Set titles for specific columns")
    parser.add_argument(
        "--mergesys", action="store_true",
        help="If given, merge systematics tree with events tree")
    parser.add_argument(
        "-i", "--ignore_column", action="append",
        help="Ignore a column from the HDF5 (can be normal column or a "
        "systematic")
    parser.add_argument(
        "-s", "--skip", action="store_true",
        help="Skip recreating existing output files")
    args = parser.parse_args()

    if args.condor > 0:
        process = python_app(process)
        parsl_config = pepper.misc.get_parsl_config(
            args.condor, retries=args.retries)
        parsl.load(parsl_config)

    if args.title is not None:
        titles = {c: title for c, title in args.title}
    else:
        titles = {}

    directories = []
    for directory in [f.path for f in os.scandir(
            os.path.realpath(args.directory)) if f.is_dir()]:
        directory = os.path.normpath(directory)
        if args.skip and os.path.exists(directory + ".root"):
            continue
        directories.append(directory)

    result_funcs = []
    for directory in directories:
        result_funcs.append(process(
            directory, titles, args.mergesys, args.ignore_column))

    if args.condor > 0:
        for func in tqdm(concurrent.futures.as_completed(result_funcs),
                         total=len(result_funcs)):
            # Raise error if there are some
            func.result()
