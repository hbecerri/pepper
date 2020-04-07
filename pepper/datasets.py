#!/usr/bin/env python3

import os
from glob import glob


def dataset_to_paths(dataset, store, ext=".root"):
    """Get the paths of the files belonging to a dataset

    Parameters:
    dataset -- name of the dataset
    store -- Path to the store directory, e.g. /pnfs/desy.de/cms/tier2/store/
    ext -- File extension the files have

    Returns a list of paths as strings
    """
    t, cv, tier = dataset.split("/")[1:]
    campaign, version = cv.split("-", 1)
    isMc = "SIM" in tier
    pat = "{}/{}/{}/{}/{}/{}/*/*{}".format(
        store, "mc" if isMc else "data", campaign, t, tier, version, ext)
    return [os.path.normpath(p) for p in glob(pat)]


def read_paths(source, store, ext=".root"):
    """Get all paths to files of a dataset, which can be interpreted from a
    source

    Parameters:
    source -- A glob pattern, dataset name or a path to a text file containing
              any of the afore mentioned (one per line). A relative path will
              be looked for within store. If it ends with ext, it will be
              considered as a glob pattern.
    store -- Path to the store directory, e.g. /pnfs/desy.de/cms/tier2/store/
    ext -- File extension the files have

    Returns a list of paths as strings
    """
    paths = []
    is_dsname = (source.count("/") == 3
                 and (source.endswith("NANOAOD")
                      or source.endswith("NANOAODSIM")))
    if not os.path.isabs(source) and not is_dsname:
        source = os.path.join(store, source)
    if source.endswith(ext):
        paths = glob(source)
    elif is_dsname:
        paths.extend(dataset_to_paths(source, store, ext))
    else:
        with open(source) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
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
    return paths


def expand_datasetdict(datasets, store, ignore_path=None, ext=".root"):
    """Interpred a dict of dataset names or paths

    Parameters:
    datasets -- A dict whose values are lists of glob patterns, dataset names
                or files containing any of the afore mentioned
    store -- Path to the store directory, e.g. /pnfs/desy.de/cms/tier2/store/
    ignore_path -- Callable of the form file path -> bool. If it evaluates to
                   not True, the file path is skipped for the output. If None,
                   no files are skipped
    ext -- File extension the files have

    Returns a tuple of two dicts. The first one is a dict mapping the keys of
    `datasets` to lists of paths for the corresponding files. The second one is
    the inverse mapping.
    """
    paths2dsname = {}
    datasetpaths = {}
    for key in datasets.keys():
        paths = list(dict.fromkeys([
            a for b in datasets[key] for a in read_paths(b, store, ext)]))
        if ignore_path:
            processed_paths = []
            for path in paths:
                if not ignore_path(path):
                    processed_paths.append(path)
            paths = processed_paths

        paths_nodups = []
        for path in paths:
            if path in paths2dsname:
                print("Path {} given for {} but already present in datasets "
                      "for {}".format(path, key, paths2dsname[path]))
            else:
                paths2dsname[path] = key
                paths_nodups.append(path)
        datasetpaths[key] = paths_nodups

    return datasetpaths, paths2dsname
