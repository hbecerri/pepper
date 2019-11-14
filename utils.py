#!/usr/bin/env python3

import os
import sys
from glob import glob
import subprocess
import numpy as np
import uproot
import coffea.lumi_tools
import json
import awkward


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
              any of the afore mentioned (one per line)
    store -- Path to the store directory, e.g. /pnfs/desy.de/cms/tier2/store/
    ext -- File extension the files have

    Returns a list of paths as strings
    """
    paths = []
    if source.endswith(ext):
        paths = glob(source)
    elif (source.count("/") == 3
          and (source.endswith("NANOAOD") or source.endswith("NANOAODSIM"))):
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


def condor_submit(executable, arguments, initialdir):
    while True:
        sub = {
            "executable": executable,
            "arguments": "\"{}\"".format(" ".join(arguments)),
            "Initialdir": os.path.realpath(initialdir),
        }
        job = "\n".join(
            "{} = {}".format(key, val) for key, val in sub.items())
        job += "\n\nqueue\n"
        condor_submit = subprocess.Popen(
            ["condor_submit"], stdin=subprocess.PIPE)
        condor_submit.communicate(job.encode())
        if (condor_submit.wait() != 0):
            print("Got error from condor_submit. Trying agin", file=sys.stderr)
            continue
        else:
            break


class ScaleFactors(object):
    def __init__(self, factors, **bins):
        self._factors = factors
        self._bins = bins

    @classmethod
    def from_hist(cls, hist, *dimlabels):
        factors, (edges,) = hist.allnumpy()
        if len(edges) != len(dimlabels):
            raise ValueError("Got {} dimenions but {} labels"
                             .format(len(edges), len(dimlabels)))
        edges_new = []
        # Set overflow bins to 1 so that events outside the histogram
        # get scaled by 1
        for i in range(factors.ndim):
            factors[tuple([slice(None)] * i + [slice(0, 1)])] = 1
            factors[tuple([slice(None)] * i + [slice(-1, None)])] = 1
        bins = dict(zip(dimlabels, edges))
        return cls(factors, **bins)

    def __call__(self, **kwargs):
        binIdxs = []
        for key, val in kwargs.items():
            if key not in self._bins:
                raise ValueError("Scale factor does not depend on \"{}\""
                                 .format(key))
            binIdxs.append(np.digitize(val, self._bins[key]) - 1)
        return self._factors[tuple(binIdxs)]


class ConfigError(RuntimeError):
    pass


class Config(object):
    def __init__(self, path):
        with open(path) as f:
            self._config = json.load(f)
        self._cache = {}

    def _get_scalefactors(self, key, is_abseta):
        sfs = []
        for sfpath in self._config[key]:
            if not isinstance(sfpath, list) or len(sfpath) != 2:
                raise ConfigError("scale factors needs to be list of"
                                  " 2-element-lists in form of"
                                  " [rootfile, histname]")
            hist = uproot.open(sfpath[0])[sfpath[1]]
            sf = ScaleFactors.from_hist(hist,
                                        "abseta" if is_abseta else "eta",
                                        "pt")
            sfs.append(sf)
        return sfs

    def _get(self, key):
        if key in self._cache:
            return self._cache[key]

        if key not in self._config:
            raise ConfigError("\"{}\" not specified in config".format(key))

        if key == "lumimask":
            self._cache["lumimask"] = coffea.lumi_tools.LumiMask(
                self._config[key])
            return self._cache["lumimask"]
        elif key == "electron_sf":
            self._cache["electron_sf"] = self._get_scalefactors("electron_sf",
                                                                False)
            return self._cache["electron_sf"]
        elif key == "muon_sf":
            self._cache["muon_sf"] = self._get_scalefactors("muon_sf", True)
            return self._cache["muon_sf"]
        elif key == "mc_lumifactors":
            factors = self._config[key]
            if isinstance(factors, str):
                with open(factors) as f:
                    factors = json.load(f)
            if not isinstance(factors, dict):
                raise ConfigError("mc_lumifactors must eiter be dict or a path"
                                  " to JSON file containing a dict")
            self._cache[key] = factors
            return factors

        return self._config[key]

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self._get(k) for k in key]
        else:
            return self._get(key)

    def __contains__(self, key):
        return key in self._config


def jaggedlike(j, content):
    return awkward.JaggedArray(j.starts, j.stops, content)
