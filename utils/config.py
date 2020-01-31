#!/usr/bin/env python3

import numpy as np
import uproot
import json
import os


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
            try:
                bins_for_key = self._bins[key]
            except KeyError:
                raise ValueError("Scale factor does not depend on \"{}\""
                                 .format(key))
            binIdxs.append(np.digitize(val, bins_for_key) - 1)
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
            fpath = self._replace_special_vars(sfpath[0])
            hist = uproot.open(fpath)[sfpath[1]]
            sf = ScaleFactors.from_hist(hist,
                                        "abseta" if is_abseta else "eta",
                                        "pt")
            sfs.append(sf)
        return sfs

    def _replace_special_vars(self, s):
        SPECIAL_VARS = {
            "$DATADIR": "datadir",
        }

        for name, configvar in SPECIAL_VARS.items():
            if name in s:
                if configvar not in self._config:
                    raise ConfigError("{} contained in {} but datadir was "
                                      "not specified in config".format(
                                          name, configvar))
                s = s.replace(name, self._config[configvar])
        return s

    def _get_path(self, key):
        return os.path.realpath(self._replace_special_vars(self._config[key]))

    def _get(self, key):
        if key in self._cache:
            return self._cache[key]

        if key not in self._config:
            raise ConfigError("\"{}\" not specified in config".format(key))

        if key == "year":
            self._cache["year"] = str(self._config[key])
            if self._cache["year"] not in ("2016", "2017", "2018"):
                raise ConfigError(
                    "Invalid year {}".format(self._cache["year"]))
            return self._cache["year"]
        if key == "electron_sf":
            self._cache["electron_sf"] = self._get_scalefactors("electron_sf",
                                                                False)
            return self._cache["electron_sf"]
        elif key == "muon_sf":
            self._cache["muon_sf"] = self._get_scalefactors("muon_sf", True)
            return self._cache["muon_sf"]
        elif key == "btag_sf":
            paths = [self._replace_special_vars(path)
                     for path in self._config[key]]
            self._cache[key] = paths
            return paths
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
        elif key in ("genhist_path", "store", "lumimask", "mc_lumifactors"):
            self._cache[key] = self._get_path(key)
            return self._cache[key]

        return self._config[key]

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self._get(k) for k in key]
        else:
            return self._get(key)

    def __contains__(self, key):
        return key in self._config
