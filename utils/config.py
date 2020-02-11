#!/usr/bin/env python3

import numpy as np
import uproot
import json
import os
import coffea
from coffea.lookup_tools.extractor import file_converters

from . import btagging


class ScaleFactors(object):
    def __init__(self, factors, factors_up, factors_down, bins):
        if "variation" in bins:
            raise ValueError("'variation' must not be in bins")
        self._factors = factors
        self._factors_up = factors_up
        self._factors_down = factors_down
        self._bins = bins

    @staticmethod
    def _setoverflow(factors, value):
        for i in range(factors.ndim):
            factors[tuple([slice(None)] * i + [slice(0, 1)])] = 1
            factors[tuple([slice(None)] * i + [slice(-1, None)])] = 1

    @classmethod
    def from_hist(cls, hist, dimlabels):
        factors, (edges,) = hist.allnumpy()
        sigmas = np.sqrt(hist.allvariances)
        factors_up = factors + sigmas
        factors_down = factors - sigmas
        if len(edges) != len(dimlabels):
            raise ValueError("Got {} dimenions but {} labels"
                             .format(len(edges), len(dimlabels)))
        edges_new = []
        # Set overflow bins to 1 so that events outside the histogram
        # get scaled by 1
        cls._setoverflow(factors, 1)
        if factors_up is not None:
            cls._setoverflow(factors_up, 1)
        if factors_down is not None:
            cls._setoverflow(factors_down, 1)
        bins = dict(zip(dimlabels, edges))
        return cls(factors, factors_up, factors_down, bins)

    def __call__(self, variation="central", **kwargs):
        if variation not in ("central", "up", "down"):
            raise ValueError("variation must be one of 'central', 'up', "
                             "'down'")
        factors = self._factors
        if variation == "up":
            factors = self._factors_up
        elif variation == "down":
            factors = self._factors_down

        binIdxs = []
        for key, bins_for_key in self._bins.items():
            try:
                val = kwargs[key]
            except KeyError:
                raise ValueError("Scale factor depends on \"{}\" but no such"
                                 "a argument was not provided"
                                 .format(key))
            binIdxs.append(np.digitize(val, bins_for_key) - 1)
        return factors[tuple(binIdxs)]


def get_evaluator(filename, fileform, filetype=None):
    if filetype is None:
        filetype = "default"
    converter = file_converters[fileform][filetype]
    extractor = coffea.lookup_tools.extractor()
    for key, value in converter(filename).items():
        extractor.add_weight_set(key[0], key[1], value)
    extractor.finalize()
    return extractor.make_evaluator()


def get_evaluator_single(filename, fileform, filetype=None):
    return next(iter(get_evaluator(filename, fileform, filetype)))


class ConfigError(RuntimeError):
    pass


class Config(object):
    def __init__(self, path):
        with open(path) as f:
            self._config = json.load(f)
        self._cache = {}

    def _get_scalefactors(self, key, dimlabels):
        sfs = []
        for sfpath in self._config[key]:
            if not isinstance(sfpath, list) or len(sfpath) < 2:
                raise ConfigError("scale factors needs to be list of"
                                  " at least 2-element-lists in form of"
                                  " [rootfile, histname, uncert1, uncert2]")
            fpath = self._replace_special_vars(sfpath[0])
            rootf = uproot.open(fpath)
            hist = rootf[sfpath[1]]
            if len(sfpath) > 2:
                sumw2 = np.zeros(len(hist._fSumw2))
                import pdb
                pdb.set_trace()
                for path in sfpath[2:]:
                    hist_err = rootf[path]
                    sumw2 += hist_err.allvariances.T.flatten() ** 2
                hist._fSumw2 = sumw2
            sf = ScaleFactors.from_hist(hist, dimlabels)
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
                                                                ["eta", "pt"])
            return self._cache["electron_sf"]
        elif key == "muon_sf":
            self._cache["muon_sf"] = self._get_scalefactors("muon_sf",
                                                            ["pt", "abseta"])
            return self._cache["muon_sf"]
        elif key == "btag_sf":
            weighters = []
            tagger = self["btag"].split(":")[0]
            year = self["year"]
            for weighter_paths in self._config[key]:
                paths = [self._replace_special_vars(path)
                         for path in weighter_paths]
                btagweighter = btagging.BTagWeighter(paths[0],
                                                     paths[1],
                                                     tagger,
                                                     year)
                weighters.append(btagweighter)
            self._cache[key] = weighters
            return weighters
        elif key == "jet_uncertainty":
            path = self._replace_special_vars(self._config[key])
            evaluator = get_evaluator(path, "txt", "junc")
            junc = coffea.jetmet_tools.JetCorrectionUncertainty(**evaluator)
            self._cache[key] = junc
            return junc
        elif key == "jet_resolution":
            path = self._replace_special_vars(self._config[key])
            evaluator = get_evaluator(path, "txt", "jr")
            jer = coffea.jetmet_tools.JetResolution(**evaluator)
            self._cache[key] = jer
            return jer
        elif key == "jet_ressf":
            path = self._replace_special_vars(self._config[key])
            evaluator = get_evaluator(path, "txt", "jersf")
            jersf = coffea.jetmet_tools.JetResolutionScaleFactor(**evaluator)
            self._cache[key] = jersf
            return jersf
        elif key == "mc_lumifactors":
            factors = self._config[key]
            if isinstance(factors, str):
                factors = self._replace_special_vars(factors)
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
