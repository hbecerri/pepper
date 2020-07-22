#!/usr/bin/env python3

import numpy as np
import awkward
import uproot
import json
import os
import coffea
from coffea.lookup_tools.extractor import file_converters
import logging
import warnings

import pepper
from pepper import btagging, HistDefinition


logger = logging.getLogger(__name__)


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
        counts = None
        for key, bins_for_key in self._bins.items():
            try:
                val = kwargs[key]
            except KeyError:
                raise ValueError("Scale factor depends on \"{}\" but no such "
                                 "argument was not provided"
                                 .format(key))
            if isinstance(val, awkward.JaggedArray):
                counts = val.counts
                val = val.flatten()
            binIdxs.append(np.digitize(val, bins_for_key) - 1)
        ret = factors[tuple(binIdxs)]
        if counts is not None:
            ret = awkward.JaggedArray.fromcounts(counts, ret)
        return ret


def get_evaluator(filename, fileform, filetype=None):
    if filetype is None:
        filetype = "default"
    converter = file_converters[fileform][filetype]
    extractor = coffea.lookup_tools.extractor()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for key, value in converter(filename).items():
            extractor.add_weight_set(key[0], key[1], value)
    extractor.finalize()
    return extractor.make_evaluator()


class TopPtWeigter():
    # Top pt reweighting according to
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting
    def __init__(self, scale, a, b):
        self.scale = scale
        self.a = a
        self.b = b

    def __call__(self, toppt, antitoppt):
        sf = np.exp(self.a + self.b * toppt)
        antisf = np.exp(self.a + self.b * antitoppt)
        return np.sqrt(sf ** 2 * antisf ** 2) * self.scale


class ConfigError(RuntimeError):
    pass


class Config(object):
    def __init__(self, path_or_file, textparser=json.load):
        """Initialize the configuration.

        Arguments:
        path_or_file -- Either a path to the file containing the configuration
                        or a file-like object of it
        textparser -- Callable to be used to parse the text contained in
                      path_or_file
        """
        if isinstance(path_or_file, str):
            with open(path_or_file) as f:
                self._config = textparser(f)
                path = path_or_file
        else:
            self._config = textparser(f)
            if hasattr(f, "name"):
                path = f.name
            else:
                path = "unknown"
        self._cache = {}
        self._config["configdir"] = os.path.dirname(os.path.realpath(path))

        logger.debug("Configuration read")
        if "datadir" in self._config:
            logger.debug(f"datadir: {self._config['datadir']}")
        if "configdir" in self._config:
            logger.debug(f"configdir: {self._config['configdir']}")
        if "store" in self._config:
            logger.debug(f"store: {self._config['store']}")

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
            "$CONFDIR": "configdir",
            "$STOREDIR": "store",
        }

        if isinstance(s, dict):
            return {self._replace_special_vars(key):
                    self._replace_special_vars(val) for key, val in s.items()}
        elif isinstance(s, list):
            return [self._replace_special_vars(item) for item in s]
        elif isinstance(s, str):
            for name, configvar in SPECIAL_VARS.items():
                if name in s:
                    if configvar not in self._config:
                        raise ConfigError(
                            "{} contained in config but {} was "
                            "not specified".format(name, configvar))
                    s = s.replace(name, self._config[configvar])
        return s

    def _get_path(self, key):
        return os.path.realpath(self._replace_special_vars(self._config[key]))

    def _get_maybe_external(self, key):
        data = self._config[key]
        if isinstance(data, str):
            data = self._replace_special_vars(data)
            with open(data) as f:
                data = json.load(f)
        return data

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
        elif key == "top_pt_reweighting":
            opts = self._config["top_pt_reweighting"]
            if not isinstance(opts, list) or len(opts) != 3:
                raise ConfigError(
                    "top_pt_reweighting must be of shape [scale, a, b]")
            self._cache[key] = TopPtWeigter(*opts)
            return self._cache[key]
        elif key == "electron_sf":
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
        elif key == "jet_correction":
            evaluators = {}
            for path in self._config[key]:
                path = self._replace_special_vars(path)
                evaluators.update(get_evaluator(path, "txt", "jec"))
            fjc = coffea.jetmet_tools.FactorizedJetCorrector(**evaluators)
            self._cache[key] = fjc
            return fjc
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
            factors = self._get_maybe_external(key)
            if not isinstance(factors, dict):
                raise ConfigError("mc_lumifactors must eiter be dict or a path"
                                  " to JSON file containing a dict")
            self._cache[key] = factors
            return factors
        elif key == "crosssection_uncertainty":
            uncerts = self._get_maybe_external(key)
            if not isinstance(uncerts, dict):
                raise ConfigError("crosssection_uncertainty must either be "
                                  "dict or a path to JSON file containing a "
                                  "dict")
            self._cache[key] = uncerts
            return uncerts
        elif key == "hists":
            hists = self._get_maybe_external(key)
            if not isinstance(hists, dict):
                raise ConfigError(f"{key} must either be "
                                  "dict or a path to JSON file containing a "
                                  "dict")
            hists = {k: HistDefinition(c) for k, c in hists.items()}
            self._cache[key] = hists
            return hists
        elif key in ("DY_SFs", "DY_SF_errs"):
            if isinstance(self._config[key], list):
                data = self._config[key][0]
                data = self._replace_special_vars(data)
                with open(data) as f:
                    data = json.load(f)
                for i in range(1, len(self._config[key])):
                    data = data[self._config[key][i]]
            else:
                data = self._config[key]
            if not isinstance(data, dict):
                raise ConfigError(f"{key} must either be a "
                                  "dict or a list, where the first index is a "
                                  "path to a json, and any subsequent indices "
                                  "are keys to the dict")
            self._cache[key] = data
            return data
        elif key in ("reco_info_file", "store", "lumimask",
                     "mc_lumifactors"):
            self._cache[key] = self._get_path(key)
            return self._cache[key]

        return self._replace_special_vars(self._config[key])

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self._get(k) for k in key]
        else:
            return self._get(key)

    def __contains__(self, key):
        return key in self._config

    def __setitem__(self, key, value):
        self._cache[key] = value

    def get_datasets(self, dsnames=None, dstype="any"):
        if dstype not in ("any", "mc", "data"):
            raise ValueError("dstype must be either 'any', 'mc' or 'data'")
        datasets = {s: v for s, v in self["exp_datasets"].items()
                    if v is not None}
        duplicate = set(datasets.keys()) & set(self["mc_datasets"])
        if len(duplicate) > 0:
            raise ConfigError("Got duplicate dataset names: {}".format(
                ", ".join(duplicate)))
        datasets.update({s: v for s, v in self["mc_datasets"].items()
                         if v is not None})
        for dataset in set(datasets.keys()):
            if ((dstype == "mc" and dataset not in self["mc_datasets"])
                    or (dstype == "data"
                        and dataset not in self["expdatasets"])
                    or (dsnames is not None and dataset not in dsnames)):
                del datasets[dataset]
        requested_datasets = datasets.keys()
        datasets, paths2dsname =\
            pepper.datasets.expand_datasetdict(datasets, self["store"])
        missing_datasets = requested_datasets - datasets.keys()
        if len(missing_datasets) > 0:
            raise ConfigError("Could not find files for: "
                              + ", ".join(missing_datasets))
        return datasets
