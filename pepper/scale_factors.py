#!/usr/bin/env python3

import numpy as np
import coffea
from coffea.lookup_tools.extractor import file_converters
import awkward
from collections import namedtuple
import warnings


def get_evaluator(filename, fileform=None, filetype=None):
    if fileform is None:
        fileform = filename.split(".")[-1]
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


class ScaleFactors:
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
    def from_hist(cls, hist, dimlabels=None):
        factors, edges = hist.allnumpy()
        if isinstance(edges, list):
            # In 2D and 3D case, edges are placed in a len 1 list
            edges = edges[0]
        if not isinstance(edges, tuple):
            # In 1D case, edges aren't wrapped in a tuple
            edges = (edges,)
        if dimlabels is None:
            dimlabels = []
            for attr in ("_fXaxis", "_fYaxis", "_fZaxis")[:len(edges)]:
                dimlabels.append(getattr(hist, attr)._fName)
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


WpTuple = namedtuple("WpTuple", ("loose", "medium", "tight"))


BTAG_WP_CUTS = {
    "deepcsv": {
        "2016": WpTuple(0.2217,  0.6321,  0.8953),
        "2017": WpTuple(0.1522,  0.4941,  0.8001),
        "2018": WpTuple(0.1241,  0.4184,  0.7527),
    },
    "deepjet": {
        "2016": WpTuple(0.0614,  0.3093,  0.7221),
        "2017": WpTuple(0.0521,  0.3033,  0.7489),
        "2018": WpTuple(0.0494,  0.2770,  0.7264),
    }
}


class BTagWeighter:
    def __init__(self, sf_filename, eff_filename, tagger, year):
        self.eff_evaluator = get_evaluator(eff_filename)
        # Suppress RuntimeWarning that Coffea raises for 2018 scale factors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sf_evaluator = get_evaluator(sf_filename)

        # Tagger name of CSV is unknown, have not API for it. Workaround
        somekey = next(iter(self.sf_evaluator.keys()))
        self.csvtaggername = somekey.split("_")[0]

        self.wps = BTAG_WP_CUTS[tagger][year]

    def _sf_func(self, wp, sys, jf_num):
        for measurement in ("mujets", "comb", "incl"):
            key = "{}_{}_{}_{}_{}".format(
                self.csvtaggername, wp, measurement, sys, jf_num)
            if key in self.sf_evaluator:
                return self.sf_evaluator[key]
        return None

    def __call__(
            self, wp, jf, eta, pt, discr, variation="central",
            efficiency="central"):
        if isinstance(wp, str):
            wp = wp.lower()
            if wp == "loose":
                wp = 0
            elif wp == "medium":
                wp = 1
            elif wp == "tight":
                wp = 2
            else:
                raise ValueError("Invalid value for wp. Expected 'loose', "
                                 "'medium' or 'tight'")
        elif not isinstance(wp, int):
            raise TypeError("Expected int or str for wp, got {}".format(wp))
        possible_variations = (
            "central", "light up", "light down", "heavy up", "heavy down")
        if variation not in possible_variations:
            raise ValueError(
                "variation must be one of: " + ", ".join(possible_variations))
        light_vari = "central"
        heavy_vari = "central"
        if variation != "central":
            vari_type, direction = variation.split(" ")
            if vari_type == "light":
                light_vari = direction
            else:
                heavy_vari = direction

        counts = pt.counts
        jf = jf.flatten()
        eta = eta.flatten()
        pt = pt.flatten()
        discr = discr.flatten()

        sf = np.ones_like(eta)
        sf[jf == 0] = self._sf_func(wp, light_vari, 2)(eta, pt, discr)[jf == 0]
        sf[jf == 4] = self._sf_func(wp, heavy_vari, 1)(eta, pt, discr)[jf == 4]
        sf[jf == 5] = self._sf_func(wp, heavy_vari, 0)(eta, pt, discr)[jf == 5]
        sf = awkward.JaggedArray.fromcounts(counts, sf)

        eff = self.eff_evaluator[efficiency](jf, pt, abs(eta))
        eff = awkward.JaggedArray.fromcounts(counts, eff)
        sfeff = sf * eff

        is_tagged = discr > self.wps[wp]
        is_tagged = awkward.JaggedArray.fromcounts(counts, is_tagged)

        p_mc = eff[is_tagged].prod() * (1 - eff)[~is_tagged].prod()
        p_data = sfeff[is_tagged].prod() * (1 - sfeff)[~is_tagged].prod()

        # TODO: What if one runs into numerical problems here?
        return p_data / p_mc

    @property
    def available_efficiencies(self):
        return set(self.eff_evaluator.keys())


class PileupWeighter:
    def __init__(self, rootfile):
        self.central = {}
        self.up = {}
        self.down = {}

        self.upsuffix = "_up"
        self.downsuffix = "_down"
        for key, hist in rootfile.items():
            key = key.decode().rsplit(";", 1)[0]
            sf = ScaleFactors.from_hist(hist, ["ntrueint"])
            if key.endswith(self.upsuffix):
                self.up[key[:-len(self.upsuffix)]] = sf
            elif key.endswith(self.downsuffix):
                self.down[key[:-len(self.downsuffix)]] = sf
            else:
                self.central[key] = sf
        if (self.central.keys() != self.up.keys()
                or self.central.keys() != self.down.keys()):
            raise ValueError(
                "Missing up/down or central weights for some datasets")

    def __call__(self, dsname, ntrueint, variation="central"):
        # If all_datasets is present, use that instead of per-dataset weights
        if "all_datasets" in self.central:
            key = "all_datasets"
        else:
            key = dsname
        if variation == "up":
            return self.up[key](ntrueint=ntrueint)
        elif variation == "down":
            return self.down[key](ntrueint=ntrueint)
        elif variation == "central":
            return self.central[key](ntrueint=ntrueint)
        else:
            raise ValueError("variation must be either 'up', 'down' or "
                             f"'central', not {variation}")


class TopPtWeigter:
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
