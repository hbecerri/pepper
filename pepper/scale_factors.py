#!/usr/bin/env python3

import numpy as np
import coffea
from coffea.lookup_tools.extractor import file_converters
from coffea.btag_tools import BTagScaleFactor
import awkward as ak
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
        edges = hist.to_numpy(flow=True)[1:]
        if dimlabels is None:
            dimlabels = []
            for member in ("fXaxis", "fYaxis", "fZaxis")[:len(edges)]:
                dimlabels.append(hist.all_members[member].all_members["fName"])
        factors = hist.values(flow=True)
        sigmas = np.sqrt(hist.variances(flow=True))
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

    @classmethod
    def from_hist_per_bin(cls, hists, dimlabels=None, **kwargs):
        def get_factors(depth, hists):
            if depth == 0:
                factors = hists.values(flow=True)
                cls._setoverflow(factors, 1)
                return factors
            else:
                return [get_factors(depth-1, h) for h in hists]

        def get_sigmas(depth, hists):
            if depth == 0:
                sigmas = np.sqrt(hists.variances(flow=True))
                cls._setoverflow(sigmas, 0)
                return sigmas
            else:
                return [get_factors(depth-1, h) for h in hists]

        tmp_hists = hists
        for i, k in enumerate(kwargs):
            tmp_hists = tmp_hists[0]
        hist_edges = list(tmp_hists.to_numpy(flow=True)[1:])
        edges = list(kwargs.values()) + hist_edges

        if dimlabels is None:
            dimlabels = list(kwargs.keys())
            for member in ("fXaxis", "fYaxis", "fZaxis")[:len(hist_edges)]:
                dimlabels.append(
                    tmp_hists.all_members[member].all_members["fName"])
        else:
            dimlabels = list(kwargs.keys()) + dimlabels

        factors = ak.from_iter(get_factors(len(kwargs), hists))
        sigmas = ak.from_iter(get_sigmas(len(kwargs), hists))
        factors_up = factors + sigmas
        factors_down = factors - sigmas
        if len(edges) != len(dimlabels):
            raise ValueError("Got {} dimenions but {} labels"
                             .format(len(edges), len(dimlabels)))
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
            if isinstance(val, ak.Array):
                counts = []
                for i in range(val.ndim - 1):
                    counts.append(ak.num(val))
                    val = ak.flatten(val)
                val = np.asarray(val)
            binIdxs.append(np.digitize(val, bins_for_key) - 1)
        ret = factors[tuple(binIdxs)]
        if counts is not None:
            for count in reversed(counts):
                ret = ak.unflatten(ret, count)
        return ret


WpTuple = namedtuple("WpTuple", ("loose", "medium", "tight"))


BTAG_WP_CUTS = {
    "deepcsv": {
        "2016": WpTuple(0.2217,  0.6321,  0.8953),
        "2017": WpTuple(0.1522,  0.4941,  0.8001),
        "2018": WpTuple(0.1241,  0.4184,  0.7527),
        "ul2017": WpTuple(0.1355, 0.4506, 0.7738),
        "ul2018": WpTuple(0.1208, 0.4168, 0.7665),
    },
    "deepjet": {
        "2016": WpTuple(0.0614,  0.3093,  0.7221),
        "2017": WpTuple(0.0521,  0.3033,  0.7489),
        "2018": WpTuple(0.0494,  0.2770,  0.7264),
        "ul2017": WpTuple(0.0532, 0.3040, 0.7476),
        "ul2018": WpTuple(0.0490, 0.2783, 0.7100),
    }
}


class BTagWeighter:
    def __init__(self, sf_filename, eff_filename, tagger, year,
                 meastype="mujets"):
        self.eff_evaluator = get_evaluator(eff_filename)

        self.sf = []
        for i in range(3):
            # Note that for light flavor normally only inclusive is available,
            # thus we fix it to 'incl' here
            self.sf.append(BTagScaleFactor(
                sf_filename, i, f"{meastype},{meastype},incl"))

        self.wps = BTAG_WP_CUTS[tagger][year]

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
        elif wp > 2:
            raise ValueError(
                f"Expected value between 0 and 2 for wp, got {wp}")
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

        sf = self.sf[wp].eval(heavy_vari, jf, eta, pt, discr)
        if light_vari != heavy_vari:
            sf = ak.where(
                jf >= 4, sf,
                self.sf[wp].eval(light_vari, jf, eta, pt, discr))

        eff = self.eff_evaluator[efficiency](jf, pt, abs(eta))
        sfeff = sf * eff
        is_tagged = discr > self.wps[wp]

        p_mc = ak.prod(eff[is_tagged], axis=1) * ak.prod(
            (1 - eff)[~is_tagged], axis=1)
        p_data = ak.prod(sfeff[is_tagged], axis=1) * ak.prod(
            (1 - sfeff)[~is_tagged], axis=1)

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
            key = key.rsplit(";", 1)[0]
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
    def __init__(self, method, scale=1.0, sys_only=False, **kwargs):
        if method.lower() == "datanlo":
            self.sffunc = self.datanlo_sf
        elif method.lower() == "theory":
            self.sffunc = self.theory_sf
        else:
            raise ValueError(f"Invalid method: {method}")
        self.scale = scale
        self.sys_only = sys_only
        self.kwargs = kwargs

    def datanlo_sf(self, pt):
        return np.exp(self.kwargs["a"] + self.kwargs["b"] * pt)

    def theory_sf(self, pt):
        arg = self.kwargs
        return arg["a"] * np.exp(arg["b"] * pt) + arg["c"] * pt + arg["d"]

    def __call__(self, toppt, antitoppt):
        sf = self.sffunc(toppt)
        antisf = self.sffunc(antitoppt)
        return np.sqrt(sf ** 2 * antisf ** 2) * self.scale
