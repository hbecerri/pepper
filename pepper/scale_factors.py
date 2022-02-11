#!/usr/bin/env python3

import numpy as np
import coffea
from coffea.lookup_tools.extractor import file_converters
from coffea.btag_tools import BTagScaleFactor
import awkward as ak
import correctionlib
from collections import namedtuple
import warnings

from pepper.misc import onedimeval


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
        """Create a new ScaleFactors instance. Useful for binned scale factors.

        Arguments:
        factors -- Numpy array with the scale factors for the central variation
        factors_up -- Numpy array with the scale factors for the up variation
        factors_down -- Numpy array with the scale factors for the down
                        variation
        bins -- A dict whose length is equal to the number of dimensions of
                factors. Its nth key gives a name to the nth dimension, so
                that it can be used as a kwarg in the __call__ method of this
                class. "variation" can not be a key. Its values are Numpy
                arrays and determine the edges of the binnding used for the
                factors.
        """
        if "variation" in bins:
            raise ValueError("'variation' must not be in bins")
        for i, (axis, edges) in enumerate(bins.items()):
            if factors.shape[i] + 1 != len(edges):
                raise ValueError(
                    f"axis '{axis}' should have {factors.shape[i] + 1} "
                    f"edges but has {len(edges)}")
        if (factors_up is not None and factors_down is not None and
            (factors.shape != factors_up.shape
                or factors.shape != factors_down.shape)):
            raise ValueError(
                "factors_up or factors_down have inconsistent shape")
        self._factors = factors
        self._factors_up = factors_up
        self._factors_down = factors_down
        self._bins = bins

    @staticmethod
    def _setoverflow(factors, value):
        for i in range(factors.ndim):
            factors[tuple([slice(None)] * i + [slice(0, 1)])] = value
            factors[tuple([slice(None)] * i + [slice(-1, None)])] = value

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
            raise ValueError(
                f"Got {len(edges)} dimensions but {len(dimlabels)} labels")
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
            raise ValueError(
                f"Got {len(edges)} dimenions but {len(dimlabels)} labels")
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

        bin_idxs = []
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
            bin_idxs.append(np.digitize(val, bins_for_key) - 1)
        ret = factors[tuple(bin_idxs)]
        if counts is not None:
            for count in reversed(counts):
                ret = ak.unflatten(ret, count)
        return ret

    @property
    def dimlabels(self):
        return self._bins.keys()


class MuonScaleFactor:
    def __init__(self, nominal, stat, syst):
        self.nominal = nominal
        self.stat = stat
        self.syst = syst

    def __call__(self, variation="central", **kwargs):
        if variation not in ("central", "up", "down", "syst up", "syst down",
                             "stat up", "stat down"):
            raise ValueError(f"Invalid variation '{variation}'")
        if variation in ("central", "up", "down"):
            sf = self.nominal
        else:
            unctype, variation = variation.split(" ")
            if unctype == "syst":
                sf = self.syst
            else:
                sf = self.stat
        return sf(variation=variation, **kwargs)

    @property
    def dimlabels(self):
        return self.nominal.dimlabels


WpTuple = namedtuple("WpTuple", ("loose", "medium", "tight"))


BTAG_WP_CUTS = {
    "deepcsv": {
        "2016": WpTuple(0.2217,  0.6321,  0.8953),
        "2017": WpTuple(0.1522,  0.4941,  0.8001),
        "2018": WpTuple(0.1241,  0.4184,  0.7527),
        "ul2016pre": WpTuple(0.2027, 0.6001, 0.8819),
        "ul2016post": WpTuple(0.1918, 0.5847, 0.8767),
        "ul2017": WpTuple(0.1355, 0.4506, 0.7738),
        "ul2018": WpTuple(0.1208, 0.4168, 0.7665),
    },
    "deepjet": {
        "2016": WpTuple(0.0614,  0.3093,  0.7221),
        "2017": WpTuple(0.0521,  0.3033,  0.7489),
        "2018": WpTuple(0.0494,  0.2770,  0.7264),
        "ul2016pre": WpTuple(0.0508, 0.2598, 0.6502),
        "ul2016post": WpTuple(0.048, 0.2489, 0.6377),
        "ul2017": WpTuple(0.0532, 0.3040, 0.7476),
        "ul2018": WpTuple(0.0490, 0.2783, 0.7100),
    }
}


BTAG_TAGGER_NAMES = {
    # tagger names with capitalization as it appears in the correction JSON
    "deepcsv": "deepCSV",
    "deepjet": "deepJet"
}


class BTagWeighter:
    def __init__(self, sf_filename, eff_filename, tagger, year,
                 method="fixedwp", meastype="mujets",
                 ignore_missing=False):

        if isinstance(method, str):
            method = method.lower()
            possible_methods = ("iterativefit", "fixedwp")
            if method not in possible_methods:
                raise ValueError(
                    "Method must be one of: " + ", ".join(possible_methods)
                )
        else:
            raise TypeError(f"Expected str for method, got {method}")

        if method == "fixedwp" and meastype == "iterativefit":
            raise ValueError(
                f"Meastype cannot be {meastype} when using fixedWP method"
            )

        tagger = tagger.lower()

        if sf_filename.endswith(".csv"):
            self.filetype = "csv"
            self.sf = []
            if method == "fixedwp":
                self.eff_evaluator = get_evaluator(eff_filename)
                for i in range(3):
                    # Note that for light flavor normally only inclusive is
                    # available, thus we fix it to 'incl' here
                    self.sf.append(BTagScaleFactor(
                        sf_filename, i, f"{meastype},{meastype},incl"))
            else:
                self.sf.append(BTagScaleFactor(
                    sf_filename, BTagScaleFactor.RESHAPE,
                    "iterativefit,iterativefit,iterativefit"))
        elif sf_filename.endswith(".json") or sf_filename.endswith(".json.gz"):
            self.filetype = "json"
            if method == "fixedwp":
                self.eff_evaluator = get_evaluator(eff_filename)
                correset = correctionlib.CorrectionSet.from_file(sf_filename)
                tagger_json = BTAG_TAGGER_NAMES[tagger]
                self.sf = [correset[f"{tagger_json}_{meastype}"],
                           correset[f"{tagger_json}_incl"]]
            else:
                raise NotImplementedError(
                    "iterativefit with json format is not implemented")

        self.method = method
        self.wps = BTAG_WP_CUTS[tagger][year]
        self.ignore_missing = ignore_missing

    def _fixedwp(self, wp, jf, eta, pt, discr, variation, efficiency):
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
            raise TypeError(f"Expected int or str for wp, got {wp}")
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
        sf = self.sf[wp].eval(
            heavy_vari, jf, eta, pt, discr, self.ignore_missing)
        if light_vari != heavy_vari:
            sf = ak.where(
                jf >= 4, sf,
                self.sf[wp].eval(
                    light_vari, jf, eta, pt, discr, self.ignore_missing))

        eff = self.eff_evaluator[efficiency](jf, pt, abs(eta))
        sfeff = sf * eff
        is_tagged = discr > self.wps[wp]

        p_mc = ak.prod(eff[is_tagged], axis=1) * ak.prod(
            (1 - eff)[~is_tagged], axis=1)
        p_data = ak.prod(sfeff[is_tagged], axis=1) * ak.prod(
            (1 - sfeff)[~is_tagged], axis=1)

        # TODO: What if one runs into numerical problems here?
        return p_data / p_mc

    def _fixedwp_json(self, wp, jf, eta, pt, discr, variation, efficiency):
        def evaluate_two_flavor(jf, abseta, pt):
            is_light = jf == 0
            isnt_light = ~is_light
            sf = np.empty(len(is_light))
            sf[isnt_light] = self.sf[0].evaluate(
                heavy_vari, wp, jf[isnt_light], abseta[isnt_light],
                pt[isnt_light])
            sf[is_light] = self.sf[1].evaluate(
                light_vari, wp, jf[is_light], abseta[is_light], pt[is_light])
            return sf

        wp = wp.lower()
        wp_val = getattr(self.wps, wp)
        if wp == "loose":
            wp = "l"
        elif wp == "medium":
            wp = "m"
        elif wp == "tight":
            wp = "t"
        else:
            raise ValueError("Invalid value for wp. Expected 'loose', "
                             "'medium' or 'tight'")
        wp = wp.upper()
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
        abseta = abs(eta)
        sf = onedimeval(evaluate_two_flavor, jf, abseta, pt)

        eff = self.eff_evaluator[efficiency](jf, pt, abseta)
        sfeff = sf * eff
        is_tagged = discr > wp_val

        p_mc = ak.prod(eff[is_tagged], axis=1) * ak.prod(
            (1 - eff)[~is_tagged], axis=1)
        p_data = ak.prod(sfeff[is_tagged], axis=1) * ak.prod(
            (1 - sfeff)[~is_tagged], axis=1)

        return p_data / p_mc

    def _iterativefit(self, wp, jf, eta, pt, discr, variation):
        possible_variations = (
            "central",
            "up_lf",
            "down_lf",
            "up_hf",
            "down_hf",
            "up_hfstats1",
            "down_hfstats1",
            "up_lfstats1",
            "down_lfstats1",
            "up_hfstats2",
            "down_hfstats2",
            "up_lfstats2",
            "down_lfstats2",
            "up_cferr1",
            "up_cferr2",
            "down_cferr1",
            "down_cferr2",
            "up_jes",
            "down_jes",
            "up_AbsoluteMPFBias",
            "down_AbsoluteMPFBias",
            "up_AbsoluteScale",
            "down_AbsoluteScale",
            "up_AbsoluteStat",
            "down_AbsoluteStat",
            "up_RelativeBal",
            "down_RelativeBal",
            "up_RelativeFSR",
            "down_RelativeFSR",
            "up_RelativeJEREC1",
            "down_RelativeJEREC1",
            "up_RelativeJEREC2",
            "down_RelativeJEREC2",
            "up_RelativeJERHF",
            "down_RelativeJERHF",
            "up_RelativePtBB",
            "down_RelativePtBB",
            "up_RelativePtEC1",
            "down_RelativePtEC1",
            "up_RelativePtEC2",
            "down_RelativePtEC2",
            "up_RelativePtHF",
            "down_RelativePtHF",
            "up_RelativeStatEC",
            "down_RelativeStatEC",
            "up_RelativeStatFSR",
            "down_RelativeStatFSR",
            "up_RelativeStatHF",
            "down_RelativeStatHF",
            "up_PileUpDataMC",
            "down_PileUpDataMC",
            "up_PileUpPtBB",
            "down_PileUpPtBB",
            "up_PileUpPtEC1",
            "down_PileUpPtEC1",
            "up_PileUpPtEC2",
            "down_PileUpPtEC2",
            "up_PileUpPtHF",
            "down_PileUpPtHF",
            "up_PileUpPtRef",
            "down_PileUpPtRef",
            "up_FlavorQCD",
            "down_FlavorQCD",
            "up_Fragmentation",
            "down_Fragmentation",
            "up_SinglePionECAL",
            "down_SinglePionECAL",
            "up_SinglePionHCAL",
            "down_SinglePionHCAL",
            "up_TimePtEta",
            "down_TimePtEta"
        )
        if variation not in possible_variations:
            raise ValueError(
                "variation must be one of: "
                + ", ".join(possible_variations))
        sf = self.sf[0].eval(
            variation, jf, eta, pt, discr, self.ignore_missing)
        # Apply a SF of 1. to c-flavored jets
        is_c = abs(jf) == 4
        if variation == "central":
            sf = ak.fill_none(ak.mask(sf, ~is_c), 1.0)
        # Apply cferr uncertainties only to c-flavored jets,
        # otherwise use central variation
        if "cferr" in variation:
            sf_central = self.sf[0].eval(
                "central", jf, eta, pt, discr, self.ignore_missing)
            sf = ak.where(is_c, sf, sf_central)
        # Apply lf, hf and jes uncertainties only to b- and
        # usdg-flavored jets, otherwise use central variation
        if "lf" in variation or "hf" in variation:
            sf_central = self.sf[0].eval(
                "central", jf, eta, pt, discr, self.ignore_missing)
            sf = ak.where(is_c, sf_central, sf)
        sf = ak.prod(sf, axis=1)
        return sf

    def __call__(
            self, wp, jf, eta, pt, discr, variation="central",
            efficiency="central"):
        if self.method == "fixedwp" and self.filetype == "csv":
            sf = self._fixedwp(
                wp, jf, eta, pt, discr, variation, efficiency)
        elif self.method == "fixedwp" and self.filetype == "json":
            sf = self._fixedwp_json(
                wp, jf, eta, pt, discr, variation, efficiency)
        elif self.method == "iterativefit":
            sf = self._iterativefit(wp, jf, eta, pt, discr, variation)
        else:
            raise ValueError("Unknown method or filetype")
        return sf

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
        return np.sqrt(sf * antisf) * self.scale
