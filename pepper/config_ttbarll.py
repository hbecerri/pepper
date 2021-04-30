#!/usr/bin/env python3

import numpy as np
import uproot
import hjson
import coffea
from functools import partial

import pepper
from pepper.scale_factors import (TopPtWeigter, PileupWeighter, BTagWeighter,
                                  get_evaluator, ScaleFactors)
from pepper import HistDefinition


class ConfigTTbarLL(pepper.Config):
    def __init__(self, path_or_file, textparser=hjson.load, cwd="."):
        """Initialize the configuration.

        Arguments:
        path_or_file -- Either a path to the file containing the configuration
                        or a file-like object of it
        textparser -- Callable to be used to parse the text contained in
                      path_or_file
        cwd -- A path to use as the working directory for relative paths in the
               config. The actual working directory of the process might change
        """
        super().__init__(path_or_file, textparser, cwd)
        self.behaviors.update({
            "year": self._get_year,
            "top_pt_reweighting": self._get_top_pt_reweighting,
            "pileup_reweighting": self._get_pileup_reweighting,
            "electron_sf": partial(
                self._get_scalefactors, dimlabels=["eta", "pt"]),
            "muon_sf": partial(
                self._get_scalefactors, dimlabels=["pt", "abseta"]),
            "btag_sf": self._get_btag_sf,
            "jet_correction": self._get_jet_correction,
            "jet_uncertainty": partial(
                self._get_jet_general, evaltype="junc",
                cls=coffea.jetmet_tools.JetCorrectionUncertainty),
            "jet_resolution": partial(
                self._get_jet_general, evaltype="jr",
                cls=coffea.jetmet_tools.JetResolution),
            "jet_ressf": partial(
                self._get_jet_general, evaltype="jersf",
                cls=coffea.jetmet_tools.JetResolutionScaleFactor),
            "MET_xy_shifts": self._get_maybe_external,
            "mc_lumifactors": self._get_maybe_external,
            "crosssection_uncertainty": self._get_maybe_external,
            "hists": self._get_hists,
            "drellyan_sf": self._get_drellyan_sf,
            "reco_info_file": self._get_path,
            "store": self._get_path,
            "lumimask": self._get_maybe_external,
            "rng_seed_file": self._get_path,
            "trigger_sfs": self._get_trigger_sfs
        })

    def _get_scalefactors(self, value, dimlabels):
        sfs = []
        for sfpath in value:
            if not isinstance(sfpath, list) or len(sfpath) < 2:
                raise pepper.config.ConfigError(
                    "scale factors needs to be list of 2-element-lists in "
                    "form of [rootfile, histname]")
            with uproot.open(self._get_path(sfpath[0])) as f:
                hist = f[sfpath[1]]
            sf = ScaleFactors.from_hist(hist, dimlabels)
            sfs.append(sf)
        return sfs

    @staticmethod
    def _get_year(value):
        year = str(value)
        if year not in ("2016", "2017", "2018"):
            raise pepper.config.ConfigError("Invalid year {}".format(year))
        return year

    @staticmethod
    def _get_top_pt_reweighting(value):
        return TopPtWeigter(**value)

    def _get_pileup_reweighting(self, value):
        with uproot.open(self._get_path(value)) as f:
            weighter = PileupWeighter(f)
        return weighter

    def _get_btag_sf(self, value):
        weighters = []
        tagger = self["btag"].split(":")[0]
        year = self["year"]
        for weighter_paths in value:
            paths = [self._get_path(path) for path in weighter_paths]
            btagweighter = BTagWeighter(
                paths[0], paths[1], tagger=tagger, year=year)
            weighters.append(btagweighter)
        return weighters

    def _get_jet_correction(self, value):
        evaluators = {}
        for path in value:
            path = self._get_path(path)
            evaluators.update(get_evaluator(path, "txt", "jec"))
        fjc = coffea.jetmet_tools.FactorizedJetCorrector(**evaluators)
        return fjc

    def _get_jet_general(self, value, evaltype, cls):
        evaluator = get_evaluator(self._get_path(value), "txt", evaltype)
        return cls(**evaluator)

    def _get_hists(self, value):
        hist_config = self._get_maybe_external(value)
        return {k: HistDefinition(c) for k, c in hist_config.items()}

    def _get_drellyan_sf(self, value):
        if isinstance(value, list):
            path, histname = value
            with uproot.open(self._get_path(path)) as f:
                hist = f[histname]
            dy_sf = ScaleFactors.from_hist(hist)
        else:
            with open(self._get_path(value)) as f:
                data = hjson.load(f)
            dy_sf = ScaleFactors(
                bins=data["bins"],
                factors=np.array(data["factors"]),
                factors_up=np.array(data["factors_up"]),
                factors_down=np.array(data["factors_down"]))
        return dy_sf

    def _get_trigger_sfs(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            with open(self._get_path(value)) as f:
                value = hjson.load(f)
        if isinstance(value, list):
            path, histnames = value
            if isinstance(histnames, str):
                with uproot.open(self._get_path(path)) as f:
                    hist = f[histnames]
                return ScaleFactors.from_hist(
                    hist, dimlabels=["lep1_pt", "lep2_pt"])
            elif isinstance(histnames[0], str):
                # histnames is a list of hists per channel
                with uproot.open(self._get_path(path)) as f:
                    hists = [f[histname] for histname in histnames]
                return ScaleFactors.from_hist_per_bin(
                    hists, dimlabels=["lep1_pt", "lep2_pt"],
                    channel=[0, 1, 2, 3])
            else:
                # histnames is a list of lists per channel of hists dependent
                # on the number of electrons in the barrel
                with uproot.open(self._get_path(path)) as f:
                    hists = [[f[histname] for histname in h]
                             for h in histnames]
                return ScaleFactors.from_hist_per_bin(
                    hists, dimlabels=["lep1_pt", "lep2_pt"],
                    channel=[0, 1, 2, 3], e_in_barrel=[0, 1, 2, 3])
        else:
            if "bins" in value:
                bins = value["bins"]
            else:
                bins = {"channel": [0, 1, 2, 3]}
            if "factors_up" in value:
                factors_up = np.array(value["factors_up"])
                factors_down = np.array(value["factors_down"])
            elif "errors" in value:
                factors_up = (np.array(value["factors"])
                              + np.array(value["errors"]))
                factors_down = (np.array(value["factors"])
                                - np.array(value["errors"]))
            else:
                raise pepper.config.ConfigError(
                    "trigger_sfs must contain one of 'factors_up' or 'errors'")
            return ScaleFactors(factors=np.array(value["factors"]),
                                factors_up=factors_up,
                                factors_down=factors_down,
                                bins=bins)
