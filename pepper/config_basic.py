#!/usr/bin/env python3

import uproot
import hjson
import coffea
from coffea import lookup_tools
from functools import partial

import pepper
from pepper.scale_factors import (
    TopPtWeigter,
    PileupWeighter,
    BTagWeighter,
    get_evaluator,
    ScaleFactors,
)
from pepper import HistDefinition


class ConfigBasicPhysics(pepper.Config):
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
        self.behaviors.update(
            {
                "year": self._get_year,
                "top_pt_reweighting": self._get_top_pt_reweighting,
                "pileup_reweighting": self._get_pileup_reweighting,
                "electron_sf": self._get_scalefactors,
                "muon_sf": self._get_scalefactors,
                "muon_rochester": self._get_rochester_corr,
                "btag_sf": self._get_btag_sf,
                "jet_correction_mc": self._get_jet_correction,
                "jet_correction_data": self._get_jet_correction,
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
                "reco_info_file": self._get_path,
                "store": self._get_path,
                "lumimask": self._get_path,
                "rng_seed_file": self._get_path,
            }
        )

    def _get_scalefactors(self, value):
        sfs = []
        for sfpath in value:
            if not isinstance(sfpath, list) or len(sfpath) < 2:
                raise pepper.config.ConfigError(
                    "scale factors needs to be list of 2-element-lists in "
                    "form of [rootfile, histname]"
                )
            with uproot.open(self._get_path(sfpath[0])) as f:
                hist = f[sfpath[1]]
            sf = ScaleFactors.from_hist(hist, sfpath[2])
            sfs.append(sf)
        return sfs

    @staticmethod
    def _get_year(value):
        year = str(value)
        if year not in (
            "2016",
            "2017",
            "2018",
            "ul2016pre",
            "ul2016post",
            "ul2017",
            "ul2018",
        ):
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
                paths[0], paths[1], tagger=tagger, year=year
            )
            weighters.append(btagweighter)
        return weighters

    def _get_btag_corr(self, value):
        # Dump content of hjson file to dict
        if value == "":
            return None
        elif "$DATADIR" in value:
            value = value.replace(
                "$DATADIR", self._config[self.special_vars["$DATADIR"]]
            )
            with open(value) as jf:
                data = hjson.load(jf)
            return data

    def _get_rochester_corr(self, value):
        path = self._get_path(value)
        rochester_data = lookup_tools.txt_converters.convert_rochester_file(
            path, loaduncs=False
        )
        rochester = lookup_tools.rochester_lookup.rochester_lookup(
            rochester_data
        )
        return rochester

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
