import sys
from functools import partial
from collections import namedtuple
import numpy as np
import awkward as ak
import logging
from copy import copy

import pepper
import pepper.config


if sys.version_info >= (3, 7):
    VariationArg = namedtuple(
        "VariationArgs", ["name", "junc", "jer", "met"],
        defaults=(None, "central", "central"))
else:
    # defaults in nampedtuple were introduced in python 3.7
    # As soon as CMSSW offers 3.7 or newer, remove this
    class VariationArg(
            namedtuple("VariationArg", ["name", "junc", "jer", "met"])):
        def __new__(cls, name, junc=None, jer="central", met="central"):
            return cls.__bases__[0].__new__(cls, name, junc, jer, met)

logger = logging.getLogger(__name__)


class Processor(pepper.ProcessorBasicPhysics):
    """Processor for the Top-pair with dileptonic decay selection"""

    config_class = pepper.ConfigTTbarLL

    def __init__(self, config, eventdir):
        """Create a new Processor

        Arguments:
        config -- A Config instance, defining the configuration to use
        eventdir -- Destination directory, where the event HDF5s are saved.
                    Every chunk will be saved in its own file. If `None`,
                    nothing will be saved.
        """
        super().__init__(config, eventdir)

    def _check_config_integrity(self, config):
        """Check integrity of configuration file."""

        super()._check_config_integrity(config)

        if "drellyan_sf" not in config:
            logger.warning("No Drell-Yan scale factor specified")

        if "trigger_sfs" not in config:
            logger.warning("No trigger scale factors specified")

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)
        if dsname.startswith("TTTo"):
            selector.set_column("gent_lc", self.gentop, lazy=True)
            if "top_pt_reweighting" in self.config:
                selector.add_cut(
                    "Top pt reweighting", self.do_top_pt_reweighting,
                    no_callback=True)
        if is_mc and "pileup_reweighting" in self.config:
            selector.add_cut("Pileup reweighting", partial(
                self.do_pileup_reweighting, dsname))
        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)
        if is_mc:
            selector.add_cut(
                "Cross section", partial(self.crosssection_scale, dsname))

        if "blinding_denom" in self.config:
            selector.add_cut("Blinding", partial(self.blinding, is_mc))
        selector.add_cut("Lumi", partial(self.good_lumimask, is_mc, dsname))

        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"], era=era)
        selector.add_cut("Trigger", partial(
            self.passing_trigger, pos_triggers, neg_triggers))
        if is_mc and self.config["year"] in ("2016", "2017", "ul2016pre",
                                             "ul2016post", "ul2017"):
            selector.add_cut("L1 prefiring", self.add_l1_prefiring_weights)

        selector.add_cut("MET filters", partial(self.met_filters, is_mc))

        selector.add_cut("No add leps",
                         partial(self.no_additional_leptons, is_mc))
        selector.set_column("Electron", self.pick_electrons)
        selector.set_column("Muon", self.pick_muons)
        selector.set_column("Lepton", partial(
            self.build_lepton_column, is_mc, selector.rng))
        # Wait with hists filling after channel masks are available
        selector.add_cut("At least 2 leps", partial(self.lepton_pair, is_mc),
                         no_callback=True)
        filler.set_cat("channels", {"is_ee", "is_em", "is_mm"})
        selector.set_multiple_columns(self.channel_masks)
        selector.set_column("mll", self.mass_lepton_pair)
        selector.set_column("dilep_pt", self.dilep_pt, lazy=True)

        selector.applying_cuts = False

        selector.add_cut("Opposite sign", self.opposite_sign_lepton_pair)
        selector.add_cut("Chn trig match",
                         partial(self.channel_trigger_matching, era))
        if "trigger_sfs" in self.config and is_mc:
            selector.add_cut(
                "Trigger SFs", partial(self.apply_trigger_sfs, dsname))
        selector.add_cut("Req lep pT", self.lep_pt_requirement)
        selector.add_cut("m_ll", self.good_mass_lepton_pair)
        selector.add_cut("Z window", self.z_window,
                         categories={"channels": ["is_ee", "is_mm"]})

        if (is_mc and self.config["compute_systematics"]
                and dsname not in self.config["dataset_for_systematics"]):
            if hasattr(filler, "sys_overwrite"):
                assert filler.sys_overwrite is None
            cats = filler.cats
            for variarg in self.get_jetmet_variation_args():
                selector_copy = copy(selector)
                filler.sys_overwrite = variarg.name
                self.process_selection_jet_part(selector_copy, is_mc,
                                                variarg, dsname, filler, era)
                if self.eventdir is not None:
                    logger.debug(f"Saving per event info for variation"
                                 f" {variarg.name}")
                    self.save_per_event_info(
                        dsname + "_" + variarg.name, selector_copy, False)
                filler.cats = cats
            filler.sys_overwrite = None

        # Do normal, no-variation run
        self.process_selection_jet_part(selector, is_mc,
                                        self.get_jetmet_nominal_arg(),
                                        dsname, filler, era)
        logger.debug("Selection done")

    def process_selection_jet_part(self, selector, is_mc, variation, dsname,
                                   filler, era):
        logger.debug(f"Running jet_part with variation {variation.name}")
        reapply_jec = ("reapply_jec" in self.config
                       and self.config["reapply_jec"])
        selector.set_multiple_columns(partial(
            self.compute_jet_factors, is_mc, reapply_jec, variation.junc,
            variation.jer, selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", self.build_jet_column)
        smear_met = "smear_met" in self.config and self.config["smear_met"]
        selector.set_column(
            "MET", partial(self.build_met_column, is_mc, variation.junc,
                           variation.jer if smear_met else None, selector.rng,
                           era, variation=variation.met))
        selector.set_multiple_columns(
            partial(self.drellyan_sf_columns, filler))
        if "drellyan_sf" in self.config and is_mc:
            selector.add_cut("DY scale", partial(self.apply_dy_sfs, dsname))
        selector.add_cut("Has jet(s)", self.has_jets)
        if (self.config["hem_cut_if_ele"] or self.config["hem_cut_if_muon"]
                or self.config["hem_cut_if_jet"]):
            selector.add_cut("HEM cut", self.hem_cut)
        selector.add_cut("Jet pt req", self.jet_pt_requirement)
        if is_mc and self.config["compute_systematics"]:
            self.scale_systematics_for_btag(selector, variation, dsname)
        selector.add_cut("Has btag(s)", partial(self.btag_cut, is_mc))
        selector.add_cut("Req MET", self.met_requirement,
                         categories={"channels": ["is_ee", "is_mm"]})

        if "reco_algorithm" in self.config:
            reco_alg = self.config["reco_algorithm"]
            selector.set_column("recolepton", self.pick_lepton_pair,
                                all_cuts=True)
            selector.set_column("recob", self.pick_bs_from_lepton_pair,
                                all_cuts=True)
            selector.set_column("recot", partial(
                self.ttbar_system, reco_alg.lower(), selector.rng),
                all_cuts=True, no_callback=True)
            selector.add_cut("Reco", self.has_ttbar_system)
            selector.set_column("reconu", self.build_nu_column_ttbar_system,
                                all_cuts=True, lazy=True)
            selector.set_column("dark_pt", self.calculate_dark_pt,
                                all_cuts=True, lazy=True)
            selector.set_column("chel", self.calculate_chel, all_cuts=True,
                                lazy=True)

    def channel_masks(self, data):
        leps = data["Lepton"]
        channels = {}
        channels["is_ee"] = ((abs(leps[:, 0].pdgId) == 11)
                             & (abs(leps[:, 1].pdgId) == 11))
        channels["is_mm"] = ((abs(leps[:, 0].pdgId) == 13)
                             & (abs(leps[:, 1].pdgId) == 13))
        channels["is_em"] = (~channels["is_ee"]) & (~channels["is_mm"])
        return channels

    def dilep_pt(self, data):
        return (data["Lepton"][:, 0] + data["Lepton"][:, 1]).pt

    def apply_trigger_sfs(self, dsname, data):
        leps = data["Lepton"]
        ones = np.ones(len(data))
        central = ones
        channels = ["is_ee", "is_em", "is_mm"]
        trigger_sfs = self.config["trigger_sfs"]
        for channel in channels:
            sf = trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                      lep2_pt=leps[:, 1].pt)
            central = ak.where(data[channel], sf, central)
        if self.config["compute_systematics"]:
            up = ones
            down = ones
            for channel in channels:
                sf = trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                          lep2_pt=leps[:, 1].pt,
                                          variation="up")
                up = ak.where(data[channel], sf, up)
                sf = trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                          lep2_pt=leps[:, 1].pt,
                                          variation="down")
                down = ak.where(data[channel], sf, down)
            return central, {"triggersf": (up / central, down / central)}
        return central

    def apply_dy_sfs(self, dsname, data):
        if dsname.startswith("DY"):
            channel = ak.where(data["is_ee"], 0, ak.where(data["is_em"], 1, 2))
            if ("bin_dy_sfs" in self.config and
                    self.config["bin_dy_sfs"] is not None):
                params = {
                    "channel": channel, "axis":
                    pepper.hist_defns.DataPicker(self.config["bin_dy_sfs"])}
            else:
                params = {"channel": channel}
            dysf = self.config["drellyan_sf"]
            central = dysf(**params)
            if self.config["compute_systematics"]:
                up = dysf(**params, variation="up")
                down = dysf(**params, variation="down")
                return central, {"DYsf": (up / central, down / central)}
            return central
        elif self.config["compute_systematics"]:
            ones = np.ones(len(data))
            return np.full(len(data), True), {"DYsf": (ones, ones)}
        return np.full(len(data), True)

    def drellyan_sf_columns(self, data, filler):
        # Dummy function, overwritten when computing DY SFs
        return {}

    def channel_trigger_matching(self, era, data):
        is_ee = data["is_ee"]
        is_mm = data["is_mm"]
        is_em = data["is_em"]
        triggers = self.config["channel_trigger_map"]

        ret = np.full(len(data), False)
        check = [
            (is_ee, "ee"), (is_mm, "mumu"), (is_em, "emu"),
            (is_ee | is_em, "e"), (is_mm | is_em, "mu")]
        for mask, channel in check:
            if (channel + "_" + era) in triggers:
                trigger = [pepper.misc.normalize_trigger_path(t)
                           for t in triggers[channel + "_" + era]]
                ret = ret | (
                    mask & self.passing_trigger(trigger, [], data))
            elif channel in triggers:
                trigger = [pepper.misc.normalize_trigger_path(t)
                           for t in triggers[channel]]
                ret = ret | (
                    mask & self.passing_trigger(trigger, [], data))

        return ret

    def z_window(self, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        is_out_window = (data["mll"] <= m_min) | (m_max <= data["mll"])
        return is_out_window

    def met_requirement(self, data):
        met = data["MET"].pt
        return met > self.config["ee/mm_min_met"]

    def build_nu_column(self, data):
        lep = data["recolepton"][:, 0:1]
        antilep = data["recolepton"][:, 1:2]
        b = data["recob"][:, 0:1]
        antib = data["recob"][:, 1:2]
        top = data["recot"][:, 0:1]
        antitop = data["recot"][:, 1:2]
        nu = top - b - antilep
        antinu = antitop - antib - lep
        return ak.concatenate([nu, antinu], axis=1)

    def calculate_dark_pt(self, data):
        nu = data["reconu"][:, 0]
        antinu = data["reconu"][:, 1]
        met = data["MET"]
        return met - nu - antinu

    def calculate_chel(self, data):
        top = data["recot"]
        lep = data["recolepton"]
        ttbar_boost = -top.sum().boostvec
        top = top.boost(ttbar_boost)
        lep = lep.boost(ttbar_boost)

        top_boost = -top.boostvec
        lep_ZMFtbar = lep[:, 0].boost(top_boost[:, 1])
        lbar_ZMFtop = lep[:, 1].boost(top_boost[:, 0])

        chel = lep_ZMFtbar.dot(lbar_ZMFtop) / lep_ZMFtbar.rho / lbar_ZMFtop.rho
        return chel
