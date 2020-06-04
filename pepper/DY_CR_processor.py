import os
import sys
import numpy as np
import awkward
import uproot
import coffea
from coffea.analysis_objects import JaggedCandidateArray as Jca
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict, namedtuple
import shutil
import parsl
import json
import logging

import pepper
from pepper.misc import jcafromjagged, sortby
from pepper.datasets import expand_datasetdict

logger = logging.getLogger(__name__)

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

class DY_processor(pepper.Processor):
    def process_selection(self, selector, dsname, is_mc, filler):
        if dsname.startswith("TTTo"):
            selector.set_column(self.gentop, "gent_lc")
            if self.topptweighter is not None:
                self.do_top_pt_reweighting(selector)
        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)
        if is_mc:
            self.add_crosssection_scale(selector, dsname)

#        if self.config["blinding_denom"] is not None:
#            selector.add_cut(partial(self.blinding, is_mc), "Blinding")
        selector.add_cut(partial(self.good_lumimask, is_mc), "Lumi")

        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.trigger_paths, self.trigger_order)
        selector.add_cut(partial(
            self.passing_trigger, pos_triggers, neg_triggers), "Trigger")

        selector.add_cut(partial(self.met_filters, is_mc), "MET filters")

        selector.set_column(self.build_lepton_column, "Lepton")
        # Wait with hists filling after channel masks are available
        selector.add_cut(partial(self.lepton_pair, is_mc), "At least 2 leps",
                         no_callback=True)
        filler.channels = ("is_ee", "is_em", "is_mm")
        selector.set_multiple_columns(self.channel_masks)
        selector.set_column(self.mll, "mll")
        selector.set_column(self.dilep_pt, "dilep_pt")

        selector.freeze_selection()

        selector.add_cut(self.opposite_sign_lepton_pair, "Opposite sign")
        selector.add_cut(partial(self.no_additional_leptons, is_mc),
                         "No add. leps")
        selector.add_cut(self.channel_trigger_matching, "Chn. trig. match")
        selector.add_cut(self.lep_pt_requirement, "Req lep pT")
        selector.add_cut(self.good_mll, "M_ll")
        selector.add_cut(self.z_window, "Z window")

        # Do normal, no-variation run
        selector.set_column(self.build_jet_column, "Jet")
        selector.set_column(partial(self.build_met_column,
                                    variation=VariationArg(None).met), "MET")
        selector.add_cut(self.has_jets, "#Jets >= %d"
                         % self.config["num_jets_atleast"])
        if (self.config["hem_cut_if_ele"] or self.config["hem_cut_if_muon"]
                or self.config["hem_cut_if_jet"]):
            selector.add_cut(self.hem_cut, "HEM cut")
        selector.add_cut(self.jet_pt_requirement, "Jet pt req")
        selector.add_cut(partial(self.btag_cut, is_mc), "At least %d btag"
                         % self.config["num_atleast_btagged"])
        selector.add_cut(self.met_requirement, "MET > %d GeV"
                         % self.config["ee/mm_min_met"])

        logger.debug("Selection done")

    def z_window(self, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        is_out_window = (data["mll"] >= m_min) & (m_max >= data["mll"])
        return ~data["is_em"] & is_out_window
