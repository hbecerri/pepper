#!/usr/bin/env python3

import os
import sys
import numpy as np
import awkward
import h5py
import uproot
import coffea
from coffea.analysis_objects import JaggedCandidateArray as CandArray
from time import time
import random
from functools import partial
from collections import OrderedDict

import utils


class Selector(object):
    """Keeps track of the current event selection and data"""

    def __init__(self, table):
        """Create a new Selector

        Arguments:
        table -- An awkward.Table holding the events' data
        """
        self.table = table
        self.mask = np.full(self.table.shape, True)
        self._recording = False
        self._cutbitpos = 0
        self._cutflow = OrderedDict({"Input": self.num_after_all_cuts})

    @property
    def masked(self):
        """Get currently selected events

        Returns an awkward.Table of the currently selected events
        """
        return self.table[self.mask]

    @property
    def num_selected(self):
        """Get the number of currently selected events

        Stays the same after the first call of start_recording.
        """
        return self.mask.sum()

    @property
    def num_after_all_cuts(self):
        """Get the number of events after all cuts applied"""
        if self._cutbitpos == 0:
            return self.num_selected
        else:
            allcutflags = (1 << self._cutbitpos) - 1
            return (self.masked["cutflags"] == allcutflags).sum()

    @property
    def is_recording(self):
        """Return whether the selector is currently recording"""
        return self._recording

    @property
    def cutflow(self):
        """Return an ordered dict of cut name -> selected events"""
        return self._cutflow

    def add_cut(self, accept, name):
        """Adds a cut to the current selection

        If the selector is not recording, the current selection will get
        modified according to the cut. Otherwise the cut will be recorded in
        a cutflags bit in the table.

        Arguments:
        accept -- A function that will be called with a table of the currently
                  selected events. The function should return a numpy array of
                  the same length as the table holding bools, indicating if an
                  event is not cut (True). Does not get called if num_selected
                  is 0 already.
        name -- A label to assoiate within the cutflow
        """
        if name in self._cutflow:
            raise ValueError("A cut with name {} exists already".format(name))
        if self.mask.any():
            accepted = accept(self.masked)
            if not self._recording:
                self.mask[self.mask] &= accepted
            else:
                flag = accepted.astype(int) << self._cutbitpos
                self.table["cutflags"][self.mask] = (self.masked["cutflags"]
                                                     | flag)
                self._cutbitpos += 1
        self._cutflow[name] = self.num_after_all_cuts

    def set_column(self, column, column_name):
        """Sets a column of the table

        Arguments:
        column -- A function that will be called with a table of the currently
                  selected events. It should return a numpy array or an
                  awkward.JaggedArray with the same length as the table.
                  Does not get called if num_selected is 0 already.
        column_name -- The name of the column to set

        """
        if not isinstance(column_name, str):
            raise ValueError("column_name needs to be string")
        if not self.mask.any():
            data = np.array([])
        else:
            data = column(self.masked)
        if isinstance(data, np.ndarray):
            unmasked_data = np.empty_like(self.mask, dtype=data.dtype)
            unmasked_data[self.mask] = data
        elif isinstance(data, awkward.JaggedArray):
            counts = np.zeros(self.mask.shape, dtype=int)
            counts[self.mask] = data.counts
            cls = awkward.array.objects.Methods.maybemixin(type(data),
                                                           awkward.JaggedArray)
            unmasked_data = cls.fromcounts(counts, data.flatten())
        else:
            raise TypeError("Unsupported column type {}".format(type(data)))
        self.table[column_name] = unmasked_data

    def start_recording(self):
        """Start recording

        This changes the behavior of add_cut.
        """
        self._recording = True
        self.table["cutflags"] = np.zeros(self.table.shape, dtype=int)

    def save_columns(self, columns, path, savecutflow=True):
        """Save the currently selected events

        Arguments:
        columns -- Iteratable of column names from the table to save
        path -- Path to the output file which to save to. The file will get
                overwritten if it exists already.
        savecutflow -- bool, whether to save the numbers of events after every
                       cut
        """
        with h5py.File(path, "w") as f:
            out = awkward.hdf5(f)
            for column in columns:
                out[column] = self.masked[column]
            if savecutflow:
                out["cutflow"] = list(self.cutflow.items())


def get_outpath(path, dest):
    if path.startswith("/pnfs/desy.de/cms/tier2/store/"):
        outpath = path.replace("/pnfs/desy.de/cms/tier2/store/", "")
    else:
        outpath = path
    outpath = os.path.splitext(outpath)[0] + ".hdf5"
    outpath = os.path.join(dest, outpath)
    return outpath


def skip_existing(dest, path):
    return os.path.exists(get_outpath(path, dest))


def get_trigger_paths_for(dataset, trigger_paths, trigger_order=None):
    """Get trigger paths needed for the specific dataset.

    Arguments:
    dataset -- Name of the dataset
    trigger_paths -- dict mapping dataset names to their triggers
    trigger_order -- List of datasets to define the order in which the triggers
                     are applied. Optional if dataset == "all"

    Returns a tuple of lists (pos_triggers, neg_triggers) describing trigger
    paths to include and to exclude respectively. if dataset == "all", all
    trigger paths will be returned as pos_triggers, while neg_triggers will be
    empty.
    """
    pos_triggers = []
    neg_triggers = []
    if dataset == "MC" or dataset == "all":
        for paths in trigger_paths.values():
            pos_triggers.extend(paths)
    else:
        for dsname in trigger_order:
            if dsname == dataset:
                break
            neg_triggers.extend(trigger_paths[dsname])
        pos_triggers = trigger_paths[dataset]
    return list(dict.fromkeys(pos_triggers)), list(dict.fromkeys(neg_triggers))


class Processor(ProcessorABC):
    def __init__(self, config):
        self.config = config

        if "lumimask" in config:
            self.lumimask = self.config["lumimask"]
        else:
            print("No lumimask specified")
            self.lumimask = None
        if ("electron_sf" in self.config
                and len(self.config["electron_sf"]) > 0):
            self.electron_sf = self.config["electron_sf"]
        else:
            print("No electron scale factors specified")
            self.electron_sf = []
        if "muon_sf" in self.config and len(self.config["muon_sf"]) > 0:
            self.muon_sf = self.config["muon_sf"]
        else:
            print("No muon scale factors specified")
            self.muon_sf = []

        self.trigger_paths = config["dataset_trigger_map"]
        self.trigger_order = config["dataset_trigger_order"]
        missing_triggers = (set(datasets.keys())
                            - set(self.trigger_paths.keys())
                            - set(["MC"]))
        if len(missing_triggers) > 0:
            raise utils.ConfigError("Missing triggers for: {}"
                                    .format(", ".join(missing_triggers)))

        self.starttime = 0

    def process(self, df):
        self.starttime = time()
        for i, (path, data) in enumerate(uproot.iterate(paths2dsname.keys(),
                                                        "Events",
                                                        self.branches,
                                                        namedecode="utf-8",
                                                        reportpath=True)):
            data = awkward.Table(data)
            selector = Selector(data)
            dsname = paths2dsname[path]

            is_mc = dsname == "MC"
            selector.add_cut(partial(self.good_lumimask, is_mc), "Lumi")

            pos_triggers, neg_triggers = get_trigger_paths_for(
                dsname, self.trigger_paths, self.trigger_order)
            selector.add_cut(partial(
                self.passing_trigger, pos_triggers, neg_triggers), "Trigger")

            selector.add_cut(partial(self.met_filters, is_mc), "MET filters")

            selector.set_column(self.good_electron, "is_good_electron")
            selector.set_column(self.good_muon, "is_good_muon")
            selector.set_column(self.build_lepton_column, "Lepton")
            selector.add_cut(self.exactly_lepton_pair, "#Lep = 2")
            selector.add_cut(self.opposite_sign_lepton_pair, "Opposite sign")
            selector.set_column(self.same_flavor, "is_same_flavor")

            selector.start_recording()

            selector.add_cut(self.channel_trigger_matching, "Chn. trig. match")
            selector.add_cut(self.lep_pt_requirement, "Req lep pT")
            selector.add_cut(self.good_mll, "M_ll")
            selector.add_cut(self.no_additional_leptons, "No add. leps")
            selector.add_cut(self.z_window, "Z window")

            selector.set_column(self.good_jet, "is_good_jet")
            selector.set_column(self.build_jet_column, "Jet")
            selector.add_cut(self.has_jets, "#Jets >= n")
            selector.add_cut(self.jet_pt_requirement, "Req jet pT")
            selector.add_cut(self.btag, "B-tag")
            selector.add_cut(self.met_requirement, "Req MET")

            selector.set_column(partial(self.compute_weight, is_mc), "weight")

            for key in ["pt", "eta", "phi", "mass"]:
                selector.set_column(lambda d: getattr(d["Lepton"], key),
                                    "Lepton_" + key)
                selector.set_column(lambda d: getattr(d["Jet"], key),
                                    "Jet_" + key)
            selector.set_column(lambda d: d["Lepton"].pdgId, "Lepton_pdgId")

            outpath = get_outpath(path, args.dest)
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            selector.save_columns(["Lepton_pt",
                                   "Lepton_eta",
                                   "Lepton_phi",
                                   "Lepton_mass",
                                   "Lepton_pdgId",
                                   "Jet_pt",
                                   "Jet_eta",
                                   "Jet_phi",
                                   "Jet_mass",
                                   "MET_sumEt",
                                   "weight",
                                   "cutflags"], outpath)

            now = time()
            print("[{}/{}] Saved {} of {} events from \"{}\". This took "
                  "{:.1f} s.".format(i + 1,
                                     num_files,
                                     selector.num_selected,
                                     data.shape[0],
                                     path,
                                     now - self.starttime))
            self.starttime = now

    def branches_for_e_id(self, e_id):
        if e_id.startswith("cut:"):
            return "Electron_cutBased"
        elif e_id == "mva:wp80":
            return "Electron_WP80"
        elif e_id == "mva:wp90":
            return "Electron_WP90"
        return None

    def branches_for_m_id(self, m_id):
        if m_id == "cut:loose":
            return "Muon_looseId"
        elif m_id == "cut:medium":
            return "Muon_mediumId"
        elif m_id == "cut:tight":
            return "Muon_tightId"
        elif m_id.startswith("mva:"):
            return "Muon_mvaId"
        return None

    def branches_for_j_id(self, j_id):
        if j_id.startswith("cut:"):
            return "Jet_jetId"
        return None

    def branches(self, branch):
        # Read branches if they exist. The less branches are read,
        # the shorter the execution time.
        # Needs uproot version 3.10.10
        req = [
            "run",  # Needed for lumimask
            "luminosityBlock",  # Needed for lumimask
            "genWeight",  # Needed for weighting and distinguishing MC and data
            "Electron_pt",
            "Electron_eta",
            "Electron_deltaEtaSC",
            "Electron_phi",
            "Electron_mass",
            "Electron_pdgId",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_pdgId",
            "Muon_pfRelIso04_all",
            "Jet_pt",
            "Jet_eta",
            "Jet_phi",
            "Jet_mass",
            "MET_sumEt"
        ]

        if self.config["filter_year"].lower() != "none":
            req.extend(["Flag_goodVertices",
                        "Flag_globalSuperTightHalo2016Filter",
                        "Flag_HBHENoiseFilter",
                        "Flag_HBHENoiseIsoFilter",
                        "Flag_EcalDeadCellTriggerPrimitiveFilter",
                        "Flag_BadPFMuonFilter",
                        "Flag_eeBadScFilter",
                        "Flag_ecalBadCalibFilterV2"])

        req.append(self.branches_for_e_id(self.config["good_ele_id"]))
        req.append(self.branches_for_e_id(self.config["additional_ele_id"]))
        req.append(self.branches_for_m_id(self.config["good_muon_id"]))
        req.append(self.branches_for_m_id(self.config["additional_muon_id"]))
        req.append(self.branches_for_j_id(self.config["good_jet_id"]))

        if self.config["btag"].startswith("deepcvs"):
            req.append("Jet_btagDeepB")
        elif self.config["btag"].startswith("deepjet"):
            req.append("Jet_btagDeepFlavB")

        req.extend(get_trigger_paths_for("all", self.trigger_paths)[0])
        for paths in self.config["channel_trigger_map"].values():
            req.extend(paths)

        return branch.name.decode("utf-8") in req

    def good_lumimask(self, is_mc, data):
        if is_mc:
            return True
        else:
            return self.lumimask(data["run"], data["luminosityBlock"])

    def passing_trigger(self, pos_triggers, neg_triggers, data):
        trigger = (
            np.any([data[trigger_path] for trigger_path in pos_triggers],
                   axis=0)
            & ~np.any([data[trigger_path] for trigger_path in neg_triggers],
                      axis=0)
        )
        return trigger

    def mpv_quality(self, data):
        # Does not include check for fake. Is this even needed?
        R = np.hypot(data["PV_x"], data["PV_y"])
        return ((data["PV_chi2"] != 0)
                & (data["PV_ndof"] > 4)
                & (abs(data["PV_z"]) <= 24)
                & (R <= 2))

    def met_filters(self, is_mc, data):
        filter_year = str(self.config["filter_year"]).lower()
        if filter_year == "none":
            return np.full(data.shape, True)
        elif filter_year in ("2018", "2017", "2016"):
            passing_filters = (data["Flag_goodVertices"]
                               & data["Flag_globalSuperTightHalo2016Filter"]
                               & data["Flag_HBHENoiseFilter"]
                               & data["Flag_HBHENoiseIsoFilter"]
                               & data["Flag_EcalDeadCellTriggerPrimitiveFilter"]
                               & data["Flag_BadPFMuonFilter"])
            if not is_mc:
                passing_filters &= data["Flag_eeBadScFilter"]
        else:
            raise utils.ConfigError("Invalid filter year: {}".format(
                filter_year))
        if filter_year in ("2018", "2017"):
            passing_filters &= data["Flag_ecalBadCalibFilterV2"]

        return passing_filters

    def in_hem1516(self, phi, eta):
        return ((-2.4 < eta) & (eta < -1.6) & (-1.4 < phi) & (phi < -1.0))

    def in_transreg(self, abs_eta):
        return (1.4442 < abs_eta) & (abs_eta < 1.566)

    def electron_id(self, e_id, data):
        if e_id == "skip":
            has_id = True
        if e_id == "cut:loose":
            has_id = data["Electron_cutBased"] >= 2
        elif e_id == "cut:medium":
            has_id = data["Electron_cutBased"] >= 3
        elif e_id == "cut:tight":
            has_id = data["Electron_cutBased"] >= 4
        elif e_id == "mva:wp80":
            has_id = data["Electron_WP80"]
        elif e_id == "mva:wp90":
            has_id = data["Electron_WP90"]
        else:
            raise ValueError("Invalid electron id string")
        return has_id

    def good_electron(self, data):
        if self.config["good_ele_cut_hem"]:
            is_in_hem1516 = self.in_hem1516(data["Electron_phi"],
                                            data["Electron_eta"])
        else:
            is_in_hem1516 = np.array(False)
        if self.config["good_ele_cut_transreg"]:
            SC_eta_abs = abs(data["Electron_eta"]
                             + data["Electron_deltaEtaSC"])
            is_in_transreg = self.in_transreg(SC_eta_abs)
        else:
            is_in_transreg = np.array(False)
        e_id, eta_min, eta_max, pt_min, pt_max = self.config[[
            "good_ele_id", "good_ele_eta_min", "good_ele_eta_max",
            "good_ele_pt_min", "good_ele_pt_max"]]
        return (self.electron_id(e_id, data)
                & (~is_in_hem1516)
                & (~is_in_transreg)
                & (eta_min < data["Electron_eta"])
                & (data["Electron_eta"] < eta_max)
                & (pt_min < data["Electron_pt"])
                & (data["Electron_pt"] < pt_max))

    def muon_id(self, m_id, data):
        if m_id == "skip":
            has_id = True
        elif m_id == "cut:loose":
            has_id = data["Muon_looseId"]
        elif m_id == "cut:medium":
            has_id = data["Muon_mediumId"]
        elif m_id == "cut:tight":
            has_id = data["Muon_tightId"]
        elif m_id == "mva:loose":
            has_id = data["Muon_mvaId"] >= 1
        elif m_id == "mva:medium":
            has_id = data["Muon_mvaId"] >= 2
        elif m_id == "mva:tight":
            has_id = data["Muon_mvaId"] >= 3
        else:
            raise ValueError("Invalid muon id string")
        return has_id

    def good_muon(self, data):
        if self.config["good_muon_cut_hem"]:
            is_in_hem1516 = self.in_hem1516(data["Muon_phi"], data["Muon_eta"])
        else:
            is_in_hem1516 = np.array(False)
        if self.config["good_muon_cut_transreg"]:
            is_in_transreg = self.in_transreg(abs(data["Muon_eta"]))
        else:
            is_in_transreg = np.array(False)
        m_id, eta_min, eta_max, pt_min, pt_max, iso = self.config[[
            "good_muon_id", "good_muon_eta_min", "good_muon_eta_max",
            "good_muon_pt_min", "good_muon_pt_max", "good_muon_iso"]]
        return (self.muon_id(m_id, data)
                & (data["Muon_pfRelIso04_all"] < iso)
                & (~is_in_hem1516)
                & (~is_in_transreg)
                & (eta_min < data["Muon_eta"])
                & (data["Muon_eta"] < eta_max)
                & (pt_min < data["Muon_pt"])
                & (data["Muon_pt"] < pt_max))

    def build_lepton_column(self, data):
        keys = ["pt", "eta", "phi", "mass", "pdgId"]
        lep_dict = {}
        ge = data["is_good_electron"]
        gm = data["is_good_muon"]
        for key in keys:
            arr = awkward.concatenate([data["Electron_" + key][ge],
                                       data["Muon_" + key][gm]], axis=1)
            offsets = arr.offsets
            lep_dict[key] = arr.flatten()
        leptons = CandArray.candidatesfromoffsets(offsets, **lep_dict)

        # Sort leptons by pt
        leptons = leptons[leptons.pt.argsort()]

        return leptons

    def require_emu(self, data):
        return data["is_good_electron"].any() & data["is_good_muon"].any()

    def exactly_lepton_pair(self, data):
        return data["Lepton"].counts == 2

    def opposite_sign_lepton_pair(self, data):
        return (np.sign(data["Lepton"][:, 0].pdgId)
                != np.sign(data["Lepton"][:, 1].pdgId))

    def same_flavor(self, data):
        return (abs(data["Lepton"][:, 0].pdgId)
                == abs(data["Lepton"][:, 1].pdgId))

    def good_jet(self, data):
        j_id, lep_dist, eta_min, eta_max, pt_min, pt_max = self.config[[
            "good_jet_id", "good_jet_lepton_distance", "good_jet_eta_min",
            "good_jet_eta_max", "good_jet_pt_min", "good_jet_pt_max"]]
        if j_id == "skip":
            has_id = True
        elif j_id == "cut:loose":
            has_id = (data["Jet_jetId"] & 0b1).astype(bool)
            # Always False in 2017 and 2018
        elif j_id == "cut:tight":
            has_id = (data["Jet_jetId"] & 0b10).astype(bool)
        elif j_id == "cut:tightlepveto":
            has_id = (data["Jet_jetId"] & 0b100).astype(bool)
        else:
            raise utils.ConfigError("Invalid good_jet_id: {}".format(j_id))

        lep_dist = self.config["good_jet_lepton_distance"]

        j_eta = data["Jet_eta"]
        j_phi = data["Jet_phi"]
        l_eta = awkward.concatenate(
            [data["Electron_eta"], data["Muon_eta"]], axis=1)
        l_phi = awkward.concatenate(
            [data["Electron_phi"], data["Muon_phi"]], axis=1)
        j_eta, l_eta = j_eta.cross(l_eta, nested=True).unzip()
        j_phi, l_phi = j_phi.cross(l_phi, nested=True).unzip()
        delta_eta = j_eta - l_eta
        delta_phi = j_phi - l_phi
        delta_R = np.hypot(delta_eta, delta_phi)
        has_lepton_close = (delta_R < lep_dist).any()

        return (has_id
                & (~has_lepton_close)
                & (eta_min < data["Jet_eta"])
                & (data["Jet_eta"] < eta_max)
                & (pt_min < data["Jet_pt"])
                & (data["Jet_pt"] < pt_max))

    def build_jet_column(self, data):
        keys = ["pt", "eta", "phi", "mass"]
        lep_dict = {}
        gj = data["is_good_jet"]
        for key in keys:
            arr = data["Jet_" + key][gj]
            offsets = arr.offsets
            lep_dict[key] = arr.flatten()
        jets = CandArray.candidatesfromoffsets(offsets, **lep_dict)

        # Sort jets by pt
        jets = jets[jets.pt.argsort()]

        return jets

    def channel_trigger_matching(self, data):
        p0 = abs(data["Lepton"].pdgId[:, 0])
        p1 = abs(data["Lepton"].pdgId[:, 1])
        is_ee = (p0 == 11) & (p1 == 11)
        is_mm = (p0 == 13) & (p1 == 13)
        is_em = (~is_ee) & (~is_mm)
        triggers = self.config["channel_trigger_map"]

        ret = np.full(data.shape, False)
        if "ee" in triggers:
            ret |= is_ee & self.passing_trigger(triggers["ee"], [], data)
        if "mumu" in triggers:
            ret |= is_mm & self.passing_trigger(triggers["mumu"], [], data)
        if "emu" in triggers:
            ret |= is_em & self.passing_trigger(triggers["emu"], [], data)
        if "e" in triggers:
            ret |= is_ee & self.passing_trigger(triggers["e"], [], data)
            ret |= is_em & self.passing_trigger(triggers["e"], [], data)
        if "mu" in triggers:
            ret |= is_mm & self.passing_trigger(triggers["mu"], [], data)
            ret |= is_em & self.passing_trigger(triggers["mu"], [], data)

        return ret

    def lep_pt_requirement(self, data):
        n = np.zeros(data.shape)
        for i, pt_min in enumerate(self.config["lep_pt_min"]):
            mask = data["Lepton"].counts > i
            n[mask] += (pt_min < data["Lepton"].pt[mask, i]).astype(int)
        return n >= self.config["lep_pt_num_satisfied"]

    def good_mll(self, data):
        return ((data["Lepton"].p4[:, 0] + data["Lepton"].p4[:, 1]).mass
                > self.config["mll_min"])

    def no_additional_leptons(self, data):
        e_sel = ~data["is_good_electron"]
        add_ele = self.electron_id(config["additional_ele_id"], data) & e_sel

        m_iso = config["additional_muon_iso"]
        m_sel = ~data["is_good_muon"]
        add_muon = (self.muon_id(config["additional_muon_id"], data)
                    & (data["Muon_pfRelIso04_all"] < m_iso)
                    & m_sel)
        return (~add_ele.any()) & (~add_muon.any())

    def z_window(self, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        is_sf = data["is_same_flavor"]
        invmass = (data["Lepton"].p4[:, 0] + data["Lepton"].p4[:, 1]).mass
        return ~is_sf | ((invmass <= m_min) & (m_max <= invmass))

    def has_jets(self, data):
        return self.config["num_jets_atleast"] <= data["Jet"].counts

    def jet_pt_requirement(self, data):
        n = np.zeros(data.shape)
        for i, pt_min in enumerate(self.config["jet_pt_min"]):
            mask = data["Jet"].counts > i
            n[mask] += (pt_min < data["Jet"].pt[mask, i]).astype(int)
        return n >= self.config["jet_pt_num_satisfied"]

    def btag(self, data):
        tagger, wp = self.config["btag"].split(":")
        if tagger == "deepcsv":
            disc = data["Jet_btagDeepB"]
            wps = {"loose": 0.1241, "medium": 0.4184, "tight": 0.7527}
            if wp not in wps:
                raise utils.ConfigError("Invalid DeepCSV working point: {}"
                                        .format(wp))
        elif tagger == "deepjet":
            disc = data["Jet_btagDeepFlavB"]
            wps = {"loose": 0.0494, "medium": 0.2770, "tight": 0.7264}
            if wp not in wps:
                raise utils.ConfigError("Invalid DeepJet working point: {}"
                                        .format(wp))
        else:
            raise utils.ConfigError("Invalid tagger name: {}".format(tagger))
        btagged = disc > wps[wp]
        return btagged.sum() >= self.config["num_atleast_btagged"]

    def met_requirement(self, data):
        is_sf = data["is_same_flavor"]
        met = data["MET_sumEt"]
        return ~is_sf | (met > self.config["ee/mm_min_met"])

    def compute_weight(self, is_mc, data):
        weight = np.ones(data.shape)
        if is_mc:
            weight *= data["genWeight"]
            electrons = data["Lepton"][abs(data["Lepton"].pdgId) == 11]
            muons = data["Lepton"][abs(data["Lepton"].pdgId) == 13]
            for sf in self.electron_sf:
                factors_flat = sf(eta=electrons.eta.flatten(),
                                  pt=electrons.pt.flatten())
                weight *= utils.jaggedlike(electrons.eta, factors_flat).prod()
            for sf in self.muon_sf:
                factors_flat = sf(abseta=abs(muons.eta.flatten()),
                                  pt=muons.pt.flatten())
                weight *= utils.jaggedlike(muons.eta, factors_flat).prod()
        return weight
