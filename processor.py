#!/usr/bin/env python3

import os
import numpy as np
import awkward
import coffea
from coffea.analysis_objects import JaggedCandidateArray as Jca
import coffea.processor as processor
from functools import partial

import utils
import AdUtils


class LazyTable(object):
    """Wrapper for LazyDataFrame to allow slicing"""
    def __init__(self, df, slic=None):
        self._df = df
        self._slice = slic

    def _mergeslice(self, slic):
        if self._slice is not None:
            return self._slice[slic]
        else:
            return slic

    def __getitem__(self, key):
        if isinstance(key, slice):
            idx = np.arange(key.indices(self.size))
        elif isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise IndexError("too many indices for table")
            if key.dtype == np.bool:
                if key.size > self.size:
                    raise IndexError("boolean index too long")
                idx = np.argwhere(key).flatten()
            elif key.dtype == np.int:
                outofbounds = key > self.size
                if any(outofbounds):
                    raise IndexError("index {} is out of bounds"
                                     .format(
                                        key[np.argmax(outofbounds)]))
                idx = key
            else:
                raise IndexError("numpy arrays used as indices must be "
                                 "of intger or boolean type")
        else:
            arr = self._df[key]
            if self._slice is not None:
                return arr[self._slice]
            else:
                return arr

        return LazyTable(self._df, self._mergeslice(idx))

    def __setitem__(self, key, value):
        self._df[key] = value

    def __delitem__(self, key):
        del self._df[key]

    def __contains__(self, key):
        return key in self._df

    @property
    def size(self):
        if self._slice is None:
            return self._df.size
        else:
            return self._slice.size


class Selector(object):
    """Keeps track of the current event selection and data"""

    def __init__(self, table, weight_col=None):
        """Create a new Selector

        Arguments:
        table -- An `awkward.Table` or `LazyTable` holding the events' data
        """
        self.table = table
        self._cuts = AdUtils.PackedSelectionAccumulator()
        self._current_cuts = []
        self._frozen = False

        self._cutflow = processor.defaultdict_accumulator(int)
        self._weight_col = weight_col

        # Add a dummy cut to inform about event number and circumvent error
        # when calling all or require before adding actual cuts
        self._cuts.add_cut("Preselection", np.full(self.table.size, True))
        self._add_cutflow("Events preselection")

    @property
    def masked(self):
        """Get currently selected events

        Returns an `awkward.Table` of the currently selected events
        """
        if len(self._current_cuts) > 0:
            return self.table[self._cur_sel]
        else:
            return self.table

    def freeze_selection(self):
        """Freezes the selection

        After a call to this method, additional cuts wont effect the current
        selection anymore.
        """

        self._frozen = True

    def _add_cutflow(self, name):
        passing_all = self._cuts.all(*(self._cuts.names or []))
        if self._weight_col and self._weight_col in self.table:
            num = self.table[self._weight_col][passing_all].sum()
        else:
            num = passing_all.sum()
        self._cutflow[name] = num

    @property
    def _cur_sel(self):
        """Get a bool mask describing the current selection"""
        return self._cuts.all(*self._current_cuts)

    @property
    def num_selected(self):
        return self._cur_idx.sum()

    @property
    def cutflow(self):
        """Return an ordered dict of cut name -> selected events"""
        return self._cutflow

    def add_cut(self, accept, name):
        """Adds a cut

        Cuts control what events get fed into later cuts, get saved and are
        given by `masked`.

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
        accepted = accept(self.masked)
        if len(self._current_cuts) > 0:
            cut = np.full(self.table.size, False)
            cut[self._cur_sel] = accepted
        else:
            cut = accepted
        self._cuts.add_cut(name, cut)
        self._add_cutflow(name)
        if not self._frozen:
            self._current_cuts.append(name)

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
        data = column(self.masked)

        # Convert data to appropriate type if possible
        if isinstance(data, awkward.ChunkedArray):
            data = awkward.concatenate(data.chunks)

        # Move data into the table with appropriate padding (important!)
        if isinstance(data, np.ndarray):
            unmasked_data = np.empty(self.table.size, dtype=data.dtype)
            unmasked_data[self._cur_sel] = data
        elif isinstance(data, awkward.JaggedArray):
            counts = np.zeros(self.table.size, dtype=int)
            counts[self._cur_sel] = data.counts
            cls = awkward.array.objects.Methods.maybemixin(type(data),
                                                           awkward.JaggedArray)
            unmasked_data = cls.fromcounts(counts, data.flatten())
        else:
            raise TypeError("Unsupported column type {}".format(type(data)))
        self.table[column_name] = unmasked_data

    def save_columns(self, part_props={}, other_cols=[], cuts="Current"):
        """Save the currently selected events

        Arguments:
        p4s - names of particle columns for which one wishes to save all
              the components of the four-momentum
        other_cols - The other column one wishes to save

        Returns:
        flat_dict - A defaultdict_accumulator containing accumulators of
                    all the flat columns to be saved
        jagged_dict - A defaultdict_accumulator containing accumulators
                      of all the jagged columns to be saved
        """
        if cuts == "Current":
            cuts = self._current_cuts
        return_dict= processor.defaultdict_accumulator(
            AdUtils.ArrayAccumulator)
        for part in part_props.keys():
            for prop in part_props[part]:
                if prop == "p4":
                    return_dict[part + "_pt"].value =\
                        self.table[self._cuts.all(*cuts)][part].pt
                    return_dict[part + "_eta"].value =\
                        self.table[self._cuts.all(*cuts)][part].eta
                    return_dict[part + "_phi"].value =\
                        self.table[self._cuts.all(*cuts)][part].phi
                    return_dict[part + "_mass"].value =\
                        self.table[self._cuts.all(*cuts)][part].mass
                else:
                    return_dict[part + "_" + prop].value =\
                        self.table[self._cuts.all(*cuts)][part][prop]
        for col in other_cols:
            return_dict[col].value = self.table[self._cuts.all(*cuts)][col]
        return return_dict

    def save_cuts(self, cuts="Current"):
        if cuts == "Current":
            cuts = self._current_cuts
        return self._cuts.mask_self(self._cuts.all(*cuts))

def get_trigger_paths_for(dataset, is_mc, trigger_paths, trigger_order=None):
    """Get trigger paths needed for the specific dataset.

    Arguments:
    dataset -- Name of the dataset
    trigger_paths -- dict mapping dataset names to their triggers
    trigger_order -- List of datasets to define the order in which the triggers
                     are applied.

    Returns a tuple of lists (pos_triggers, neg_triggers) describing trigger
    paths to include and to exclude respectively.
    """
    pos_triggers = []
    neg_triggers = []
    if is_mc:
        for paths in trigger_paths.values():
            pos_triggers.extend(paths)
    else:
        for dsname in trigger_order:
            if dsname == dataset:
                break
            neg_triggers.extend(trigger_paths[dsname])
        pos_triggers = trigger_paths[dataset]
    return list(dict.fromkeys(pos_triggers)), list(dict.fromkeys(neg_triggers))


class Processor(processor.ProcessorABC):
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

    @property
    def accumulator(self):
        self._accumulator = processor.dict_accumulator({
            "cutflow": processor.defaultdict_accumulator(
                partial(processor.defaultdict_accumulator, int)),
            "cols to save": processor.defaultdict_accumulator( partial(
                processor.defaultdict_accumulator, AdUtils.ArrayAccumulator)),
            "cut arrays": processor.defaultdict_accumulator(
                AdUtils.PackedSelectionAccumulator)
        })
        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator

    def process(self, df):
        output = self.accumulator.identity()
        selector = Selector(LazyTable(df), "genWeight")

        dsname = df["dataset"]
        is_mc = (dsname in self.config["mc_datasets"].keys())
        selector.add_cut(partial(self.good_lumimask, is_mc), "Lumi")

        pos_triggers, neg_triggers = get_trigger_paths_for(
            dsname, is_mc, self.trigger_paths, self.trigger_order)
        selector.add_cut(partial(
            self.passing_trigger, pos_triggers, neg_triggers), "Trigger")

        selector.add_cut(partial(self.met_filters, is_mc), "MET filters")

        selector.set_column(self.good_electron, "is_good_electron")
        selector.set_column(self.good_muon, "is_good_muon")
        selector.set_column(self.build_lepton_column, "Lepton")
        selector.add_cut(self.exactly_lepton_pair, "#Lep = 2")
        selector.add_cut(self.opposite_sign_lepton_pair, "Opposite sign")
        selector.set_column(self.same_flavor, "is_same_flavor")

        selector.freeze_selection()

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

        output["cols to save"][dsname] = selector.save_columns(
            part_props={"Lepton":["p4", "pdgId"], "Jet":["p4"]},
            other_cols=["MET_sumEt", "weight"])
        output["cut arrays"][dsname]=selector.save_cuts()
        output["cutflow"][dsname] = selector.cutflow
        return output

    def good_lumimask(self, is_mc, data):
        if is_mc:
            return np.full(len(data["genWeight"]), True)
        else:
            run = np.array(data["run"])
            luminosityBlock = np.array(data["luminosityBlock"])
            return self.lumimask(run, luminosityBlock)

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
            passing_filters =\
                (data["Flag_goodVertices"]
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
        leptons = Jca.candidatesfromoffsets(offsets, **lep_dict)

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
        jets = Jca.candidatesfromoffsets(offsets, **lep_dict)

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

        ret = np.full(data.size, False)
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
        n = np.zeros(data.size)
        for i, pt_min in enumerate(self.config["lep_pt_min"]):
            mask = data["Lepton"].counts > i
            n[mask] += (pt_min < data["Lepton"].pt[mask, i]).astype(int)
        return n >= self.config["lep_pt_num_satisfied"]

    def good_mll(self, data):
        return ((data["Lepton"].p4[:, 0] + data["Lepton"].p4[:, 1]).mass
                > self.config["mll_min"])

    def no_additional_leptons(self, data):
        e_sel = ~data["is_good_electron"]
        add_ele = (self.electron_id(self.config["additional_ele_id"], data)
                   & e_sel)

        m_iso = self.config["additional_muon_iso"]
        m_sel = ~data["is_good_muon"]
        add_muon = (self.muon_id(self.config["additional_muon_id"], data)
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
        n = np.zeros(data.size)
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
        weight = np.ones(data.size)
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
