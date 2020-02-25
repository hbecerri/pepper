#!/usr/bin/env python3

import os
from functools import partial
import numpy as np
import awkward
import coffea
import coffea.lumi_tools
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray as Jca
import coffea.processor as processor
import uproot
from uproot_methods import TLorentzVectorArray
import h5py
import time

import utils.config
import utils.misc
from utils.accumulator import PackedSelectionAccumulator
import utils.btagging
from kin_reco.sonnenschein import kinreco


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
            idx = np.arange(*key.indices(self.size))
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

    def __init__(self, table, weight=None, on_cutdone=None):
        """Create a new Selector

        Arguments:
        table -- An `awkward.Table` or `LazyTable` holding the events' data
        weight -- A 1d numpy array of size equal to `table` size, describing
                  the events' weight or None
        on_cutdown -- callable that gets called after a cut is done (`add_cut`)
        """
        self.table = table
        self._cuts = PackedSelectionAccumulator()
        self._current_cuts = []
        self._frozen = False
        self.on_cutdone = on_cutdone

        self._cutflow = processor.defaultdict_accumulator(int)
        self.channel_cutflows = None
        if weight is not None:
            tabled = awkward.Table({"weight": weight})
            counts = np.full(self.table.size, 1)
            self.systematics = awkward.JaggedArray.fromcounts(counts, tabled)
        else:
            self.systematics = None

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

    @property
    def weight(self):
        """Get the event weights for the currently selected events
        """
        if self.systematics is None:
            return None
        weight = self.systematics["weight"].flatten()
        if len(self._current_cuts) > 0:
            return weight[self._cur_sel]
        else:
            return weight

    @property
    def masked_systematics(self):
        """Get the systematics for the currently selected events

        Returns an `awkward.Table`, where "weight" maps to the event weight.
        All other columns are named by the scale factor they belong to.
        """
        if self.systematics is None:
            return None
        if len(self._current_cuts) > 0:
            return self.systematics[self._cur_sel]
        else:
            return self.systematics

    @property
    def final(self):
        """Get events which have passed all cuts
        (both those before and after freeze_selection)
        """
        return self.table[self._cuts.all(*self._cuts.names)]

    @property
    def final_systematics(self):
        """Get the systematics for the events which have passed all cuts
        """
        if self.systematics is None:
            return None
        return self.systematics[self._cuts.all(*self._cuts.names)]

    def freeze_selection(self):
        """Freezes the selection

        After a call to this method, additional cuts wont effect the current
        selection anymore.
        """

        self._frozen = True

    def _add_cutflow(self, name):
        passing_all = self._cuts.all(*(self._cuts.names or []))
        if self.systematics is not None:
            num = self.systematics["weight"][passing_all].flatten().sum()
        else:
            num = passing_all.sum()
        self._cutflow[name] = num

    @property
    def _cur_sel(self):
        """Get a bool mask describing the current selection"""
        return self._cuts.all(*self._current_cuts)

    @property
    def num_selected(self):
        return self._cur_sel.sum()

    @property
    def cutflow(self):
        """Return an ordered dict of cut name -> selected events"""
        return self._cutflow
 
    def _add_channel_cutflow(self, name):
        passing_all = self._cuts.all(*(self._cuts.names or []))
        for ch in self.channel_cutflows.keys():
            if self.systematics is not None:
                num = self.systematics["weight"][passing_all][np.where(self.final["Channel"] == ch)].flatten().sum()
            else:
                num = np.where(self.final["Channel"] == ch, True, False).sum()
            self.channel_cutflows[ch][name] = num

    def initalise_channel_cutflows(self, channels, ch_func):
        self.channel_cutflows=processor.dict_accumulator(
            {ch: processor.defaultdict_accumulator(int) for ch in channels})
        self.set_column(ch_func, "Channel")
        self._add_channel_cutflow(self._cuts.names[-1])

    def add_cut(self, accept, name):
        """Adds a cut

        Cuts control what events get fed into later cuts, get saved and are
        given by `masked`.

        Arguments:
        accept -- A function that will be called with a table of the currently
                  selected events. The function should return an array of bools
                  or a tuple of this array and a dict.
                  The array has the same length as the table and indicates if
                  an event is not cut (True).
                  The dict maps names of scale factors to a tuples of the form
                  (sf, (up, down)) where sf, up and down are arrays of floats
                  giving central, up and down variation for a scale factor for
                  each event, thus making only sense in case of MC. In case of
                  no up/down variations (sf, None) is a valid value. `up` and
                  `down` must be given relative to sf.
                  accept does not get called if num_selected is already 0.
        name -- A label to assoiate within the cutflow
        """
        if name in self._cutflow:
            raise ValueError("A cut with name {} already exists".format(name))
        accepted = accept(self.masked)
        if isinstance(accepted, tuple):
            accepted, weight = accepted
        else:
            weight = {}
        if len(self._current_cuts) > 0:
            cut = np.full(self.table.size, False)
            cut[self._cur_sel] = accepted
        else:
            cut = accepted
        self._cuts.add_cut(name, cut)
        self._add_cutflow(name)
        if self.channel_cutflows is not None:
            self._add_channel_cutflow(name)
        if not self._frozen:
            self._current_cuts.append(name)
            mask = None
        else:
            mask = accepted
        for weightname, factors in weight.items():
            self.modify_weight(weightname, factors[0], factors[1], mask)
        if self.on_cutdone is not None:
            self.on_cutdone(data=self.final,
                            systematics=self.final_systematics,
                            cut=name)

    def _pad_npcolumndata(self, data, defaultval=None, mask=None):
        padded = np.empty(self.table.size, dtype=data.dtype)
        if defaultval:
            padded[:] = defaultval
        if mask is not None:
            total_mask = self._cur_sel
            total_mask[self._cur_sel] = mask
            padded[total_mask] = data
        else:
            padded[self._cur_sel] = data
        return padded

    def set_systematic(self, name, up, down, mask=None):
        """Set the systematic up/down variation for a systematic given by
        `name`. `mask` is an array of bools and indicates, which events the
        systematic applies to. If `None`, the systematic applies to all events.
        `up` and `down` must be given relative to the central value.
        """
        if name == "weight":
            raise ValueError("The name of a systematic can't be 'weight'")
        self.systematics[name + "_up"] = self._pad_npcolumndata(up, 1, mask)
        self.systematics[name + "_down"] = self._pad_npcolumndata(
            down, 1, mask)

    def modify_weight(self, name, factor=None, updown=None, mask=None):
        """Modify the event weight. The weight will be multiplied by `factor`.
        `name` gives the name of the factor and is important to keep track of
        the systematics supplied by `updown`. If updown is not None, it should
        be a tuple of up and down variation factors relative to `factor`.
        `mask` is an array of bools and indicates, which events the
        systematic applies to.
        """
        if factor is not None:
            factor = self._pad_npcolumndata(factor, 1, mask)
            print(self.systematics["weight"], factor)
            time.sleep(10)
            self.systematics["weight"] = self.systematics["weight"] * factor
        if updown is not None:
            self.set_systematic(name, updown[0], updown[1], mask)

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
            unmasked_data = self._pad_npcolumndata(data)
        elif isinstance(data, awkward.JaggedArray):
            counts = np.zeros(self.table.size, dtype=int)
            counts[self._cur_sel] = data.counts
            cls = awkward.array.objects.Methods.maybemixin(type(data),
                                                           awkward.JaggedArray)
            unmasked_data = cls.fromcounts(counts, data.flatten())
        else:
            raise TypeError("Unsupported column type {}".format(type(data)))
        self.table[column_name] = unmasked_data

    def get_columns(self, part_props={}, other_cols=set(), cuts="Current",
                    prefix=""):
        """Get columns of events passing cuts

        Arguments:
        part_props- A dictionary of particles, followed by a list of properties
                     one wishes to save for those particles- "p4" will add all
                     components of the 4-momentum
        other_cols- The other columns one wishes to save
        cuts      - "Current", "All" or a list of cuts - the list of cuts to
                     apply before saving- The default, "Current", only applies
                     the cuts before freeze_selection
        prefix    - A string that gets prepended to every key in return_dict

        Returns:
        return_dict - A dict containing JaggedArrays or numpy arrays of the
                      columns
        """
        if cuts == "Current":
            cuts = self._current_cuts
        elif cuts == "All":
            cuts = self._cuts.names
        elif not isinstance(cuts, list):
            raise ValueError("cuts needs to be one of 'Current', 'All' or a "
                             "list")
        data = self.table[self._cuts.all(*cuts)]
        return_dict = {}
        for part in part_props.keys():
            props = set(part_props[part])
            if "p4" in props:
                props |= {"pt", "eta", "phi", "mass"}
            props -= {"p4"}
            for prop in props:
                if not hasattr(data[part], prop):
                    continue
                arr = utils.misc.jagged_reduce(getattr(data[part], prop))
                return_dict[part + "_" + prop] = arr
        for col in other_cols:
            if col not in data:
                continue
            return_dict[prefix + col] = utils.misc.jagged_reduce(data[col])
        if self.systematics is not None:
            return_dict["weight"] = self.weight
        return return_dict

    def get_columns_from_config(self, to_save, prefix=""):
        return self.get_columns(to_save["part_props"],
                                to_save["other_cols"],
                                to_save["cuts"],
                                prefix=prefix)

    def get_cuts(self, cuts="Current"):
        """Get information on what events pass which cuts

        Arguments:
        cuts -- "Current", "All" or a list of cuts - the list of cuts to
                apply before saving- The default, "Current", only applies
                the cuts before freeze_selection
        """
        if cuts == "Current":
            cuts = self._current_cuts
        elif cuts == "All":
            cuts = self._cuts.names
        elif not isinstance(cuts, list):
            raise ValueError("cuts needs to be one of 'Current', 'All' or a "
                             "list")
        return self._cuts.mask[self._cuts.all(*cuts)]


class Empty_err(Exception):
    pass


class Processor(processor.ProcessorABC):
    def __init__(self, config, destdir,
                 sel_hists=None, reco_hists=None):
        """Create a new Processor

        Arguments:
        config -- A Config instance, defining the configuration to use
        destdir -- Destination directory, where the event HDF5s are saved.
                   Every chunk will be saved in its own file. If `None`,
                   nothing will be saved.
        sel_hists -- A dictionary of histogram names (strings) to callables.
                     The callable should be in form of
                     `f(data, dsname, is_mc)`, should create a new Hist, fill
                     it if possible and return it.
        reco_hists -- Same as `sel_hists`, with the difference that the
                      callables get caleld after the reconstruction.
        """
        self.config = config
        self.destdir = destdir
        self.sel_hists = sel_hists if sel_hists is not None else {}
        self.reco_hists = reco_hists if reco_hists is not None else {}

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
        if "btag_sf" in self.config and len(self.config["btag_sf"]) > 0:
            self.btagweighters = config["btag_sf"]
        else:
            print("No btag scale factor specified")
            self.btagweighters = None

        if "jet_resolution" in self.config and "jet_ressf" in self.config:
            self._jer = self.config["jet_resolution"]
            self._jersf = self.config["jet_ressf"]
        else:
            print("No jet resolution or no jet resolution scale factor "
                  "specified")
            self._jer = None
            self._jersf = None

        self.trigger_paths = config["dataset_trigger_map"]
        self.trigger_order = config["dataset_trigger_order"]
        if "genhist_path" in self.config:
            self.hist_mlb_path = self.config["genhist_path"]
        else:
            print("No Mlb hist for picking b jets specified")

    @property
    def accumulator(self):
        self._accumulator = processor.dict_accumulator({
            "sel_hists": processor.dict_accumulator(),
            "reco_hists": processor.dict_accumulator(),
            "cutflow": processor.defaultdict_accumulator(
                partial(processor.defaultdict_accumulator, int)),
        })
        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator

    def _open_output(self, dsname):
        dsname = dsname.replace("/", "_")
        dsdir = os.path.join(self.destdir, dsname)
        os.makedirs(dsdir, exist_ok=True)
        i = 0
        while True:
            filepath = os.path.join(dsdir, str(i).zfill(4) + ".hdf5")
            try:
                f = h5py.File(filepath, "x")
            except (FileExistsError, OSError):
                i += 1
                continue
            else:
                break
        return f

    def _save_per_event_info(self, dsname, selector, reco_selector):
        with self._open_output(dsname) as f:
            outf = awkward.hdf5(f)
            out_dict = selector.get_columns_from_config(
                self.config["selector_cols_to_save"])
            out_dict.update(reco_selector.get_columns_from_config(
                self.config["reco_cols_to_save"], "reco"))
            out_dict["cutflags"] = selector.get_cuts()
            out_dict["cutflow"] = selector.cutflow

            for key in out_dict.keys():
                outf[key] = out_dict[key]

    def process(self, df):
        output = self.accumulator.identity()
        dsname = df["dataset"]
        is_mc = (dsname in self.config["mc_datasets"].keys())
        sel_cb = partial(self.fill_accumulator,
                         hist_dict=self.sel_hists,
                         accumulator=output["sel_hists"],
                         is_mc=is_mc,
                         dsname=dsname)
        if is_mc:
            genweight = df["genWeight"]
        else:
            genweight = None
        selector = Selector(LazyTable(df), genweight, sel_cb)

        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(selector)
            self.add_crosssection_uncertainties(selector, dsname)

        selector.add_cut(partial(self.blinding, is_mc), "Blinding")
        selector.add_cut(partial(self.good_lumimask, is_mc), "Lumi")

        pos_triggers, neg_triggers = utils.misc.get_trigger_paths_for(
            dsname, is_mc, self.trigger_paths, self.trigger_order)
        selector.add_cut(partial(
            self.passing_trigger, pos_triggers, neg_triggers), "Trigger")

        selector.add_cut(partial(self.met_filters, is_mc), "MET filters")

        selector.set_column(self.build_lepton_column, "Lepton")
        selector.initalise_channel_cutflows(["ee", "emu", "mumu", "None"], self.set_channels)
        selector.add_cut(partial(self.lepton_pair, is_mc), "At least 2 leps")
        selector.set_column(self.same_flavor, "is_same_flavor")
        selector.set_column(self.mll, "mll")

        selector.freeze_selection()

        selector.add_cut(self.opposite_sign_lepton_pair, "Opposite sign")
        selector.add_cut(partial(self.no_additional_leptons, is_mc),
                         "No add. leps")
        selector.add_cut(self.channel_trigger_matching, "Chn. trig. match")
        selector.add_cut(self.lep_pt_requirement, "Req lep pT")
        selector.add_cut(self.good_mll, "M_ll")
        selector.add_cut(self.z_window, "Z window")

        if is_mc and self._jer is not None and self._jersf is not None:
            selector.set_column(self.compute_jer_factor, "jerfac")
        selector.set_column(self.build_jet_column, "Jet")
        selector.set_column(self.build_met_column, "MET")
        selector.add_cut(self.hem_cut, "HEM cut")
        selector.add_cut(self.has_jets, "#Jets >= %d"
                         % self.config["num_jets_atleast"])
        selector.add_cut(self.jet_pt_requirement, "Jet pt req")
        selector.add_cut(self.btag_cut, "At least %d btag"
                         % self.config["num_atleast_btagged"])
        selector.add_cut(self.met_requirement, "MET > %d GeV"
                         % self.config["ee/mm_min_met"])

        if is_mc:
            if self.btagweighters is not None:
                selector.set_column(self.compute_weight_btag, "weight_btag")

        '''lep, antilep = self.pick_leps(selector.final)
        b, bbar = self.choose_bs(selector.final, lep, antilep)
        neutrino, antineutrino = kinreco(lep["p4"], antilep["p4"],
                                         b["p4"], bbar["p4"],
                                         selector.final["MET_pt"],
                                         selector.final["MET_phi"])
        if is_mc:
            weight = selector.final_systematics["weight"]
        else:
            weight = np.full(selector.final.size, 1.)
        reco_cb = partial(self.fill_accumulator,
                          hist_dict=self.reco_hists,
                          accumulator=output["reco_hists"],
                          is_mc=is_mc,
                          dsname=dsname)
        reco_objects = Selector(awkward.Table(lep=lep, antilep=antilep,
                                              b=b, bbar=bbar,
                                              neutrino=neutrino,
                                              antineutrino=antineutrino),
                                weight, reco_cb)
        reco_objects.add_cut(self.passing_reco, "Reco")
        reco_objects.set_column(self.wminus, "Wminus")
        reco_objects.set_column(self.wplus, "Wplus")
        reco_objects.set_column(self.top, "top")
        reco_objects.set_column(self.antitop, "antitop")
        reco_objects.set_column(self.ttbar, "ttbar")'''

        output["cutflow"][dsname] = selector.cutflow

        '''if self.destdir is not None:
            self._save_per_event_info(dsname, selector, reco_objects)'''

        return output

    def fill_accumulator(self, hist_dict, accumulator, is_mc, dsname, data,
                         systematics, cut):
        if systematics is not None:
            weight = systematics["weight"].flatten()
        else:
            weight = None
        for histname, fill_func in hist_dict.items():
            dsforsys = self.config["dataset_for_systematics"]
            nominal_hist = fill_func(data=data,
                                     dsname=dsname,
                                     is_mc=is_mc,
                                     weight=weight)
            # If this ds is dedicated for a systematic, save hist accordingly
            if dsname in dsforsys:
                # But only if we want to compute systematics
                if systematics is not None:
                    replacename, sysname = dsforsys[dsname]
                    accumulator[(cut, histname, sysname)] = nominal_hist
            else:
                accumulator[(cut, histname)] = nominal_hist

                if systematics is not None:
                    for syscol in systematics.columns:
                        if syscol == "weight":
                            continue
                        sysweight = weight * systematics[syscol]
                        hist = fill_func(data=data, dsname=dsname, is_mc=is_mc,
                                         weight=sysweight)
                        accumulator[(cut, histname, syscol)] = hist
                    # In order to have the hists specific to dedicated
                    # systematic datasets contain also all the events from
                    # unaffected datasets, copy the nominal hists
                    for sysds, (replace, sys) in dsforsys.items():
                        if nominal_hist is not None:
                            accumulator[(cut, histname, sys)] = nominal_hist.copy()

    def add_generator_uncertainies(self, selector):
        # Matrix-element renormalization and factorization scale
        # Get describtion of individual columns of this branch with
        # Events->GetBranch("LHEScaleWeight")->GetTitle() in ROOT
        data = selector.masked
        selector.set_systematic("MEren",
                                data["LHEScaleWeight"][:, 7],
                                data["LHEScaleWeight"][:, 1])
        selector.set_systematic("MEfac",
                                data["LHEScaleWeight"][:, 5],
                                data["LHEScaleWeight"][:, 3])
        # Parton shower scale
        selector.set_systematic("PSisr",
                                data["PSWeight"][:, 2],
                                data["PSWeight"][:, 0])
        selector.set_systematic("PSfsr",
                                data["PSWeight"][:, 3],
                                data["PSWeight"][:, 1])

    def add_crosssection_uncertainties(self, selector, dsname):
        xsuncerts = self.config["crosssection_uncertainty"]
        num_events = selector.num_selected
        for name, affected_datasets in xsuncerts.items():
            for affected_dataset, uncert in affected_datasets.items():
                if dsname == affected_dataset:
                    break
            else:
                uncert = 0
            selector.set_systematic(name + "XS",
                                    np.full(num_events, 1 + uncert),
                                    np.full(num_events, 1 - uncert))

    def blinding(self, is_mc, data):
        if (not is_mc) and (self.config["blinding_denom"] is not None):
            return np.where(np.mod(data["event"],
                                   self.config["blinding_denom"]) == 0,
                            True,
                            False)
        else:
            return np.full(len(data["event"]), True)

    def good_lumimask(self, is_mc, data):
        if is_mc:
            # Lumimask only present in data, all events pass in MC
            # Compute lumi variation here
            allpass = np.full(len(data["genWeight"]), True)
            if self.config["compute_systematics"]:
                weight = {}
                if (self.config["year"] == "2018"
                        or self.config["year"] == "2016"):
                    weight["lumi"] = (None,
                                      np.full(data.size, 1 + 0.025),
                                      np.full(data.size, 1 - 0.025))
                elif self.config["year"] == "2017":
                    weight["lumi"] = (None,
                                      np.full(data.size, 1 + 0.023),
                                      np.full(data.size, 1 - 0.023))
                return (allpass, weight)
            else:
                return allpass
        else:
            run = np.array(data["run"])
            luminosity_block = np.array(data["luminosityBlock"])
            lumimask = coffea.lumi_tools.LumiMask(self.lumimask)
            return lumimask(run, luminosity_block)

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
        year = str(self.config["year"]).lower()
        if not self.config["apply_met_filters"]:
            return np.full(data.shape, True)
        else:
            passing_filters =\
                (data["Flag_goodVertices"]
                 & data["Flag_globalSuperTightHalo2016Filter"]
                 & data["Flag_HBHENoiseFilter"]
                 & data["Flag_HBHENoiseIsoFilter"]
                 & data["Flag_EcalDeadCellTriggerPrimitiveFilter"]
                 & data["Flag_BadPFMuonFilter"])
            if not is_mc:
                passing_filters &= data["Flag_eeBadScFilter"]
        if year in ("2018", "2017"):
            passing_filters &= data["Flag_ecalBadCalibFilterV2"]

        return passing_filters

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
        elif e_id == "mva:noIso80":
            has_id = data["Electron_mvaFall17V2noIso_WP80"]
        elif e_id == "mva:noIso90":
            has_id = data["Electron_mvaFall17V2noIso_WP90"]
        elif e_id == "mva:Iso80":
            has_id = data["Electron_mvaFall17V2Iso_WP80"]
        elif e_id == "mva:Iso90":
            has_id = data["Electron_mvaFall17V2Iso_WP90"]
        else:
            raise ValueError("Invalid electron id string")
        return has_id

    def electron_cuts(self, data, good_lep):
        if self.config["ele_cut_transreg"]:
            sc_eta_abs = abs(data["Electron_eta"]
                             + data["Electron_deltaEtaSC"])
            is_in_transreg = self.in_transreg(sc_eta_abs)
        else:
            is_in_transreg = np.array(False)
        if good_lep:
            e_id, pt_min = self.config[[
                "good_ele_id", "good_ele_pt_min"]]
        else:
            e_id, pt_min = self.config[[
                "additional_ele_id", "additional_ele_pt_min"]]
        eta_min, eta_max = self.config[["ele_eta_min", "ele_eta_max"]]
        return (self.electron_id(e_id, data)
                & (~is_in_transreg)
                & (eta_min < data["Electron_eta"])
                & (data["Electron_eta"] < eta_max)
                & (pt_min < data["Electron_pt"]))

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

    def muon_iso(self, iso, data):
        if iso == "cut:very_loose":
            return data["Muon_pfIsoId"] > 0
        elif iso == "cut:loose":
            return data["Muon_pfIsoId"] > 1
        elif iso == "cut:medium":
            return data["Muon_pfIsoId"] > 2
        elif iso == "cut:tight":
            return data["Muon_pfIsoId"] > 3
        elif iso == "cut:very_tight":
            return data["Muon_pfIsoId"] > 4
        elif iso == "cut:very_very_tight":
            return data["Muon_pfIsoId"] > 5
        else:
            iso, iso_value = iso.split(":")
            value = float(iso_value)
            if iso == "dR<0.3_chg":
                return data["Muon_pfRelIso03_chg"] < value
            elif iso == "dR<0.3_all":
                return data["Muon_pfRelIso03_all"] < value
            elif iso == "dR<0.4_all":
                return data["Muon_pfRelIso04_all"] < value
        raise ValueError("Invalid muon iso string")

    def muon_cuts(self, data, good_lep):
        if self.config["muon_cut_transreg"]:
            is_in_transreg = self.in_transreg(abs(data["Muon_eta"]))
        else:
            is_in_transreg = np.array(False)
        if good_lep:
            m_id, pt_min, iso = self.config[[
                "good_muon_id", "good_muon_pt_min", "good_muon_iso"]]
        else:
            m_id, pt_min, iso = self.config[[
                "additional_muon_id", "additional_muon_pt_min",
                "additional_muon_iso"]]
        eta_min, eta_max = self.config[["muon_eta_min", "muon_eta_max"]]
        return (self.muon_id(m_id, data)
                & self.muon_iso(iso, data)
                & (~is_in_transreg)
                & (eta_min < data["Muon_eta"])
                & (data["Muon_eta"] < eta_max)
                & (pt_min < data["Muon_pt"]))

    def build_lepton_column(self, data):
        keys = ["pt", "eta", "phi", "mass", "pdgId"]
        lep_dict = {}
        ge = self.electron_cuts(data, good_lep=True)
        gm = self.muon_cuts(data, good_lep=True)
        for key in keys:
            arr = self.safe_concat(data["Electron_" + key], ge,
                                   data["Muon_" + key], gm)
            offsets = arr.offsets
            lep_dict[key] = arr.flatten()  # Could use concatenate here
        # Add supercluster eta, which only is given for electrons
        arr = self.safe_concat(data["Electron_eta"]
                               + data["Electron_deltaEtaSC"], ge,
                               data["Muon_eta"], gm)
        lep_dict["sceta"] = arr.flatten()

        leptons = Jca.candidatesfromoffsets(offsets, **lep_dict)

        # Sort leptons by pt
        leptons = leptons[leptons.pt.argsort()]
        return leptons

    def safe_concat(self, a, a_in, b, b_in):
        if len(a.flatten()) > 0 and len(b.flatten()) > 0:
            arr = awkward.concatenate([a[a_in], b[b_in]], axis=1)
        elif len(a.flatten()) > 0:
            arr = a[a_in]
        elif len(b.flatten()) > 0:
            arr = b[b_in]
        else:
            arr = awkward.JaggedArray.fromcounts([0], [])
        return arr

    def set_channels(self, data):
        leps = data["Lepton"]
        ch = np.empty(len(leps), dtype=str)
        ch[leps.counts < 2] = "None"
        twoleps = leps[leps.counts > 1]
        ee = ((np.abs(twoleps[:, 0].pdgId) == 11)
              & (np.abs(twoleps[:, 1].pdgId) == 11))
        ch[leps.counts > 1][ee] ="ee"
        mumu = ((np.abs(twoleps[:, 0].pdgId) == 13)
                & (np.abs(twoleps[:, 1].pdgId) == 13))
        ch[leps.counts > 1][mumu] = "mumu"
        emu = (np.abs(twoleps[:, 0].pdgId)
               != np.abs(twoleps[:, 1].pdgId))
        ch[leps.counts > 1][emu] = "emu"
        return ch

    def compute_lepton_sf(self, data):
        is_ele = abs(data["Lepton"].pdgId) == 11
        eles = data["Lepton"][is_ele]
        muons = data["Lepton"][~is_ele]

        weights = {}
        # Electron identification efficiency
        for i, sffunc in enumerate(self.electron_sf):
            central = sffunc(eta=eles.sceta, pt=eles.pt).prod()
            up = sffunc(eta=eles.sceta, pt=eles.pt, variation="up").prod()
            down = sffunc(eta=eles.sceta, pt=eles.pt, variation="down").prod()
            key = "electronsf{}".format(i)
            weights[key] = (central, (up / central, down / central))
        # Muon identification and isolation efficiency
        for i, sffunc in enumerate(self.muon_sf):
            central = sffunc(abseta=abs(muons.eta), pt=muons.pt).prod()
            up = sffunc(
                abseta=abs(muons.eta), pt=muons.pt, variation="up").prod()
            down = sffunc(
                abseta=abs(muons.eta), pt=muons.pt, variation="down").prod()
            key = "muonsf{}".format(i)
            weights[key] = (central, (up / central, down / central))
        return weights

    def lepton_pair(self, is_mc, data):
        accept = data["Lepton"].counts >= 2
        if self.config["compute_systematics"] and is_mc:
            return (accept, self.compute_lepton_sf(data[accept]))
        else:
            return accept

    def opposite_sign_lepton_pair(self, data):
        return (np.sign(data["Lepton"][:, 0].pdgId)
                != np.sign(data["Lepton"][:, 1].pdgId))

    def same_flavor(self, data):
        return (abs(data["Lepton"][:, 0].pdgId)
                == abs(data["Lepton"][:, 1].pdgId))

    def mll(self, data):
        return (data["Lepton"].p4[:, 0] + data["Lepton"].p4[:, 1]).mass

    def compute_jer_factor(self, data, variation="central"):
        # Coffea offers a class named JetTransformer for this. Unfortunately
        # it is more inconvinient and bugged than useful.
        counts = data["Jet_pt"].counts
        pt = awkward.JaggedArray.fromcounts(counts, data["Jet_pt"].flatten())
        eta = awkward.JaggedArray.fromcounts(counts, data["Jet_eta"].flatten())
        rho = np.repeat(data["fixedGridRhoFastjetAll"], counts)
        rho = awkward.JaggedArray.fromcounts(counts, rho)
        genidx = data["Jet_genJetIdx"]
        hasgen = ((0 <= genidx) & (genidx < data["GenJet_pt"].counts))
        genpt = pt.zeros_like()
        genpt[hasgen] = data["GenJet_pt"][genidx[hasgen]]
        jer = self._jer.getResolution(JetPt=pt, JetEta=eta, Rho=rho)
        jersmear = jer * np.random.normal(size=jer.size)
        jersf = self._jersf.getScaleFactor(JetPt=pt, JetEta=eta, Rho=rho)
        if variation == "central":
            jersf = jersf[:, :, 0]
        elif variation == "up":
            jersf = jersf[:, :, 1]
        elif variation == "down":
            jersf = jersf[:, :, 2]
        else:
            raise ValueError("variation must be one of 'central', 'up' or "
                             "'down'")
        # Hybrid method: compute stochastic smearing, apply scaling if possible
        factor = 1 + np.sqrt(np.maximum(jersf**2 - 1, 0)) * jersmear
        factor[hasgen] = (1 + (jersf - 1) * (pt - genpt) / pt)[hasgen]
        return factor

    def good_jet(self, data):
        j_id, j_puId, lep_dist, eta_min, eta_max, pt_min = self.config[[
            "good_jet_id", "good_jet_puId", "good_jet_lepton_distance",
            "good_jet_eta_min", "good_jet_eta_max", "good_jet_pt_min"]]
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
            raise utils.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_id))
        if j_puId == "skip":
            has_puId = True
        elif j_puId == "cut:loose":
            has_puId = (data["Jet_puId"] & 0b100).astype(bool)
        elif j_puId == "cut:medium":
            has_puId = (data["Jet_puId"] & 0b10).astype(bool)
        elif j_puId == "cut:tight":
            has_puId = (data["Jet_puId"] & 0b1).astype(bool)
        else:
            raise utils.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_puId))
        # Only apply PUID if pT < 50 GeV
        has_puId = has_puId | (data["Jet_pt"] >= 50)

        if "jerfac" in data:
            j_pt = data["Jet_pt"] * data["jerfac"]
        else:
            j_pt = data["Jet_pt"]

        j_eta = data["Jet_eta"]
        j_phi = data["Jet_phi"]
        l_eta = data["Lepton"].eta
        l_phi = data["Lepton"].phi
        j_eta, l_eta = j_eta.cross(l_eta, nested=True).unzip()
        j_phi, l_phi = j_phi.cross(l_phi, nested=True).unzip()
        delta_eta = j_eta - l_eta
        delta_phi = j_phi - l_phi
        delta_r = np.hypot(delta_eta, delta_phi)
        has_lepton_close = (delta_r < lep_dist).any()

        return (has_id & has_puId
                & (~has_lepton_close)
                & (eta_min < data["Jet_eta"])
                & (data["Jet_eta"] < eta_max)
                & (pt_min < j_pt))

    def build_jet_column(self, data):
        keys = ["pt", "eta", "phi", "mass", "hadronFlavour"]
        jet_dict = {}
        gj = self.good_jet(data)
        for key in keys:
            if "Jet_" + key not in data:
                continue
            if key in ("pt", "mass") and "jerfac" in data:
                # Scale pt and mass according to smearing
                arr = (data["Jet_" + key] * data["jerfac"])[gj]
            else:
                arr = data["Jet_" + key][gj]
            offsets = arr.offsets
            jet_dict[key] = arr.flatten()

        # Evaluate b-tagging
        tagger, wp = self.config["btag"].split(":")
        if tagger == "deepcsv":
            disc = data["Jet_btagDeepB"][gj]
        elif tagger == "deepjet":
            disc = data["Jet_btagDeepFlavB"][gj]
        else:
            raise utils.config.ConfigError(
                "Invalid tagger name: {}".format(tagger))
        year = self.config["year"]
        wptuple = utils.btagging.BTAG_WP_CUTS[tagger][year]
        if not hasattr(wptuple, wp):
            raise utils.config.ConfigError(
                "Invalid working point \"{}\" for {} in year {}".format(
                    wp, tagger, year))
        jet_dict["btag"] = disc.flatten()
        jet_dict["btagged"] = (disc > getattr(wptuple, wp)).flatten()

        jets = Jca.candidatesfromoffsets(offsets, **jet_dict)

        # Sort jets by pt
        jets = jets[jets.pt.argsort()]
        return jets

    def build_met_column(self, data):
        met = TLorentzVectorArray.from_ptetaphim(data["MET_pt"],
                                                 np.zeros(data.size),
                                                 data["MET_pt"],
                                                 np.zeros(data.size))
        if "jerfac" in data:
            jets = TLorentzVectorArray.from_ptetaphim(data["Jet_pt"],
                                                      data["Jet_eta"],
                                                      data["Jet_phi"],
                                                      data["Jet_mass"])
            newx = met.x - (jets.x * (data["jerfac"] - 1)).sum()
            newy = met.y - (jets.y * (data["jerfac"] - 1)).sum()
            met = TLorentzVectorArray.from_ptetaphim(np.hypot(newx, newy),
                                                     np.zeros(data.size),
                                                     np.arctan2(newy, newx),
                                                     np.zeros(data.size))
        return Jca.candidatesfromoffsets(np.arange(data.size + 1),
                                         pt=met.pt,
                                         eta=met.eta,
                                         phi=met.phi,
                                         mass=met.mass)

    def in_hem1516(self, phi, eta):
        return ((-3.0 < eta) & (eta < -1.3) & (-1.57 < phi) & (phi < -0.87))

    def hem_cut(self, data):
        cut_ele = self.config["hem_cut_if_ele"]
        cut_muon = self.config["hem_cut_if_muon"]
        cut_jet = self.config["hem_cut_if_jet"]

        keep = np.full(data.size, True)
        if cut_ele:
            ele = data["Lepton"][abs(data["Lepton"].pdgId) == 11]
            keep &= ~self.in_hem1516(ele.phi, ele.eta).any()
        if cut_muon:
            muon = data["Lepton"][abs(data["Lepton"].pdgId) == 13]
            keep &= ~self.in_hem1516(muon.phi, muon.eta).any()
        if cut_jet:
            jet = data["Jet"]
            keep &= ~self.in_hem1516(jet.phi, jet.eta).any()
        return keep

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
        return data["mll"] > self.config["mll_min"]

    def no_additional_leptons(self, is_mc, data):
        add_ele = self.electron_cuts(data, good_lep=False)
        add_muon = self.muon_cuts(data, good_lep=False)
        accept = add_ele.sum() + add_muon.sum() <= 2
        if self.config["compute_systematics"] and is_mc:
            return (accept, self.compute_lepton_sf(data[accept]))
        else:
            return accept

    def z_window(self, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        is_sf = data["is_same_flavor"]
        return ~is_sf | ((data["mll"] <= m_min) | (m_max <= data["mll"]))

    def has_jets(self, data):
        return self.config["num_jets_atleast"] <= data["Jet"].counts

    def jet_pt_requirement(self, data):
        n = np.zeros(data.size)
        for i, pt_min in enumerate(self.config["jet_pt_min"]):
            mask = data["Jet"].counts > i
            n[mask] += (pt_min < data["Jet"].pt[mask, i]).astype(int)
        return n >= self.config["jet_pt_num_satisfied"]

    def btag_cut(self, data):
        num_btagged = data["Jet"]["btagged"].sum()
        return num_btagged >= self.config["num_atleast_btagged"]

    def met_requirement(self, data):
        is_sf = data["is_same_flavor"]
        met = data["MET_pt"]
        return ~is_sf | (met > self.config["ee/mm_min_met"])

    def compute_weight_btag(self, data, variation="central"):
        if self.config["num_atleast_btagged"] == 0:
            return np.full(data.size, 1.)
        jets = data["Jet"]
        wp = self.config["btag"].split(":", 1)[1]
        flav = jets["hadronFlavour"]
        eta = jets.eta
        pt = jets.pt
        discr = jets["btag"]
        return np.prod(list(weighter(wp, flav, eta, pt, discr, variation)
                       for weighter in self.btagweighters), axis=0)

    def pick_leps(self, data):
        lep_pair = data["Lepton"][:, :2]
        lep = lep_pair[lep_pair.pdgId > 0]
        antilep = lep_pair[lep_pair.pdgId < 0]
        return lep, antilep

    def choose_bs(self, data, lep, antilep):
        btags = data["Jet"][data["Jet"].btagged]
        jetsnob = data["Jet"][~data["Jet"].btagged]
        b0, b1 = utils.misc.pairswhere(btags.counts > 1,
                                       btags.distincts(),
                                       btags.cross(jetsnob))
        bs = utils.misc.concatenate(b0, b1)
        bbars = utils.misc.concatenate(b1, b0)
        alb = bs.cross(antilep)
        lbbar = bbars.cross(lep)
        hist_mlb = uproot.open(self.hist_mlb_path)["Mlb"]
        p_m_alb = awkward.JaggedArray.fromcounts(
            bs.counts, hist_mlb.allvalues[np.searchsorted(
                hist_mlb.alledges, alb.mass.content)-1])
        p_m_lbbar = awkward.JaggedArray.fromcounts(
            bs.counts, hist_mlb.allvalues[np.searchsorted(
                hist_mlb.alledges, lbbar.mass.content)-1])
        bestbpair_mlb = (p_m_alb*p_m_lbbar).argmax()
        return bs[bestbpair_mlb], bbars[bestbpair_mlb]

    def passing_reco(self, data):
        return data["neutrino"].counts > 0

    def wminus(self, data):
        return data["antineutrino"].cross(data["lep"])

    def wplus(self, data):
        return data["neutrino"].cross(data["antilep"])

    def top(self, data):
        return data["Wplus"].cross(data["b"])

    def antitop(self, data):
        return data["Wminus"].cross(data["bbar"])

    def ttbar(slef, data):
        return data["top"].p4 + data["antitop"].p4
