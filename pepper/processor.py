import os
import sys
from functools import partial
from collections import namedtuple
import numpy as np
import awkward
import coffea
import coffea.lumi_tools
from coffea.analysis_objects import JaggedCandidateArray as Jca
import coffea.processor as processor
import uproot
from uproot_methods import TLorentzVectorArray
import h5py
import logging
from copy import copy

import pepper
from pepper import sonnenschein, betchart, Selector, LazyTable, OutputFiller
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


class Processor(processor.ProcessorABC):
    def __init__(self, config, destdir):
        """Create a new Processor

        Arguments:
        config -- A Config instance, defining the configuration to use
        destdir -- Destination directory, where the event HDF5s are saved.
                   Every chunk will be saved in its own file. If `None`,
                   nothing will be saved.
        """
        self._check_config_integrity(config)
        self.config = config
        if destdir is not None:
            self.destdir = os.path.realpath(destdir)
        else:
            self.destdir = None
        self.hists = self._get_hists_from_config(
            self.config, "hists", "hists_to_do")

        if "top_pt_reweighting" in self.config:
            self.topptweighter = self.config["top_pt_reweighting"]
        else:
            self.topptweighter = None

        if "lumimask" in config:
            self.lumimask = self.config["lumimask"]
        else:
            logger.warning("No lumimask specified")
            self.lumimask = None
        if ("electron_sf" in self.config
                and len(self.config["electron_sf"]) > 0):
            self.electron_sf = self.config["electron_sf"]
        else:
            logger.warning("No electron scale factors specified")
            self.electron_sf = []
        if "muon_sf" in self.config and len(self.config["muon_sf"]) > 0:
            self.muon_sf = self.config["muon_sf"]
        else:
            logger.warning("No muon scale factors specified")
            self.muon_sf = []
        if "btag_sf" in self.config and len(self.config["btag_sf"]) > 0:
            self.btagweighters = config["btag_sf"]
        else:
            logger.warning("No btag scale factor specified")
            self.btagweighters = None

        if "jet_uncertainty" in self.config:
            self._junc = self.config["jet_uncertainty"]
        else:
            if config["compute_systematics"]:
                logger.warning("No jet uncertainty specified")
            self._junc = None

        if "jet_resolution" in self.config and "jet_ressf" in self.config:
            self._jer = self.config["jet_resolution"]
            self._jersf = self.config["jet_ressf"]
        else:
            logger.warning("No jet resolution or no jet resolution scale "
                           "factor specified. This is necessary for "
                           "smearing, even if not computing systematics")
            self._jer = None
            self._jersf = None

        self.trigger_paths = config["dataset_trigger_map"]
        self.trigger_order = config["dataset_trigger_order"]
        if "reco_info_file" in self.config:
            self.reco_info_filepath = self.config["reco_info_file"]
        elif self.config["reco_algorithm"] is not None:
            raise pepper.config.ConfigError(
                "Need reco_info_file for kinematic reconstruction")
        self.mc_lumifactors = config["mc_lumifactors"]

        # Construct a dict of datasets for which we are not processing the
        # relevant dedicated systematic datasets, and so the nominal histograms
        # need to be copied
        self.copy_nominal = {}
        for sysdataset, syst in self.config["dataset_for_systematics"].items():
            replaced, sysname = syst
            if sysname not in self.copy_nominal:
                self.copy_nominal[sysname] = []
                # Copy all normal mc datasets
                for dataset in self.config["mc_datasets"].keys():
                    if dataset in self.config["dataset_for_systematics"]:
                        continue
                    self.copy_nominal[sysname].append(dataset)
            try:
                # Remove the ones that get replaced by a dedicated dataset
                self.copy_nominal[sysname].remove(replaced)
            except ValueError:
                pass

        if "randomised_parameter_scan_datasets" in config:
            self.rps_datasets = config["randomised_parameter_scan_datasets"]
        else:
            self.rps_datasets = []

    @staticmethod
    def _check_config_integrity(config):
        inv_datasets_for_systematics = {}
        dataset_for_systematics = config["dataset_for_systematics"]
        for sysds, (replaceds, variation) in dataset_for_systematics.items():
            if sysds not in config["mc_datasets"]:
                raise pepper.config.ConfigError(
                    "Got systematic dataset that is not mentioned in "
                    f"mc_datasets: {sysds}")
            if replaceds not in config["mc_datasets"]:
                raise pepper.config.ConfigError(
                    "Got dataset to be replaced by a systematic dataset and "
                    f"that is not mentioned in mc_datasets: {replaceds}")
            if (replaceds, variation) in inv_datasets_for_systematics:
                prevds = inv_datasets_for_systematics[(replaceds, variation)]
                raise pepper.config.ConfigError(
                    f"{replaceds} already being replaced for {variation} by "
                    f"{prevds} but is being repeated with {sysds}")
            inv_datasets_for_systematics[(replaceds, variation)] = sys

        if "crosssection_uncertainty" in config:
            xsuncerts = config["crosssection_uncertainty"]
            for dsname in xsuncerts.keys():
                if dsname not in config["mc_datasets"]:
                    raise pepper.config.ConfigError(
                        f"{dsname} in crosssection_uncertainty but not in "
                        "mc_datasets")
            for dsname in config["mc_datasets"].keys():
                if dsname in dataset_for_systematics:
                    continue
                if dsname not in xsuncerts:
                    raise pepper.config.ConfigError(
                        f"{dsname} in mc_datasets but not in "
                        "crosssection_uncertainty")

        # TODO: Check other config variables if necessary

    @staticmethod
    def _get_hists_from_config(config, key, todokey):
        if key in config:
            hists = config[key]
        else:
            hists = {}
        if todokey in config and len(config[todokey]) > 0:
            new_hists = {}
            for name in config[todokey]:
                if name in hists:
                    new_hists[name] = hists[name]
            hists = new_hists
            logger.info("Doing only the histograms: " +
                        ", ".join(hists.keys()))
        return hists

    @property
    def accumulator(self):
        self._accumulator = processor.dict_accumulator({
            "hists": processor.dict_accumulator(),
            "cutflows": processor.dict_accumulator(),
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
        logger.debug(f"Opened output {filepath}")
        return f

    def _prepare_saved_columns(self, selector):
        columns = {}
        for specifier in self.config["columns_to_save"]:
            item = selector.masked
            if isinstance(specifier, str):
                specifier = [specifier]
            elif not isinstance(specifier, list):
                raise pepper.config.ConfigError(
                    "columns_to_save must be str or list")
            for subspecifier in specifier:
                try:
                    item = item[subspecifier]
                except KeyError:
                    try:
                        item = getattr(item, subspecifier)
                    except AttributeError:
                        logger.info("Skipping to save column because it is "
                                    f"not present: {specifier}")
                        break
            else:
                if isinstance(item, awkward.JaggedArray):
                    # Strip rows that are not selected from memory
                    item = item.deepcopy()
                    if isinstance(item.content, awkward.Table):
                        # Remove columns used for caching by Coffea
                        for column in item.content.columns:
                            if column.startswith("__"):
                                del item.content[column]
                key = "_".join(specifier)
                if key in columns:
                    raise pepper.config.ConfigError(
                        f"Ambiguous column to save '{key}', (from "
                        f"{specifier})")
                columns[key] = item
        return awkward.Table(columns)

    def _save_per_event_info(self, dsname, selector, identifier):
        with self._open_output(dsname) as f:
            outf = awkward.hdf5(f)
            out_dict = {"dsname": dsname, "identifier": identifier}
            out_dict["events"] = self._prepare_saved_columns(selector)
            out_dict["weight"] = selector.weight
            cutnames, cutflags = selector.get_cuts()
            out_dict["cutnames"] = cutnames
            out_dict["cutflags"] = cutflags

            for key in out_dict.keys():
                outf[key] = out_dict[key]

    def process(self, df):
        dsname = df["dataset"]
        filename = df["filename"]
        entrystart = df._branchargs["entrystart"]
        entrystop = df._branchargs["entrystop"]
        logger.debug(f"Started processing {filename} from event "
                     f"{entrystart} to {entrystop - 1} for dataset {dsname}")
        if dsname in self.rps_datasets:
            return self.process_rps(
                df, dsname, filename, entrystart, entrystop)
        data = LazyTable(df)
        is_mc = (dsname in self.config["mc_datasets"].keys())

        filler = self.setup_outputfiller(data, dsname, is_mc)
        selector = self.setup_selection(data, dsname, is_mc, filler)
        self.process_selection(selector, dsname, is_mc, filler)

        if self.destdir is not None:
            logger.debug("Saving per event info")
            self._save_per_event_info(
                dsname, selector, (filename, entrystart, entrystop))

        logger.debug("Processing finished")
        return filler.output

    def process_rps(self, df, dsname, filename, entrystart, entrystop):
        logger.debug("This is a randomised parameter signal sample- processing"
                     " each scan point separately")
        data = LazyTable(df)
        is_mc = (dsname in self.config["mc_datasets"].keys())
        filler = self.setup_outputfiller(data, dsname, is_mc)
        scanpoints = [key for key in data.columns
                      if key.startswith("GenModel_")]
        for sp in scanpoints:
            logger.debug(f"Processing scan point {sp}")
            dsname = sp.split("_", 1)[1]
            filler.update_ds(dsname, dsname, None)
            selector = self.setup_selection(copy(data), dsname, is_mc, filler)
            selector.add_cut(partial(self.pick_scan_point, sp),
                             "Select scan point", no_callback=True)
            self.process_selection(selector, dsname, is_mc, filler)
            if self.destdir is not None:
                logger.debug("Saving per event info")
                self._save_per_event_info(
                    dsname, selector, (filename, entrystart, entrystop))

        logger.debug("Processing finished")
        return filler.output

    def pick_scan_point(self, sp, data):
        return data[sp]

    def setup_outputfiller(self, data, dsname, is_mc):
        output = self.accumulator.identity()
        sys_enabled = self.config["compute_systematics"]

        if dsname in self.config["dataset_for_systematics"]:
            dsname_in_hist = self.config["dataset_for_systematics"][dsname][0]
            sys_overwrite = self.config["dataset_for_systematics"][dsname][1]
        else:
            dsname_in_hist = dsname
            sys_overwrite = None

        filler = OutputFiller(
            output, self.hists, is_mc, dsname, dsname_in_hist, sys_enabled,
            sys_overwrite=sys_overwrite, copy_nominal=self.copy_nominal)

        return filler

    def setup_selection(self, data, dsname, is_mc, filler):
        if is_mc:
            genweight = data["genWeight"]
        else:
            genweight = None
        selector = Selector(data, genweight, filler.get_callbacks())
        return selector

    def process_selection(self, selector, dsname, is_mc, filler):
        if dsname.startswith("TTTo"):
            selector.set_column(self.gentop, "gent_lc")
            if self.topptweighter is not None:
                self.do_top_pt_reweighting(selector)
        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)
        if is_mc:
            self.add_crosssection_scale(selector, dsname)

        if self.config["blinding_denom"] is not None:
            selector.add_cut(partial(self.blinding, is_mc), "Blinding")
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

        if (is_mc and self.config["compute_systematics"]
                and dsname not in self.config["dataset_for_systematics"]):
            assert filler.sys_overwrite is None
            for variarg in self.get_jetmet_variation_args():
                selector_copy = selector.copy()
                filler.sys_overwrite = variarg.name
                self.process_selection_jet_part(selector_copy, is_mc, variarg)
            filler.sys_overwrite = None

        # Do normal, no-variation run
        self.process_selection_jet_part(
            selector, is_mc, self.get_jetmet_nominal_arg())
        logger.debug("Selection done")

    def get_jetmet_variation_args(self):
        ret = []
        ret.append(VariationArg("UncMET_up", met="up"))
        ret.append(VariationArg("UncMET_down", met="down"))
        if self._junc is not None:
            for source in self._junc.levels:
                if source == "jes":
                    name = "Junc_"
                else:
                    name = f"Junc{source}_"
                ret.append(VariationArg(name + "up", junc=("up", source)))
                ret.append(VariationArg(name + "down", junc=("down", source)))
        if self._jer is not None and self._jersf is not None:
            ret.append(VariationArg("Jer_up", jer="up"))
            ret.append(VariationArg("Jer_down", jer="down"))
        return ret

    def get_jetmet_nominal_arg(self):
        if self._jer is not None and self._jersf is not None:
            return VariationArg(None)
        else:
            return VariationArg(None, jer=None)

    def process_selection_jet_part(self, selector, is_mc, variation):
        logger.debug(f"Running jet_part with variation {variation.name}")
        if is_mc:
            selector.set_column(partial(
                self.compute_jet_factor, variation.junc, variation.jer),
                "jetfac")
        selector.set_column(self.build_jet_column, "Jet")
        selector.set_column(partial(self.build_met_column,
                                    variation=variation.met), "MET")
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

        reco_alg = self.config["reco_algorithm"]
        if reco_alg is not None:
            selector.set_column(self.pick_leps, "recolepton", all_cuts=True)
            selector.set_column(self.pick_bs, "recob", all_cuts=True)
            selector.set_column(
                partial(self.ttbar_system, reco_alg.lower()),
                "recot", all_cuts=True, no_callback=True)
            selector.add_cut(self.passing_reco, "Reco")
            selector.set_column(self.build_nu_column, "reconu", all_cuts=True)
            selector.set_column(self.calculate_dark_pt, "dark_pt",
                                all_cuts=True)

    def gentop(self, data):
        part = pepper.misc.jcafromjagged(
            pt=data["GenPart_pt"],
            eta=data["GenPart_eta"],
            phi=data["GenPart_phi"],
            mass=data["GenPart_mass"],
            pdgId=data["GenPart_pdgId"],
            statusflags=data["GenPart_statusFlags"]
        )
        motheridx = data["GenPart_genPartIdxMother"]
        hasmother = ((0 <= motheridx) & (motheridx < part.counts))
        part = part[hasmother]
        is_last_copy = part.statusflags >> 13 & 1 == 1
        part = part[is_last_copy]
        abspdg = abs(part.pdgId)
        return pepper.misc.sortby(part[abspdg == 6], "pdgId")

    def do_top_pt_reweighting(self, selector):
        pt = selector.masked["gent_lc"].pt
        sf = self.topptweighter(pt[:, 0], pt[:, 1])
        selector.modify_weight("Top pt reweighting", sf)

    def add_generator_uncertainies(self, dsname, selector):
        # Matrix-element renormalization and factorization scale
        # Get describtion of individual columns of this branch with
        # Events->GetBranch("LHEScaleWeight")->GetTitle() in ROOT
        data = selector.masked
        if "LHEScaleWeight" in data:
            norm = self.config["mc_lumifactors"][dsname + "_LHEScaleSumw"]
            selector.set_systematic("MEren",
                                    data["LHEScaleWeight"][:, 7] * norm[7],
                                    data["LHEScaleWeight"][:, 1] * norm[1])
            selector.set_systematic("MEfac",
                                    data["LHEScaleWeight"][:, 5] * norm[5],
                                    data["LHEScaleWeight"][:, 3] * norm[3])
        else:
            selector.set_systematic(
                "MEren", np.ones(data.size), np.ones(data.size))
            selector.set_systematic(
                "MEfac", np.ones(data.size), np.ones(data.size))

        # Parton shower scale
        psweight = data["PSWeight"]
        if psweight.counts[0] != 1:
            # NanoAOD containts one 1.0 per event in PSWeight if there are no
            # PS weights available, otherwise all counts > 1.
            selector.set_systematic("PSisr", psweight[:, 2], psweight[:, 0])
            selector.set_systematic("PSfsr", psweight[:, 3], psweight[:, 1])
        else:
            selector.set_systematic(
                "PSisr", np.ones(data.size), np.ones(data.size))
            selector.set_systematic(
                "PSfsr", np.ones(data.size), np.ones(data.size))

    def add_crosssection_scale(self, selector, dsname):
        num_events = selector.num_selected
        lumifactors = self.mc_lumifactors
        factor = np.full(num_events, lumifactors[dsname])
        selector.modify_weight("lumi_factor", factor)
        if (self.config["compute_systematics"]
                and dsname not in self.config["dataset_for_systematics"]):
            xsuncerts = self.config["crosssection_uncertainty"]
            for group in set(v[0] for v in xsuncerts.values()):
                if xsuncerts[dsname] is None or group != xsuncerts[dsname][0]:
                    uncert = 0
                else:
                    uncert = xsuncerts[dsname][1]
                selector.set_systematic(group + "XS",
                                        np.full(num_events, 1 + uncert),
                                        np.full(num_events, 1 - uncert))

    def blinding(self, is_mc, data):
        if not is_mc:
            return np.mod(data["event"], self.config["blinding_denom"]) == 0
        else:
            return (np.full(data.size, True),
                    {"Blinding_sf":
                     np.full(data.size, 1/self.config["blinding_denom"])})

    def good_lumimask(self, is_mc, data):
        if is_mc:
            # Lumimask only present in data, all events pass in MC
            # Compute lumi variation here
            allpass = np.full(data.size, True)
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
        return (1.444 < abs_eta) & (abs_eta < 1.566)

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
            arr = awkward.concatenate([data["Electron_" + key][ge],
                                      data["Muon_" + key][gm]], axis=1)
            offsets = arr.offsets
            lep_dict[key] = arr.flatten()  # Could use concatenate here
        # Add supercluster eta, which only is given for electrons
        arr = awkward.concatenate([data["Electron_eta"][ge]
                                  + data["Electron_deltaEtaSC"][ge],
                                  data["Muon_eta"][gm]], axis=1)
        lep_dict["sceta"] = arr.flatten()

        leptons = Jca.candidatesfromoffsets(offsets, **lep_dict)

        # Sort leptons by pt
        leptons = leptons[leptons.pt.argsort()]
        return leptons

    def channel_masks(self, data):
        leps = data["Lepton"]
        firstpdg = abs(leps[:, 0].pdgId)
        secondpdg = abs(leps[:, 1].pdgId)
        channels = {}
        channels["is_ee"] = (firstpdg == 11) & (secondpdg == 11)
        channels["is_mm"] = (firstpdg == 13) & (secondpdg == 13)
        channels["is_em"] = (~channels["is_ee"]) & (~channels["is_mm"])
        return channels

    def compute_lepton_sf(self, data):
        is_ele = abs(data["Lepton"].pdgId) == 11
        eles = data["Lepton"][is_ele]
        muons = data["Lepton"][~is_ele]

        weights = {}
        # Electron identification efficiency
        for i, sffunc in enumerate(self.electron_sf):
            central = sffunc(eta=eles.sceta, pt=eles.pt).prod()
            key = "electronsf{}".format(i)
            if self.config["compute_systematics"]:
                up = sffunc(
                    eta=eles.sceta, pt=eles.pt, variation="up").prod()
                down = sffunc(
                    eta=eles.sceta, pt=eles.pt, variation="down").prod()
                weights[key] = (central, up / central, down / central)
            else:
                weights[key] = central
        # Muon identification and isolation efficiency
        for i, sffunc in enumerate(self.muon_sf):
            central = sffunc(abseta=abs(muons.eta), pt=muons.pt).prod()
            key = "muonsf{}".format(i)
            if self.config["compute_systematics"]:
                up = sffunc(
                    abseta=abs(muons.eta), pt=muons.pt, variation="up").prod()
                down = sffunc(
                    abseta=abs(muons.eta), pt=muons.pt,
                    variation="down").prod()
                weights[key] = (central, up / central, down / central)
            else:
                weights[key] = central
        return weights

    def lepton_pair(self, is_mc, data):
        accept = data["Lepton"].counts >= 2
        if is_mc:
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

    def dilep_pt(self, data):
        return (data["Lepton"].p4[:, 0] + data["Lepton"].p4[:, 1]).pt

    def compute_junc_factor(self, data, variation, source="jes"):
        if variation not in ("up", "down"):
            raise ValueError("variation must be either 'up' or 'down'")
        if source not in self._junc.levels:
            raise ValueError(f"Jet uncertainty not found: {source}")
        counts = data["Jet_pt"].counts
        if counts.size == 0 or counts.sum() == 0:
            return awkward.JaggedArray.fromcounts(counts, [])
        # Make sure pt and eta aren't offset arrays.
        # Needed by JetCorrectionUncertainty
        pt = awkward.JaggedArray.fromcounts(counts, data["Jet_pt"].flatten())
        eta = awkward.JaggedArray.fromcounts(counts, data["Jet_eta"].flatten())
        junc = dict(self._junc.getUncertainty(JetPt=pt, JetEta=eta))[source]
        if variation == "up":
            return junc[:, :, 0]
        else:
            return junc[:, :, 1]

    def compute_jer_factor(self, data, variation="central"):
        # Coffea offers a class named JetTransformer for this. Unfortunately
        # it is more inconvinient and bugged than useful.
        counts = data["Jet_pt"].counts
        if counts.size == 0 or counts.sum() == 0:
            return awkward.JaggedArray.fromcounts(counts, [])
        # Make sure pt and eta aren't offset arrays. Needed by JetResolution
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

    def compute_jet_factor(self, junc, jer, data):
        factor = data["Jet_pt"].ones_like()
        if junc is not None:
            factor = factor * self.compute_junc_factor(data, *junc)
        if jer is not None:
            factor = factor * self.compute_jer_factor(data, jer)
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
            raise pepper.config.ConfigError(
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
            raise pepper.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_puId))
        # Only apply PUID if pT < 50 GeV
        has_puId = has_puId | (data["Jet_pt"] >= 50)

        j_pt = data["Jet_pt"]
        if "jetfac" in data:
            j_pt = j_pt * data["jetfac"]

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
            arr = data["Jet_" + key]
            if key in ("pt", "mass") and "jetfac" in data:
                arr = arr * data["jetfac"]
            arr = arr[gj]
            offsets = arr.offsets
            jet_dict[key] = arr.flatten()

        # Evaluate b-tagging
        tagger, wp = self.config["btag"].split(":")
        if tagger == "deepcsv":
            disc = data["Jet_btagDeepB"][gj]
        elif tagger == "deepjet":
            disc = data["Jet_btagDeepFlavB"][gj]
        else:
            raise pepper.config.ConfigError(
                "Invalid tagger name: {}".format(tagger))
        year = self.config["year"]
        wptuple = pepper.btagging.BTAG_WP_CUTS[tagger][year]
        if not hasattr(wptuple, wp):
            raise pepper.config.ConfigError(
                "Invalid working point \"{}\" for {} in year {}".format(
                    wp, tagger, year))
        jet_dict["btag"] = disc.flatten()
        jet_dict["btagged"] = (disc > getattr(wptuple, wp)).flatten()

        jets = Jca.candidatesfromoffsets(offsets, **jet_dict)

        # Sort jets by pt
        jets = jets[jets.pt.argsort()]
        return jets

    def build_met_column(self, data, variation="central"):
        met = TLorentzVectorArray.from_ptetaphim(data["MET_pt"],
                                                 np.zeros(data.size),
                                                 data["MET_phi"],
                                                 np.zeros(data.size))
        if variation == "up":
            dx = data["MET_MetUnclustEnUpDeltaX"]
            dy = data["MET_MetUnclustEnUpDeltaY"]
            met = TLorentzVectorArray.from_cartesian(
                met.x + dx, met.y + dy, met.z, met.t)
        elif variation == "down":
            dx = data["MET_MetUnclustEnUpDeltaX"]
            dy = data["MET_MetUnclustEnUpDeltaY"]
            met = TLorentzVectorArray.from_cartesian(
                met.x - dx, met.y - dy, met.z, met.t)
        elif variation != "central":
            raise ValueError(
                "variation must be one of 'central', 'up' or 'down'")
        if "jetfac" in data:
            jets = TLorentzVectorArray.from_ptetaphim(data["Jet_pt"],
                                                      data["Jet_eta"],
                                                      data["Jet_phi"],
                                                      data["Jet_mass"])
            factor = data["jetfac"] - 1
            newx = met.x - (jets.x * factor).sum()
            newy = met.y - (jets.y * factor).sum()
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
        if is_mc:
            return (accept, self.compute_lepton_sf(data[accept]))
        else:
            return accept

    def z_window(self, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        is_out_window = (data["mll"] <= m_min) | (m_max <= data["mll"])
        return data["is_em"] | is_out_window

    def has_jets(self, data):
        return self.config["num_jets_atleast"] <= data["Jet"].counts

    def jet_pt_requirement(self, data):
        n = np.zeros(data.size)
        for i, pt_min in enumerate(self.config["jet_pt_min"]):
            mask = data["Jet"].counts > i
            n[mask] += (pt_min < data["Jet"].pt[mask, i]).astype(int)
        return n >= self.config["jet_pt_num_satisfied"]

    def compute_weight_btag(self, data):
        jets = data["Jet"]
        wp = self.config["btag"].split(":", 1)[1]
        flav = jets["hadronFlavour"]
        eta = jets.eta
        pt = jets.pt
        discr = jets["btag"]
        weight = {}
        for i, weighter in enumerate(self.btagweighters):
            central = weighter(wp, flav, eta, pt, discr, "central")
            if self.config["compute_systematics"]:
                up = weighter(wp, flav, eta, pt, discr, "up")
                down = weighter(wp, flav, eta, pt, discr, "down")
                weight[f"btagsf{i}"] = (central, up / central, down / central)
            else:
                weight[f"btagsf{i}"] = central
        return weight

    def btag_cut(self, is_mc, data):
        num_btagged = data["Jet"]["btagged"].sum()
        is_tagged = num_btagged >= self.config["num_atleast_btagged"]
        if is_mc and self.btagweighters is not None:
            return (is_tagged, self.compute_weight_btag(data[is_tagged]))
        else:
            return is_tagged

    def met_requirement(self, data):
        met = data["MET"].pt
        return data["is_em"] | (met > self.config["ee/mm_min_met"])

    def pick_leps(self, data):
        return pepper.misc.sortby(data["Lepton"], "pdgId")

    def pick_bs(self, data):
        recolepton = data["recolepton"]
        lep = recolepton.fromjagged(awkward.JaggedArray.fromcounts(
            np.full(data.size, 1), recolepton[:, 0]))
        antilep = recolepton.fromjagged(awkward.JaggedArray.fromcounts(
            np.full(data.size, 1), recolepton[:, 1]))
        btags = data["Jet"][data["Jet"].btagged]
        jetsnob = data["Jet"][~data["Jet"].btagged]
        b0, b1 = pepper.misc.pairswhere(btags.counts > 1,
                                        btags.distincts(),
                                        btags.cross(jetsnob))
        bs = pepper.misc.concatenate(b0, b1)
        bbars = pepper.misc.concatenate(b1, b0)
        alb = bs.cross(antilep)
        lbbar = bbars.cross(lep)
        hist_mlb = uproot.open(self.reco_info_filepath)["mlb"]
        p_m_alb = awkward.JaggedArray.fromcounts(
            bs.counts, hist_mlb.allvalues[np.searchsorted(
                hist_mlb.alledges, alb.mass.content)-1])
        p_m_lbbar = awkward.JaggedArray.fromcounts(
            bs.counts, hist_mlb.allvalues[np.searchsorted(
                hist_mlb.alledges, lbbar.mass.content)-1])
        bestbpair_mlb = (p_m_alb*p_m_lbbar).argmax()
        return awkward.concatenate([bs[bestbpair_mlb], bbars[bestbpair_mlb]],
                                   axis=1)

    def ttbar_system(self, reco_alg, data):
        lep = data["recolepton"].p4[:, 0]
        antilep = data["recolepton"].p4[:, 1]
        b = data["recob"].p4[:, 0]
        antib = data["recob"].p4[:, 1]
        met = data["MET"].p4.flatten()

        with uproot.open(self.reco_info_filepath) as f:
            if self.config["reco_num_smear"] is None:
                energyfl = energyfj = 1
                alphal = alphaj = 0
                num_smear = 1
                mlb = None
            else:
                energyfl = f["energyfl"]
                energyfj = f["energyfj"]
                alphal = f["alphal"]
                alphaj = f["alphaj"]
                mlb = f["mlb"]
                num_smear = self.config["reco_num_smear"]
            if isinstance(self.config["reco_w_mass"], (int, float)):
                mw = self.config["reco_w_mass"]
            else:
                mw = f[self.config["reco_w_mass"]]
            if isinstance(self.config["reco_t_mass"], (int, float)):
                mt = self.config["reco_t_mass"]
            else:
                mt = f[self.config["reco_t_mass"]]
        if reco_alg == "sonnenschein":
            top, antitop = sonnenschein(
                lep, antilep, b, antib, met, mwp=mw, mwm=mw, mt=mt, mat=mt,
                energyfl=energyfl, energyfj=energyfj, alphal=alphal,
                alphaj=alphaj, hist_mlb=mlb, num_smear=num_smear)
            top = awkward.concatenate([top, antitop], axis=1)
            return Jca.candidatesfromcounts(top.counts, p4=top.flatten())
        elif reco_alg == "betchart":
            top, antitop = betchart(
                lep, antilep, b, antib, met, MW=mw, Mt=mt)
            return awkward.concatenate([top, antitop], axis=1)
        else:
            raise ValueError(f"Invalid value for reco algorithm: {reco_alg}")

    def passing_reco(self, data):
        return data["recot"].counts > 0

    def build_nu_column(self, data):
        lep = data["recolepton"].p4[:, 0]
        antilep = data["recolepton"].p4[:, 1]
        b = data["recob"].p4[:, 0]
        antib = data["recob"].p4[:, 1]
        top = data["recot"].p4[:, 0]
        antitop = data["recot"].p4[:, 1]
        nu = Jca.candidatesfromcounts(np.ones(data.size), p4=top - b - antilep)
        antinu = Jca.candidatesfromcounts(np.ones(data.size),
                                          p4=antitop - antib - lep)
        return awkward.concatenate([nu, antinu], axis=1)

    def calculate_dark_pt(self, data):
        nu = data["reconu"].p4[:, 0]
        antinu = data["reconu"].p4[:, 1]
        met = data["MET"].p4.flatten()
        return Jca.candidatesfromcounts(np.ones(data.size),
                                        p4=met - nu - antinu)
