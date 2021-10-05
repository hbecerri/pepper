import sys
from functools import partial, reduce
from collections import namedtuple
import numpy as np
import awkward as ak
import coffea
import coffea.lumi_tools
import coffea.jetmet_tools
import uproot
import logging
from copy import copy

import pepper
from pepper import sonnenschein, betchart
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


class Processor(pepper.Processor):
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

        if "top_pt_reweighting" in self.config:
            self.topptweighter = self.config["top_pt_reweighting"]
        else:
            self.topptweighter = None

        if "lumimask" in config:
            self.lumimask = self.config["lumimask"]
        else:
            logger.warning("No lumimask specified")
            self.lumimask = None
        if "pileup_reweighting" in self.config:
            self.puweighter = self.config["pileup_reweighting"]
        else:
            logger.warning("No pileup reweigthing specified")
            self.puweighter = None
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
            self.btagweighters = []

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
        if ((self._jer is not None or self._junc is not None)
                and "jet_correction" not in self.config):
            raise pepper.config.ConfigError(
                "Need jet_correction for propagating jet corrections to MET")
        else:
            self._jec = self.config["jet_correction"]

        self.trigger_paths = config["dataset_trigger_map"]
        self.trigger_order = config["dataset_trigger_order"]
        if "reco_info_file" in self.config:
            self.reco_info_filepath = self.config["reco_info_file"]
        elif "reco_algorithm" in self.config:
            raise pepper.config.ConfigError(
                "Need reco_info_file for kinematic reconstruction")
        self.mc_lumifactors = config["mc_lumifactors"]
        if "drellyan_sf" in self.config:
            self.drellyan_sf = self.config["drellyan_sf"]
        else:
            logger.warning("No Drell-Yan scale factor specified")
            self.drellyan_sf = None
        if "MET_xy_shifts" in self.config:
            self.met_xy_shifts = self.config["MET_xy_shifts"]
        else:
            self.met_xy_shifts = None
        if "trigger_sfs" in self.config:
            self.trigger_sfs = self.config["trigger_sfs"]
        else:
            logger.warning("No trigger scale factors specified")
            self.trigger_sfs = None
        if "pdf_types" in config:
            self.pdf_types = config["pdf_types"]
        else:
            if config["compute_systematics"]:
                logger.warning(
                    "pdf_type not specified; will not compute pdf "
                    "uncertainties. (Options are 'Hessian', 'MC' and "
                    "'MC_Gaussian')")
            self.pdf_types = None

    def _check_config_integrity(self, config):
        super()._check_config_integrity(config)

        # Skip if no systematics, as currently only checking syst configs
        if not config["compute_systematics"]:
            return
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

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)
        if dsname.startswith("TTTo"):
            selector.set_column("gent_lc", self.gentop, lazy=True)
            if self.topptweighter is not None:
                selector.add_cut(
                    "Top pt reweighting", self.do_top_pt_reweighting,
                    no_callback=True)
        if self.config["compute_systematics"] and is_mc:
            self.add_generator_uncertainies(dsname, selector)
        if is_mc:
            selector.add_cut(
                "Cross section", partial(self.crosssection_scale, dsname))

        if "blinding_denom" in self.config:
            selector.add_cut("Blinding", partial(self.blinding, is_mc))
        selector.add_cut("Lumi", partial(self.good_lumimask, is_mc, dsname))

        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.trigger_paths, self.trigger_order, era=era)
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
        selector.set_column("Lepton", self.build_lepton_column)
        # Wait with hists filling after channel masks are available
        selector.add_cut("At least 2 leps", partial(self.lepton_pair, is_mc),
                         no_callback=True)
        filler.channels = ("is_ee", "is_em", "is_mm")
        selector.set_multiple_columns(self.channel_masks)
        selector.set_column("mll", self.mll)
        selector.set_column("dilep_pt", self.dilep_pt, lazy=True)

        selector.applying_cuts = False

        selector.add_cut("Opposite sign", self.opposite_sign_lepton_pair)
        selector.add_cut("Chn trig match",
                         partial(self.channel_trigger_matching, era))
        if self.trigger_sfs is not None and is_mc:
            selector.add_cut(
                "Trigger SFs", partial(self.apply_trigger_sfs, dsname))
        selector.add_cut("Req lep pT", self.lep_pt_requirement)
        selector.add_cut("m_ll", self.good_mll)
        selector.add_cut("Z window", self.z_window)

        if (is_mc and self.config["compute_systematics"]
                and dsname not in self.config["dataset_for_systematics"]):
            if hasattr(filler, "sys_overwrite"):
                assert filler.sys_overwrite is None
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
            filler.sys_overwrite = None

        # Do normal, no-variation run
        self.process_selection_jet_part(selector, is_mc,
                                        self.get_jetmet_nominal_arg(),
                                        dsname, filler, era)
        logger.debug("Selection done")

    def get_jetmet_variation_args(self):
        ret = []
        ret.append(VariationArg("UncMET_up", met="up"))
        ret.append(VariationArg("UncMET_down", met="down"))
        if self._jer is not None and self._jersf is not None:
            jer = "central"
        else:
            jer = None
        if self._junc is not None:
            if "junc_sources_to_use" in self.config:
                levels = self.config["junc_sources_to_use"]
            else:
                levels = self._junc.levels
            for source in levels:
                if source not in self._junc.levels:
                    raise pepper.config.ConfigError(
                        f"Source not in jet uncertainties: {source}")
                if source == "jes":
                    name = "Junc_"
                else:
                    name = f"Junc{source}_"
                ret.append(VariationArg(
                    name + "up", junc=("up", source), jer=jer))
                ret.append(VariationArg(
                    name + "down", junc=("down", source), jer=jer))
        if self._jer is not None and self._jersf is not None:
            ret.append(VariationArg("Jer_up", jer="up"))
            ret.append(VariationArg("Jer_down", jer="down"))
        return ret

    def get_jetmet_nominal_arg(self):
        if self._jer is not None and self._jersf is not None:
            return VariationArg(None)
        else:
            return VariationArg(None, jer=None)

    def process_selection_jet_part(self, selector, is_mc, variation, dsname,
                                   filler, era):
        logger.debug(f"Running jet_part with variation {variation.name}")
        if is_mc:
            selector.set_multiple_columns(partial(
                self.compute_jet_factors, variation.junc, variation.jer,
                selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", self.build_jet_column)
        smear_met = "smear_met" in self.config and self.config["smear_met"]
        selector.set_column(
            "MET", partial(self.build_met_column, variation.junc,
                           variation.jer if smear_met else None, selector.rng,
                           era, variation=variation.met))
        selector.set_multiple_columns(
            partial(self.drellyan_sf_columns, filler))
        if self.drellyan_sf is not None and is_mc:
            selector.add_cut("DY scale", partial(self.apply_dy_sfs, dsname))
        selector.add_cut("Has jet(s)", self.has_jets)
        if (self.config["hem_cut_if_ele"] or self.config["hem_cut_if_muon"]
                or self.config["hem_cut_if_jet"]):
            selector.add_cut("HEM cut", self.hem_cut)
        selector.add_cut("Jet pt req", self.jet_pt_requirement)
        if is_mc and self.config["compute_systematics"]:
            self.scale_systematics_for_btag(selector, variation, dsname)
        selector.add_cut("Has btag(s)", partial(self.btag_cut, is_mc))
        selector.add_cut("Req MET", self.met_requirement)

        if "reco_algorithm" in self.config:
            reco_alg = self.config["reco_algorithm"]
            selector.set_column("recolepton", self.pick_leps, all_cuts=True)
            selector.set_column("recob", self.pick_bs, all_cuts=True)
            selector.set_column("recot", partial(
                self.ttbar_system, reco_alg.lower(), selector.rng),
                all_cuts=True, no_callback=True)
            selector.add_cut("Reco", self.passing_reco)
            selector.set_column("reconu", self.build_nu_column, all_cuts=True,
                                lazy=True)
            selector.set_column("dark_pt", self.calculate_dark_pt,
                                all_cuts=True, lazy=True)
            selector.set_column("chel", self.calculate_chel, all_cuts=True,
                                lazy=True)

    def gentop(self, data):
        part = data["GenPart"]
        part = part[~ak.is_none(part.parent, axis=1)]
        part = part[part.hasFlags("isLastCopy")]
        part = part[abs(part.pdgId) == 6]
        part = part[ak.argsort(part.pdgId, ascending=False)]
        return part

    def do_top_pt_reweighting(self, data):
        pt = data["gent_lc"].pt
        rwght = self.topptweighter(pt[:, 0], pt[:, 1])
        if self.topptweighter.sys_only:
            if self.config["compute_systematics"]:
                return np.full(len(data), True), {"Top_pt_reweighting": rwght}
            else:
                return np.full(len(data), True)
        else:
            if self.config["compute_systematics"]:
                return rwght, {"Top_pt_reweighting": 1/rwght}
            else:
                return rwght

    def add_generator_uncertainies(self, dsname, selector):
        # Matrix-element renormalization and factorization scale
        # Get describtion of individual columns of this branch with
        # Events->GetBranch("LHEScaleWeight")->GetTitle() in ROOT
        data = selector.data
        if dsname + "_LHEScaleSumw" in self.config["mc_lumifactors"]:
            norm = self.config["mc_lumifactors"][dsname + "_LHEScaleSumw"]
            # Workaround for https://github.com/cms-nanoAOD/cmssw/issues/537
            if len(norm) == 44:
                selector.set_systematic(
                    "MEren", data["LHEScaleWeight"][:, 34] * norm[34],
                    data["LHEScaleWeight"][:, 5] * norm[5])
                selector.set_systematic(
                    "MEfac", data["LHEScaleWeight"][:, 24] * norm[24],
                    data["LHEScaleWeight"][:, 15] * norm[15])
            else:
                selector.set_systematic(
                    "MEren", data["LHEScaleWeight"][:, 7] * norm[7],
                    data["LHEScaleWeight"][:, 1] * norm[1])
                selector.set_systematic(
                    "MEfac", data["LHEScaleWeight"][:, 5] * norm[5],
                    data["LHEScaleWeight"][:, 3] * norm[3])
        else:
            selector.set_systematic(
                "MEren", np.ones(len(data)), np.ones(len(data)))
            selector.set_systematic(
                "MEfac", np.ones(len(data)), np.ones(len(data)))
        # Parton shower scale
        psweight = data["PSWeight"]
        if len(psweight) > 0 and ak.num(psweight)[0] != 1:
            # NanoAOD containts one 1.0 per event in PSWeight if there are no
            # PS weights available, otherwise all counts > 1.
            if self.config["year"].startswith("ul"):
                # Workaround for PSWeight number changed their order in
                # NanoAODv8, meaning non-UL is unaffected
                selector.set_systematic(
                    "PSisr", psweight[:, 0], psweight[:, 2])
                selector.set_systematic(
                    "PSfsr", psweight[:, 1], psweight[:, 3])
            else:
                selector.set_systematic(
                    "PSisr", psweight[:, 2], psweight[:, 0])
                selector.set_systematic(
                    "PSfsr", psweight[:, 3], psweight[:, 1])
        else:
            selector.set_systematic(
                "PSisr", np.ones(len(data)), np.ones(len(data)))
            selector.set_systematic(
                "PSfsr", np.ones(len(data)), np.ones(len(data)))
        # Add PDF uncertainties, using the methods described here:
        # https://arxiv.org/pdf/1510.03865.pdf#section.6
        split_pdf_uncs = False
        if "split_pdf_uncs" in self.config:
            if self.config["split_pdf_uncs"]:
                split_pdf_uncs = True
        if (("LHEPdfWeight" not in data.fields) or (self.pdf_types is None)):
            return
        pdfs = data["LHEPdfWeight"]
        pdf_type = None
        for LHA_ID, _type in self.pdf_types.items():
            if LHA_ID in pdfs.__doc__:
                pdf_type = _type.lower()
        if split_pdf_uncs:
            # Just output variations - user
            # will need to combine these for limit setting
            selector.set_systematic("PDF",
                                    *[pdfs[:, i] for i
                                      in range(1, ak.num(pdfs)[0] - 2)],
                                    scheme="numeric")
        else:
            if pdf_type == "hessian":
                eigen_vals = np.reshape(ak.to_numpy(pdfs[:, 1:-2]),
                                        (len(pdfs), -1, 2))
                central, eigenvals = ak.broadcast_arrays(
                    pdfs[:, 0, None, None], eigen_vals)
                var_up = ak.max((eigen_vals - central), axis=2)
                var_up = ak.where(var_up > 0, var_up, 0)
                var_up = np.sqrt(ak.sum(var_up ** 2, axis=1))
                var_down = ak.max((central - eigen_vals), axis=2)
                var_down = ak.where(var_down > 0, var_down, 0)
                var_down = np.sqrt(ak.sum(var_down ** 2, axis=1))
                selector.set_systematic("PDF", 1 + var_up, 1 - var_down)
            elif pdf_type == "mc":
                # ak.sort produces an error here. Work-around:
                variations = np.sort(ak.to_numpy(pdfs[:, 1:-2]))
                nvar = ak.num(variations)[0]
                tot_unc = (variations[:, int(round(0.841344746*nvar))]
                           - variations[:, int(round(0.158655254*nvar))]) / 2
                selector.set_systematic("PDF", 1 + tot_unc, 1 - tot_unc)
            elif pdf_type == "mc_gaussian":
                mean = ak.mean(pdfs[:, 1:-2], axis=1)
                tot_unc = np.sqrt((ak.sum(pdfs[:, 1:-2] - mean) ** 2)
                                  / (ak.num(pdfs)[0] - 3))
                selector.set_systematic("PDF", 1 + tot_unc, 1 - tot_unc)
            elif pdf_type is None:
                raise pepper.config.ConfigError(
                    "PDF LHA Id not included in config. PDF docstring is: "
                    + pdfs.__doc__)
            else:
                raise pepper.config.ConfigError(
                    f"PDF type {pdf_type} not recognised. Valid options "
                    "are 'Hessian', 'MC' and 'MC_Gaussian'")
        # Add PDF alpha_s uncertainties
        unc = (pdfs[:, -1] - pdfs[:, -2]) / 2
        selector.set_systematic("PDF_alpha_s", 1 + unc, 1 - unc)

    def crosssection_scale(self, dsname, data):
        num_events = len(data)
        lumifactors = self.mc_lumifactors
        factor = np.full(num_events, lumifactors[dsname])
        if (self.config["compute_systematics"]
                and dsname not in self.config["dataset_for_systematics"]):
            xsuncerts = self.config["crosssection_uncertainty"]
            groups = set(v[0] for v in xsuncerts.values() if v is not None)
            systematics = {}
            for group in groups:
                if xsuncerts[dsname] is None or group != xsuncerts[dsname][0]:
                    uncert = 0
                else:
                    uncert = xsuncerts[dsname][1]
                systematics[group + "XS"] = (np.full(num_events, 1 + uncert),
                                             np.full(num_events, 1 - uncert))
            return factor, systematics
        else:
            return factor

    def blinding(self, is_mc, data):
        if not is_mc:
            return np.mod(data["event"], self.config["blinding_denom"]) == 0
        else:
            return (np.full(data.size, True),
                    {"Blinding_sf":
                     np.full(data.size, 1/self.config["blinding_denom"])})

    def good_lumimask(self, is_mc, dsname, data):
        if is_mc:
            # Lumimask only present in data, all events pass in MC
            # Compute pileup reweighting and lumi variation here
            ntrueint = data["Pileup"]["nTrueInt"]
            sys = {}
            if self.puweighter is not None:
                weight = self.puweighter(dsname, ntrueint)
                if self.config["compute_systematics"]:
                    # If central is zero, let up and down factors also be zero
                    weight_nonzero = np.where(weight == 0, np.inf, weight)
                    up = self.puweighter(dsname, ntrueint, "up")
                    down = self.puweighter(dsname, ntrueint, "down")
                    sys["pileup"] = (up / weight_nonzero,
                                     down / weight_nonzero)
            else:
                weight = np.ones(len(data))
            if self.config["compute_systematics"]:
                if self.config["year"] in ("2018", "2016", "ul2018",
                                           "ul2016pre", "ul2016post"):
                    sys["lumi"] = (np.full(len(data), 1 + 0.025),
                                   np.full(len(data), 1 - 0.025))
                elif self.config["year"] in ("2017", "ul2017"):
                    sys["lumi"] = (np.full(len(data), 1 + 0.023),
                                   np.full(len(data), 1 - 0.023))
                return weight, sys
            else:
                return weight
        elif self.lumimask is None:
            return np.full(len(data), True)
        else:
            run = np.array(data["run"])
            luminosity_block = np.array(data["luminosityBlock"])
            lumimask = coffea.lumi_tools.LumiMask(self.lumimask)
            return lumimask(run, luminosity_block)

    def get_era(self, data, is_mc):
        if is_mc:
            return self.config["year"] + "MC"
        else:
            if len(data) == 0:
                return "no_events"
            run = np.array(data["run"])[0]
            # Assumes all runs in file come from same era
            for era, startstop in self.config["data_eras"].items():
                if ((run >= startstop[0]) & (run <= startstop[1])):
                    return era
            raise ValueError(f"Run {run} does not correspond to any era")

    def passing_trigger(self, pos_triggers, neg_triggers, data):
        hlt = data["HLT"]
        trigger = (
            np.any([np.asarray(hlt[trigger_path])
                    for trigger_path in pos_triggers], axis=0)
            & ~np.any([np.asarray(hlt[trigger_path])
                       for trigger_path in neg_triggers], axis=0)
        )
        return trigger

    def add_l1_prefiring_weights(self, data):
        w = data["L1PreFiringWeight"]
        nom = w["Nom"]
        if self.config["compute_systematics"]:
            sys = {"L1 prefiring": (w["Up"] / nom, w["Dn"] / nom)}
            return nom, sys
        return nom

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
                (data["Flag"]["goodVertices"]
                 & data["Flag"]["globalSuperTightHalo2016Filter"]
                 & data["Flag"]["HBHENoiseFilter"]
                 & data["Flag"]["HBHENoiseIsoFilter"]
                 & data["Flag"]["EcalDeadCellTriggerPrimitiveFilter"]
                 & data["Flag"]["BadPFMuonFilter"])
            if not is_mc:
                passing_filters = (
                    passing_filters & data["Flag"]["eeBadScFilter"])
        if year in ("2018", "2017"):
            passing_filters = (
                passing_filters & data["Flag"]["ecalBadCalibFilterV2"])
        if year in ("ul2018", "ul2017"):
            passing_filters = (
                passing_filters & data["Flag"]["ecalBadCalibFilter"])
        if year in ("ul2018", "ul2017", "ul2016"):
            passing_filters = (
                passing_filters & data["Flag"]["eeBadScFilter"])

        return passing_filters

    def in_transreg(self, abs_eta):
        return (1.444 < abs_eta) & (abs_eta < 1.566)

    def electron_id(self, e_id, electron):
        if e_id == "skip":
            has_id = True
        if e_id == "cut:loose":
            has_id = electron["cutBased"] >= 2
        elif e_id == "cut:medium":
            has_id = electron["cutBased"] >= 3
        elif e_id == "cut:tight":
            has_id = electron["cutBased"] >= 4
        elif e_id == "mva:noIso80":
            has_id = electron["mvaFall17V2noIso_WP80"]
        elif e_id == "mva:noIso90":
            has_id = electron["mvaFall17V2noIso_WP90"]
        elif e_id == "mva:Iso80":
            has_id = electron["mvaFall17V2Iso_WP80"]
        elif e_id == "mva:Iso90":
            has_id = electron["mvaFall17V2Iso_WP90"]
        else:
            raise ValueError("Invalid electron id string")
        return has_id

    def electron_cuts(self, electron, good_lep):
        if self.config["ele_cut_transreg"]:
            sc_eta_abs = abs(electron["eta"]
                             + electron["deltaEtaSC"])
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
        return (self.electron_id(e_id, electron)
                & (~is_in_transreg)
                & (eta_min < electron["eta"])
                & (electron["eta"] < eta_max)
                & (pt_min < electron["pt"]))

    def pick_electrons(self, data):
        electrons = data["Electron"]
        return electrons[self.electron_cuts(electrons, good_lep=True)]

    def muon_id(self, m_id, muon):
        if m_id == "skip":
            has_id = True
        elif m_id == "cut:loose":
            has_id = muon["looseId"]
        elif m_id == "cut:medium":
            has_id = muon["mediumId"]
        elif m_id == "cut:tight":
            has_id = muon["tightId"]
        elif m_id == "mva:loose":
            has_id = muon["mvaId"] >= 1
        elif m_id == "mva:medium":
            has_id = muon["mvaId"] >= 2
        elif m_id == "mva:tight":
            has_id = muon["mvaId"] >= 3
        else:
            raise ValueError("Invalid muon id string")
        return has_id

    def muon_iso(self, iso, muon):
        if iso == "skip":
            return True
        elif iso == "cut:very_loose":
            return muon["pfIsoId"] > 0
        elif iso == "cut:loose":
            return muon["pfIsoId"] > 1
        elif iso == "cut:medium":
            return muon["pfIsoId"] > 2
        elif iso == "cut:tight":
            return muon["pfIsoId"] > 3
        elif iso == "cut:very_tight":
            return muon["pfIsoId"] > 4
        elif iso == "cut:very_very_tight":
            return muon["pfIsoId"] > 5
        else:
            iso, iso_value = iso.split(":")
            value = float(iso_value)
            if iso == "dR<0.3_chg":
                return muon["pfRelIso03_chg"] < value
            elif iso == "dR<0.3_all":
                return muon["pfRelIso03_all"] < value
            elif iso == "dR<0.4_all":
                return muon["pfRelIso04_all"] < value
        raise ValueError("Invalid muon iso string")

    def muon_cuts(self, muon, good_lep):
        if self.config["muon_cut_transreg"]:
            is_in_transreg = self.in_transreg(abs(muon["eta"]))
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
        return (self.muon_id(m_id, muon)
                & self.muon_iso(iso, muon)
                & (~is_in_transreg)
                & (eta_min < muon["eta"])
                & (muon["eta"] < eta_max)
                & (pt_min < muon["pt"]))

    def pick_muons(self, data):
        muons = data["Muon"]
        return muons[self.muon_cuts(muons, good_lep=True)]

    def build_lepton_column(self, data):
        electron = data["Electron"]
        muon = data["Muon"]
        columns = ["pt", "eta", "phi", "mass", "pdgId"]
        lepton = {}
        for column in columns:
            lepton[column] = ak.concatenate([electron[column], muon[column]],
                                            axis=1)
        lepton = ak.zip(lepton, with_name="PtEtaPhiMLorentzVector",
                        behavior=data.behavior)

        # Sort leptons by pt
        lepton = lepton[ak.argsort(lepton["pt"], ascending=False)]
        return lepton

    def channel_masks(self, data):
        leps = data["Lepton"]
        channels = {}
        channels["is_ee"] = ((abs(leps[:, 0].pdgId) == 11)
                             & (abs(leps[:, 1].pdgId) == 11))
        channels["is_mm"] = ((abs(leps[:, 0].pdgId) == 13)
                             & (abs(leps[:, 1].pdgId) == 13))
        channels["is_em"] = (~channels["is_ee"]) & (~channels["is_mm"])
        return channels

    def compute_lepton_sf(self, data):
        eles = data["Electron"]
        muons = data["Muon"]

        weight = np.ones(len(data))
        systematics = {}
        # Electron identification efficiency
        for i, sffunc in enumerate(self.electron_sf):
            sceta = eles.eta + eles.deltaEtaSC
            params = {}
            for dimlabel in sffunc.dimlabels:
                if dimlabel == "abseta":
                    params["abseta"] = abs(sceta)
                elif dimlabel == "eta":
                    params["eta"] = sceta
                else:
                    params[dimlabel] = getattr(eles, dimlabel)
            central = ak.prod(sffunc(**params), axis=1)
            key = "electronsf{}".format(i)
            if self.config["compute_systematics"]:
                up = ak.prod(sffunc(**params, variation="up"), axis=1)
                down = ak.prod(sffunc(**params, variation="down"), axis=1)
                systematics[key] = (up / central, down / central)
            weight = weight * central
        # Muon identification and isolation efficiency
        for i, sffunc in enumerate(self.muon_sf):
            params = {}
            for dimlabel in sffunc.dimlabels:
                if dimlabel == "abseta":
                    params["abseta"] = abs(muons.eta)
                else:
                    params[dimlabel] = getattr(muons, dimlabel)
            central = ak.prod(sffunc(**params), axis=1)
            key = "muonsf{}".format(i)
            if self.config["compute_systematics"]:
                up = ak.prod(sffunc(**params, variation="up"), axis=1)
                down = ak.prod(sffunc(**params, variation="down"), axis=1)
                systematics[key] = (up / central, down / central)
            weight = weight * central
        return weight, systematics

    def lepton_pair(self, is_mc, data):
        accept = np.asarray(ak.num(data["Lepton"]) >= 2)
        if is_mc:
            weight, systematics = self.compute_lepton_sf(data[accept])
            accept = accept.astype(float)
            accept[accept.astype(bool)] *= np.asarray(weight)
            return accept, systematics
        else:
            return accept

    def opposite_sign_lepton_pair(self, data):
        return (np.sign(data["Lepton"][:, 0].pdgId)
                != np.sign(data["Lepton"][:, 1].pdgId))

    def same_flavor(self, data):
        return (abs(data["Lepton"][:, 0].pdgId)
                == abs(data["Lepton"][:, 1].pdgId))

    def mll(self, data):
        return (data["Lepton"][:, 0] + data["Lepton"][:, 1]).mass

    def dilep_pt(self, data):
        return (data["Lepton"][:, 0] + data["Lepton"][:, 1]).pt

    def compute_junc_factor(self, data, variation, source="jes", pt=None,
                            eta=None):
        if variation not in ("up", "down"):
            raise ValueError("variation must be either 'up' or 'down'")
        if source not in self._junc.levels:
            raise ValueError(f"Jet uncertainty not found: {source}")
        if pt is None:
            pt = data["Jet"].pt
        if eta is None:
            eta = data["Jet"].eta
        counts = ak.num(pt)
        if ak.sum(counts) == 0:
            return ak.unflatten([], counts)
        # TODO: test this
        junc = dict(self._junc.getUncertainty(JetPt=pt, JetEta=eta))[source]
        if variation == "up":
            return junc[:, :, 0]
        else:
            return junc[:, :, 1]

    def compute_jer_factor(self, data, rng, variation="central", pt=None,
                           eta=None, hybrid=True):
        # Coffea offers a class named JetTransformer for this. Unfortunately
        # it is more inconvinient and bugged than useful.
        if pt is None:
            pt = data["Jet"].pt
        if eta is None:
            eta = data["Jet"].eta
        counts = ak.num(pt)
        if ak.sum(counts) == 0:
            return ak.unflatten([], counts)
        jer = self._jer.getResolution(
            JetPt=pt, JetEta=eta, Rho=data["fixedGridRhoFastjetAll"])
        jersf = self._jersf.getScaleFactor(
            JetPt=pt, JetEta=eta, Rho=data["fixedGridRhoFastjetAll"])
        jersmear = jer * rng.normal(size=len(jer))
        if variation == "central":
            jersf = jersf[:, :, 0]
        elif variation == "up":
            jersf = jersf[:, :, 1]
        elif variation == "down":
            jersf = jersf[:, :, 2]
        else:
            raise ValueError("variation must be one of 'central', 'up' or "
                             "'down'")
        factor_stoch = 1 + np.sqrt(np.maximum(jersf**2 - 1, 0)) * jersmear
        if hybrid:
            # Hybrid method: Apply scaling relative to genpt if possible
            genpt = data["Jet"].matched_gen.pt
            factor_scale = 1 + (jersf - 1) * (pt - genpt) / pt
            factor = ak.where(
                ak.is_none(genpt, axis=1), factor_stoch, factor_scale)
        else:
            factor = factor_stoch
        return factor

    def compute_jet_factors(self, junc, jer, rng, data):
        if junc is not None:
            juncfac = self.compute_junc_factor(data, *junc)
        else:
            juncfac = ak.ones_like(data["Jet"].pt)
        if jer is not None:
            jerfac = self.compute_jer_factor(data, rng, jer)
        else:
            jerfac = ak.ones_like(data["Jet"].pt)
        factor = juncfac * jerfac
        return {"jetfac": factor, "juncfac": juncfac, "jerfac": jerfac}

    def good_jet(self, data):
        jets = data["Jet"]
        leptons = data["Lepton"]
        j_id, j_puId, lep_dist, eta_min, eta_max, pt_min = self.config[[
            "good_jet_id", "good_jet_puId", "good_jet_lepton_distance",
            "good_jet_eta_min", "good_jet_eta_max", "good_jet_pt_min"]]
        if j_id == "skip":
            has_id = True
        elif j_id == "cut:loose":
            has_id = jets.isLoose
            # Always False in 2017 and 2018
        elif j_id == "cut:tight":
            has_id = jets.isTight
        elif j_id == "cut:tightlepveto":
            has_id = jets.isTightLeptonVeto
        else:
            raise pepper.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_id))
        if j_puId == "skip":
            has_puId = True
        elif j_puId == "cut:loose":
            has_puId = ak.values_astype(jets["puId"] & 0b100, bool)
        elif j_puId == "cut:medium":
            has_puId = ak.values_astype(jets["puId"] & 0b10, bool)
        elif j_puId == "cut:tight":
            has_puId = ak.values_astype(jets["puId"] & 0b1, bool)
        else:
            raise pepper.config.ConfigError(
                    "Invalid good_jet_id: {}".format(j_puId))
        # Only apply PUID if pT < 50 GeV
        has_puId = has_puId | (jets.pt >= 50)

        j_pt = jets.pt
        if "jetfac" in ak.fields(data):
            j_pt = j_pt * data["jetfac"]
        j_eta = jets.eta
        j_phi = jets.phi
        l_eta = leptons.eta
        l_phi = leptons.phi
        j_eta, l_eta = ak.unzip(ak.cartesian([j_eta, l_eta], nested=True))
        j_phi, l_phi = ak.unzip(ak.cartesian([j_phi, l_phi], nested=True))
        delta_eta = j_eta - l_eta
        delta_phi = j_phi - l_phi
        delta_r = np.hypot(delta_eta, delta_phi)
        has_lepton_close = ak.any(delta_r < lep_dist, axis=2)

        return (has_id & has_puId
                & (~has_lepton_close)
                & (eta_min < jets.eta)
                & (jets.eta < eta_max)
                & (pt_min < j_pt))

    def build_jet_column(self, data):
        is_good_jet = self.good_jet(data)
        jets = data["Jet"][is_good_jet]
        if "jetfac" in ak.fields(data):
            jets["pt"] = jets["pt"] * data["jetfac"][is_good_jet]
            jets["mass"] = jets["mass"] * data["jetfac"][is_good_jet]

        # Evaluate b-tagging
        tagger, wp = self.config["btag"].split(":")
        if tagger == "deepcsv":
            jets["btag"] = jets["btagDeepB"]
        elif tagger == "deepjet":
            jets["btag"] = jets["btagDeepFlavB"]
        else:
            raise pepper.config.ConfigError(
                "Invalid tagger name: {}".format(tagger))
        year = self.config["year"]
        wptuple = pepper.scale_factors.BTAG_WP_CUTS[tagger][year]
        if not hasattr(wptuple, wp):
            raise pepper.config.ConfigError(
                "Invalid working point \"{}\" for {} in year {}".format(
                    wp, tagger, year))
        jets["btagged"] = jets["btag"] > getattr(wptuple, wp)

        return jets

    def build_lowptjet_column(self, junc, jer, rng, data):
        jets = data["CorrT1METJet"]
        # For MET we care about jets close to 15 GeV. JEC is derived with
        # pt > 10 GeV and |eta| < 5.2, thus cut there
        jets = jets[(jets.rawPt > 10) & (abs(jets.eta) < 5.2)]
        l1l2l3 = self._jec.getCorrection(
            JetPt=jets.rawPt, JetEta=jets.eta, JetA=jets.area,
            Rho=data["fixedGridRhoFastjetAll"])
        jets["pt"] = l1l2l3 * jets.rawPt
        jets["pt_nomuon"] = jets["pt"] * (1 - jets["muonSubtrFactor"])

        jets["factor"] = ak.ones_like(jets["pt"])
        if junc is not None:
            jets["factor"] = jets["factor"] * self.compute_junc_factor(
                data, *junc, pt=jets["pt"], eta=jets["eta"])
        if jer is not None:
            jets["factor"] = jets["factor"] * self.compute_jer_factor(
                data, rng, jer, jets["pt"], jets["eta"], False)

        jets["emef"] = jets["mass"] = ak.zeros_like(jets["pt"])
        jets.behavior = data["Jet"].behavior
        jets = ak.with_name(jets, "Jet")

        return jets

    def build_met_column(self, junc, jer, rng, era, data, variation="central"):
        met = data["MET"]
        metx = met.pt * np.cos(met.phi)
        mety = met.pt * np.sin(met.phi)
        if (self.met_xy_shifts and era != "no_events"):
            metx += -(self.met_xy_shifts["METxcorr"][era][0]
                      * data["PV"]["npvs"]
                      + self.met_xy_shifts["METxcorr"][era][1])
            mety += -(self.met_xy_shifts["METycorr"][era][0]
                      * data["PV"]["npvs"]
                      + self.met_xy_shifts["METycorr"][era][1])

        if variation == "up":
            metx = metx + met.MetUnclustEnUpDeltaX
            mety = mety + met.MetUnclustEnUpDeltaY
        elif variation == "down":
            metx = metx - met.MetUnclustEnUpDeltaX
            mety = mety - met.MetUnclustEnUpDeltaY
        elif variation != "central":
            raise ValueError(
                "variation must be one of 'central', 'up' or 'down'")
        if jer is not None and "jetfac" in ak.fields(data):
            factors = data["jetfac"]
        elif "juncfac" in ak.fields(data):
            factors = data["juncfac"]
        else:
            factors = None
        if factors is not None and ak.any(factors != 1):
            # Do MET type-1 and smearing corrections
            jets = data["OrigJet"]
            jets = ak.zip({
                "pt": jets.pt,
                "pt_nomuon": jets.pt * (1 - jets.muonSubtrFactor),
                "factor": factors - 1,
                "eta": jets.eta,
                "phi": jets.phi,
                "mass": jets.mass,
                "emef": jets.neEmEF + jets.chEmEF
            }, with_name="Jet", behavior=jets.behavior)
            lowptjets = self.build_lowptjet_column(junc, jer, rng, data)
            # Cut according to MissingETRun2Corrections Twiki
            jets = jets[(jets["pt_nomuon"] > 15) & (jets["emef"] < 0.9)]
            lowptjets = lowptjets[(lowptjets["pt_nomuon"] > 15)
                                  & (lowptjets["emef"] < 0.9)]
            # lowptjets lose their type here. Probably a bug, workaround
            lowptjets = ak.with_name(lowptjets, "Jet")
            lowptfac = lowptjets["factor"] - 1
            metx = metx - (
                ak.sum(jets.x * jets["factor"], axis=1)
                + ak.sum(lowptjets.x * lowptfac, axis=1))
            mety = mety - (
                ak.sum(jets.y * jets["factor"], axis=1)
                + ak.sum(lowptjets.y * lowptfac, axis=1))
        met["pt"] = np.hypot(metx, mety)
        met["phi"] = np.arctan2(mety, metx)
        return met

    def apply_trigger_sfs(self, dsname, data):
        leps = data["Lepton"]
        ones = np.ones(len(data))
        central = ones
        channels = ["is_ee", "is_em", "is_mm"]
        for channel in channels:
            sf = self.trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                           lep2_pt=leps[:, 1].pt)
            central = ak.where(data[channel], sf, central)
        if self.config["compute_systematics"]:
            up = ones
            down = ones
            for channel in channels:
                sf = self.trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
                                               lep2_pt=leps[:, 1].pt,
                                               variation="up")
                up = ak.where(data[channel], sf, up)
                sf = self.trigger_sfs[channel](lep1_pt=leps[:, 0].pt,
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
            central = self.drellyan_sf(**params)
            if self.config["compute_systematics"]:
                up = self.drellyan_sf(**params, variation="up")
                down = self.drellyan_sf(**params, variation="down")
                return central, {"DYsf": (up / central, down / central)}
            return central
        elif self.config["compute_systematics"]:
            ones = np.ones(len(data))
            return np.full(len(data), True), {"DYsf": (ones, ones)}
        return np.full(len(data), True)

    def drellyan_sf_columns(self, data, filler):
        # Dummy function, overwritten when computing DY SFs
        return {}

    def in_hem1516(self, phi, eta):
        return ((-3.0 < eta) & (eta < -1.3) & (-1.57 < phi) & (phi < -0.87))

    def hem_cut(self, data):
        cut_ele = self.config["hem_cut_if_ele"]
        cut_muon = self.config["hem_cut_if_muon"]
        cut_jet = self.config["hem_cut_if_jet"]

        keep = np.full(len(data), True)
        if cut_ele:
            ele = data["Electron"]
            keep = keep & (~self.in_hem1516(ele.phi, ele.eta).any())
        if cut_muon:
            muon = data["Muon"]
            keep = keep & (~self.in_hem1516(muon.phi, muon.eta).any())
        if cut_jet:
            jet = data["Jet"]
            keep = keep & (~self.in_hem1516(jet.phi, jet.eta).any())
        return keep

    def channel_trigger_matching(self, era, data):
        is_ee = data["is_ee"]
        is_mm = data["is_mm"]
        is_em = data["is_em"]
        triggers = self.config["channel_trigger_map"]

        ret = np.full(len(data), False)
        check = [
            (is_ee, "ee"), (is_mm, "mumu"), (is_em, "emu"),
            (is_ee | is_em, "e"), (is_mm | is_em, "mu")]
        for mask, trigname in check:
            if (trigname + "_" + era) in triggers:
                trigger = [pepper.misc.normalize_trigger_path(t)
                           for t in triggers[trigname + "_" + era]]
                ret = ret | (
                    mask & self.passing_trigger(trigger, [], data))
            elif trigname in triggers:
                trigger = [pepper.misc.normalize_trigger_path(t)
                           for t in triggers[trigname]]
                ret = ret | (
                    mask & self.passing_trigger(trigger, [], data))

        return ret

    def lep_pt_requirement(self, data):
        n = np.zeros(len(data))
        # This assumes leptons are ordered by pt highest first
        for i, pt_min in enumerate(self.config["lep_pt_min"]):
            mask = ak.num(data["Lepton"]) > i
            n[mask] += np.asarray(
                pt_min < data["Lepton"].pt[mask, i]).astype(int)
        return n >= self.config["lep_pt_num_satisfied"]

    def good_mll(self, data):
        return data["mll"] > self.config["mll_min"]

    def no_additional_leptons(self, is_mc, data):
        add_ele = self.electron_cuts(data["Electron"], good_lep=False)
        add_muon = self.muon_cuts(data["Muon"], good_lep=False)
        return ak.sum(add_ele, axis=1) + ak.sum(add_muon, axis=1) <= 2

    def z_window(self, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        is_out_window = (data["mll"] <= m_min) | (m_max <= data["mll"])
        return data["is_em"] | is_out_window

    def has_jets(self, data):
        return self.config["num_jets_atleast"] <= ak.num(data["Jet"])

    def jet_pt_requirement(self, data):
        n = np.zeros(len(data))
        # This assumes jets are ordered by pt highest first
        for i, pt_min in enumerate(self.config["jet_pt_min"]):
            mask = ak.num(data["Jet"]) > i
            n[mask] += np.asarray(pt_min < data["Jet"].pt[mask, i]).astype(int)
        return n >= self.config["jet_pt_num_satisfied"]

    def compute_weight_btag(self, data, efficiency="central", never_sys=False):
        jets = data["Jet"]
        wp = self.config["btag"].split(":", 1)[1]
        flav = jets["hadronFlavour"]
        eta = jets.eta
        pt = jets.pt
        discr = jets["btag"]
        weight = np.ones(len(data))
        systematics = {}
        for i, weighter in enumerate(self.btagweighters):
            central = weighter(wp, flav, eta, pt, discr, "central", efficiency)
            if not never_sys and self.config["compute_systematics"]:
                light_up = weighter(
                    wp, flav, eta, pt, discr, "light up", efficiency)
                light_down = weighter(
                    wp, flav, eta, pt, discr, "light down", efficiency)
                up = weighter(
                    wp, flav, eta, pt, discr, "heavy up", efficiency)
                down = weighter(
                    wp, flav, eta, pt, discr, "heavy down", efficiency)
                systematics[f"btagsf{i}"] = (up / central, down / central)
                systematics[f"btagsf{i}light"] = (
                    light_up / central, light_down / central)
            weight = weight * central
        if never_sys:
            return weight
        else:
            return weight, systematics

    def scale_systematics_for_btag(self, selector, variation, dsname):
        """Modifies factors in the systematic table to account for differences
        in b-tag efficiencies. This is only done for variations the efficiency
        ROOT file contains a histogram with the name of the variation."""
        if len(self.btagweighters) == 0:
            return
        available = set.intersection(
            *(w.available_efficiencies for w in self.btagweighters))
        data = selector.data
        systematics = selector.systematics
        central = self.compute_weight_btag(data, never_sys=True)
        if (variation == self.get_jetmet_nominal_arg()
                and dsname not in self.config["dataset_for_systematics"]):
            for name in ak.fields(systematics):
                if name == "weight":
                    continue
                if name not in available:
                    continue
                sys = systematics[name]
                varied_sf = self.compute_weight_btag(data, name, True)
                selector.set_systematic(name, sys / central * varied_sf)
        elif dsname in self.config["dataset_for_systematics"]:
            name = self.config["dataset_for_systematics"][dsname][1]
            if name in available:
                varied_sf = self.compute_weight_btag(data, name, True)
                selector.set_systematic("weight", sys / central * varied_sf)
        elif variation.name in available:
            varied_sf = self.compute_weight_btag(data, variation.name, True)
            selector.set_systematic("weight", sys / central * varied_sf)

    def btag_cut(self, is_mc, data):
        num_btagged = ak.sum(data["Jet"]["btagged"], axis=1)
        accept = np.asarray(num_btagged >= self.config["num_atleast_btagged"])
        if is_mc and len(self.btagweighters) != 0:
            weight, systematics = self.compute_weight_btag(data[accept])
            accept = accept.astype(float)
            accept[accept.astype(bool)] *= np.asarray(weight)
            return accept, systematics
        else:
            return accept

    def met_requirement(self, data):
        met = data["MET"].pt
        return data["is_em"] | (met > self.config["ee/mm_min_met"])

    def pick_leps(self, data):
        # Sort so that we get the order [lepton, antilepton]
        return data["Lepton"][
            ak.argsort(data["Lepton"]["pdgId"], ascending=False)]

    def pick_bs(self, data):
        recolepton = data["recolepton"]
        lep = recolepton[:, 0]
        antilep = recolepton[:, 1]
        # Build a reduced jet collection to avoid loading all branches and
        # make make this function faster overall
        columns = ["pt", "eta", "phi", "mass", "btagged"]
        jets = ak.with_name(data["Jet"][columns], "PtEtaPhiMLorentzVector")
        btags = jets[data["Jet"].btagged]
        jetsnob = jets[~data["Jet"].btagged]
        num_btags = ak.num(btags)
        b0, b1 = ak.unzip(ak.where(
            num_btags > 1, ak.combinations(btags, 2),
            ak.where(
                num_btags == 1, ak.cartesian([btags, jetsnob]),
                ak.combinations(jetsnob, 2))))
        bs = ak.concatenate([b0, b1], axis=1)
        bs_rev = ak.concatenate([b1, b0], axis=1)
        mass_alb = reduce(
            lambda a, b: a + b, ak.unzip(ak.cartesian([bs, antilep]))).mass
        mass_lb = reduce(
            lambda a, b: a + b, ak.unzip(ak.cartesian([bs_rev, lep]))).mass
        with uproot.open(self.reco_info_filepath) as f:
            mlb_prob = pepper.scale_factors.ScaleFactors.from_hist(f["mlb"])
        p_m_alb = mlb_prob(mlb=mass_alb)
        p_m_lb = mlb_prob(mlb=mass_lb)
        bestbpair_mlb = ak.unflatten(
            ak.argmax(p_m_alb * p_m_lb, axis=1), np.full(len(bs), 1))
        return ak.concatenate([bs[bestbpair_mlb], bs_rev[bestbpair_mlb]],
                              axis=1)

    def ttbar_system(self, reco_alg, rng, data):
        lep = data["recolepton"][:, 0]
        antilep = data["recolepton"][:, 1]
        b = data["recob"][:, 0]
        antib = data["recob"][:, 1]
        met = data["MET"]

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
            if reco_alg == "sonnenschein":
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
                alphaj=alphaj, hist_mlb=mlb, num_smear=num_smear, rng=rng)
            return ak.concatenate([top, antitop], axis=1)
        elif reco_alg == "betchart":
            top, antitop = betchart(lep, antilep, b, antib, met)
            return ak.concatenate([top, antitop], axis=1)
        else:
            raise ValueError(f"Invalid value for reco algorithm: {reco_alg}")

    def passing_reco(self, data):
        return ak.num(data["recot"]) > 0

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
