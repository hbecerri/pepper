from functools import partial

import hjson

import pepper


class TriggerSFConfig(pepper.ConfigTTbarLL):
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
        self["exp_datasets"] = self["MET_trigger_datasets"]


class TriggerSFProducer(pepper.ProcessorTTbarLL):

    config_class = TriggerSFConfig

    def preprocess(self, datasets):
        processed = {}
        for key, value in datasets.items():
            if (key in self.config["exp_datasets"]
                    or key.startswith("TTTo2L2Nu")):
                processed[key] = value
        return processed

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)
        if dsname.startswith("TTTo"):
            selector.set_column("gent_lc", self.gentop, lazy=True)
            if self.topptweighter is not None:
                selector.add_cut(
                    "Top pt reweighting", self.do_top_pt_reweighting,
                    no_callback=True)
        if is_mc:
            selector.add_cut(
                "Cross section", partial(self.crosssection_scale, dsname))

        selector.add_cut("Lumi", partial(self.good_lumimask, is_mc, dsname))

        selector.set_column("MET triggers", partial(
            self.passing_trigger, self.config["MET_triggers"], []))

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
        selector.set_column("dilep triggers", partial(
            self.dilep_triggers, dsname, era))
        selector.set_column("mll", self.mll)
        selector.set_column("dilep_pt", self.dilep_pt, lazy=True)

        selector.add_cut("Opposite sign", self.opposite_sign_lepton_pair)
        selector.add_cut("Req lep pT", self.lep_pt_requirement)

        # Set jets and MET now so the can be used in sys calculations if
        # Mll cut used as WP
        variation = self.get_jetmet_nominal_arg()
        if is_mc:
            selector.set_multiple_columns(partial(
                self.compute_jet_factors, variation.junc, variation.jer,
                selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", self.build_jet_column)
        selector.set_column(
            "MET", partial(self.build_met_column, variation.junc, is_mc,
                           dsname, era, variation=variation.met))

        selector.add_cut("m_ll", self.good_mll)
        selector.add_cut("Z window", self.z_window)

        selector.add_cut("Has jet(s)", self.has_jets)
        if (self.config["hem_cut_if_ele"] or self.config["hem_cut_if_muon"]
                or self.config["hem_cut_if_jet"]):
            selector.add_cut("HEM cut", self.hem_cut)
        selector.add_cut("Jet pt req", self.jet_pt_requirement)
        selector.add_cut("Has btag(s)", partial(self.btag_cut, is_mc))
        selector.add_cut("Req MET", self.met_requirement)

    def dilep_triggers(self, dsname, era, data):
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, True, self.trigger_paths, self.trigger_order, era=era)
        trig = self.passing_trigger(pos_triggers, neg_triggers, data)
        chmatch = self.channel_trigger_matching(era, data)
        return (trig & chmatch)


if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        TriggerSFProducer, "Run the TriggerSFProducer to get "
        "the numbers needed for the trigger SF calculation")
