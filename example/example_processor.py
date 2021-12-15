# This file illustrates how to implement a processor, realizing the selection
# steps and outputting histograms and a cutflow with efficiencies.
# Here we create a very simplified version of the ttbar-to-dilep processor.
# One can run this processor using
# 'python3 -m pepper.runproc --debug example_processor.py example_config.json'
# Above command probably will need a little bit of time before all cuts are
# applied once. This is because a chunk of events are processed simultaneously.
# You change adjust the number of events in a chunk and thereby the memory
# usage by using the --chunksize parameter (the default value is 500000).

import pepper
import awkward as ak
from functools import partial


# All processors should inherit from pepper.ProcessorBasicPhysics
class Processor(pepper.ProcessorBasicPhysics):
    # We use the ConfigTTbarLL instead of its base Config, to use some of its
    # predefined extras
    config_class = pepper.ConfigTTbarLL

    def __init__(self, config, eventdir):
        # Initialize the class, maybe overwrite some config variables and
        # load additional files if needed
        # Can set and modify configuration here as well
        config["histogram_format"] = "root"
        # Need to call parent init to make histograms and such ready
        super().__init__(config, eventdir)

        # It is not recommended to put anything as member variable into a
        # a Processor because the Processor instance is sent as raw bytes
        # between nodes when running on HTCondor.

    def process_selection(self, selector, dsname, is_mc, filler):
        # Implement the selection steps: add cuts, define objects and/or
        # compute event weights

        # Add a cut only allowing events according to the golden JSON
        # The good_lumimask method is specified in pepper.ProcessorBasicPhysics
        # It also requires a lumimask to be specified in config
        if not is_mc:
            selector.add_cut("Lumi", partial(
                self.good_lumimask, is_mc, dsname))

        # Only allow events that pass triggers specified in config
        # This also takes into account a trigger order to avoid triggering
        # the same event if it's in two different data datasets.
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"])
        selector.add_cut("Trigger", partial(
            self.passing_trigger, pos_triggers, neg_triggers))

        # Pick electrons satisfying our criterias
        selector.set_column("Electron", self.pick_electrons)
        # Also pick muons
        selector.set_column("Muon", self.pick_muons)

        # Only accept events that have to leptons
        selector.add_cut("Exactly 2 leptons", self.lepton_pair)

        # Only accept events that have oppositely changed leptons
        selector.add_cut("OC leptons", self.opposite_sign_lepton_pair)

        # Things that could be done next: Adding cuts on the jets, MET or
        # only allowing events that have a certain m_ll. A full implementation
        # can be found in processor_ttbarll.py.

    def pick_electrons(self, data):
        ele = data["Electron"]

        # We do not want electrons that are between the barrel and the end cap
        # For this, we need the eta of the electron with respect to its
        # supercluster
        sc_eta_abs = abs(ele.eta + ele.deltaEtaSC)
        is_in_transreg = (1.444 < sc_eta_abs) & (sc_eta_abs < 1.566)

        # Electron ID, as an example we use the MVA one here
        has_id = ele.mvaFall17V2Iso_WP90

        # Finally combine all the requirements
        is_good = (
            has_id
            & (~is_in_transreg)
            & (self.config["ele_eta_min"] < ele.eta)
            & (ele.eta < self.config["ele_eta_max"])
            & (self.config["good_ele_pt_min"] < ele.pt))

        # Return all electrons with are deemed to be good
        return ele[is_good]

    def pick_muons(self, data):
        muon = data["Muon"]
        has_id = muon.mediumId
        has_iso = muon.pfIsoId > 1
        is_good = (
            has_id
            & has_iso
            & (self.config["muon_eta_min"] < muon.eta)
            & (muon.eta < self.config["muon_eta_max"])
            & (self.config["good_muon_pt_min"] < muon.pt))

        return muon[is_good]

    def lepton_pair(self, data):
        # We only want events with excatly two leptons, thus look at our
        # electron and muon counts and pick events accordingly
        return ak.num(data["Electron"]) + ak.num(data["Muon"]) == 2

    def opposite_sign_lepton_pair(self, data):
        # At this point we only have events with exactly two leptons, but now
        # we want only events where they have opposite charge

        # First concatenate the charge of our electron(s) and our muon(s)
        # into one array
        charge = ak.concatenate(
            [data["Electron"].charge, data["Muon"].charge], axis=1)

        # Now in this array we can simply compare the first and the second
        # element. Note that this is done on axis 1, axis 0 is always used for
        # event indexing, e.g. you would compare charges from event 0 and 1 if
        # you do charge[0] != charge[1]
        return charge[:, 0] != charge[:, 1]
