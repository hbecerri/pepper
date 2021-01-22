# This file illustrates how to implement a processor, realizing the selection
# steps and outputting histograms and a cutflow with efficiencies.
# Here we create a very simplified version of the ttbar-to-dilep processor.

import pepper
import numpy as np
import awkward as ak
import coffea
import coffea.lumi_tools
from coffea.nanoevents import NanoAODSchema
from functools import partial
import argparse
import parsl
import awkward
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import uproot3  # noqa: E402


# All processors should inherit from pepper.Processor
class MyProcessor(pepper.Processor):
    def __init__(self, config):
        # Initialize the class, maybe overwrite some config variables and
        # load additional files if needed
        # Need to call parent init to make histograms and such ready
        super().__init__(config, None)

        self.lumimask = self.config["lumimask"]
        self.trigger_paths = config["dataset_trigger_map"]
        self.trigger_order = config["dataset_trigger_order"]

    def process_selection(self, selector, dsname, is_mc, filler):
        # Implement the selection steps: add cuts, define objects and/or
        # compute event weights

        # Add a cut only allowing events according to the golden JSON
        if not is_mc:
            selector.add_cut("Lumi", partial(self.good_lumimask, is_mc))

        # Only allow events that pass triggers specified in config
        # This also takes into account a trigger order to avoid triggering
        # the same event if it's in two different data datasets.
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.trigger_paths, self.trigger_order)
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

    def good_lumimask(self, is_mc, data):
        # This function gets called by the selector in order to get information
        # on which events to discard or how to modify the event weight.
        # The data argument is given by the selector and is a nanoevent object,
        # giving us access to all info inside the Events tree of the NanoAOD
        run = np.array(data["run"])
        luminosity_block = np.array(data["luminosityBlock"])
        lumimask = coffea.lumi_tools.LumiMask(self.lumimask)
        return lumimask(run, luminosity_block)

    def passing_trigger(self, pos_triggers, neg_triggers, data):
        hlt = data["HLT"]
        trigger = (
            np.any([hlt[trigger_path] for trigger_path in pos_triggers],
                   axis=0)
            & ~np.any([hlt[trigger_path] for trigger_path in neg_triggers],
                      axis=0)
        )
        return trigger

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


parser = argparse.ArgumentParser(description="Run MyProcessor")
parser.add_argument("config", help="Configuration file in JSON format")
parser.add_argument("--condor", action="store_true", help="Run on HTCondor")
args = parser.parse_args()

# Enable verbose output
logger = logging.getLogger("pepper")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

# Here we use the ConfigTTbarLL instead of its base Config, to use some of its
# predefined extras
config = pepper.ConfigTTbarLL(args.config)
processor = MyProcessor(config)
datasets = config.get_datasets()

# In order for this to not take hours, only use one file and skip the rest
datasets = {
    next(iter(datasets.keys())): [datasets[next(iter(datasets.keys()))][0]]}

if args.condor:
    executor = coffea.processor.parsl_executor
    parsl_config = pepper.misc.get_parsl_config(num_jobs=10, retries=1)
    parsl.load(parsl_config)
else:
    executor = coffea.processor.iterative_executor

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, {"schema": NanoAODSchema})

# Save cutflows
coffea.util.save(output["cutflows"], "cutflows.coffea")

# Save histograms
os.makedirs("hists", exist_ok=True)
for cut, hist in output["hists"].items():
    fname = '_'.join(cut).replace('/', '_') + ".root"
    roothists = pepper.misc.export_with_sparse(hist)
    if len(roothists) == 0:
        continue
    with uproot3.recreate(os.path.join("hists", fname)) as f:
        for key, subhist in roothists.items():
            key = "_".join(key).replace("/", "_")
            f[key] = subhist
