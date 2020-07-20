# This file illustrates how to implement a processor, realizing the selection
# steps and outputting histograms and a cutflow with efficiencies.
# Here we create a very simplified version of the ttbar-to-dilep processor.

import pepper
import numpy as np
import coffea
import coffea.lumi_tools
from functools import partial
import argparse
import parsl
import uproot
import awkward
import os
import logging


# All processors should inherid from pepper.Processor
class MyProcessor(pepper.Processor):
    def __init__(self, config):
        # Initialize the class, maybe overwrite some config variables and
        # load additional files if needed
        # Need to call parent init to make histograms and such ready
        super().__init__(config, None)

        self.lumimask = self.config["lumimask"]
        self.mc_lumifactors = config["mc_lumifactors"]
        self.trigger_paths = config["dataset_trigger_map"]
        self.trigger_order = config["dataset_trigger_order"]

    def process_selection(self, selector, dsname, is_mc, filler):
        # Implement the selection steps: add cuts, definine objects and/or
        # compute event weights

        # Add a cut only allowing events according to the golden JSON
        if not is_mc:
            selector.add_cut(partial(self.good_lumimask, is_mc), "Lumi")

        # Only allow events that pass triggers specified in config
        # This also takes into account a trigger order to avoid triggering
        # the same event if it's in two different data datasets.
        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.trigger_paths, self.trigger_order)
        selector.add_cut(partial(
            self.passing_trigger, pos_triggers, neg_triggers), "Trigger")

        # Define our first object: the leptons
        selector.set_column(self.build_lepton_column, "Lepton")

        # Only accept events that have to leptons
        selector.add_cut(self.lepton_pair, "Exactly 2 leptons")

        # Only accept events that have oppositly changed leptons
        selector.add_cut(self.opposite_sign_lepton_pair, "OC leptons")

    def scale_to_luminosity(self, selector, dsname):
        num_events = selector.num_selected
        lumifactors = self.mc_lumifactors
        factor = np.full(num_events, lumifactors[dsname])
        selector.modify_weight("lumi_factor", factor)

    def good_lumimask(self, is_mc, data):
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

    def build_lepton_column(self, data):
        # First find leptons that we think a good enough for our purposes
        good_electron = self.electron_cuts(data)
        good_muon = self.muon_cuts(data)

        keys = ["pt", "eta", "phi", "mass", "pdgId"]
        lep_dict = {}
        # Here we collect the observables we need for electrons and muons.
        # We are not intereseted in taus so skip them.
        for key in keys:
            electron_data = data["Electron_" + key][good_electron]
            muon_data = data["Muon_" + key][good_muon]
            lep_dict[key] = awkward.concatenate(
                [electron_data, muon_data], axis=1)

        # Create a JaggedCandidateArrays
        # It is made up from multiple jagged arrays and always
        # includes a Lorentz vector, just what we need for particles.
        # For documentation see coffea.analysis_objects.JaggedCandidateMethods
        leptons = pepper.misc.jcafromjagged(**lep_dict)

        # Sort leptons by pt
        leptons = leptons[leptons.pt.argsort()]
        return leptons

    def electron_cuts(self, data):
        # We do not want electrons that are between the barrel and the end cap
        sc_eta_abs = abs(data["Electron_eta"] + data["Electron_deltaEtaSC"])
        is_in_transreg = (1.444 < sc_eta_abs) & (sc_eta_abs < 1.566)

        # Electron ID, as an example we use the MVA one here
        has_id = data["Electron_mvaFall17V2Iso_WP90"]

        # Finally combine all the requirements
        return (has_id
                & (~is_in_transreg)
                & (self.config["ele_eta_min"] < data["Electron_eta"])
                & (data["Electron_eta"] < self.config["ele_eta_max"])
                & (self.config["good_ele_pt_min"] < data["Electron_pt"]))

    def muon_cuts(self, data):
        has_id = data["Muon_mediumId"]
        has_iso = data["Muon_pfIsoId"] > 1
        return (has_id
                & has_iso
                & (self.config["muon_eta_min"] < data["Muon_eta"])
                & (data["Muon_eta"] < self.config["muon_eta_max"])
                & (self.config["good_muon_pt_min"] < data["Muon_pt"]))

    def lepton_pair(self, data):
        return data["Lepton"].counts == 2

    def opposite_sign_lepton_pair(self, data):
        return (np.sign(data["Lepton"][:, 0].pdgId)
                != np.sign(data["Lepton"][:, 1].pdgId))


parser = argparse.ArgumentParser(description="Run MyProcessor")
parser.add_argument("config", help="Configuration file in JSON format")
parser.add_argument("--condor", action="store_true", help="Run on HTCondor")
args = parser.parse_args()

# Enable verbose output
logger = logging.getLogger("pepper")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

config = pepper.Config(args.config)
processor = MyProcessor(config)
datasets = config.get_datasets()

# In order for this to not take hours, only use one file and skip the rest
datasets = {
    next(iter(datasets.keys())): [datasets[next(iter(datasets.keys()))][0]]}

if args.condor:
    executor = coffea.processor.parsl_executor
    parsl_config = pepper.misc.get_parsl_config(num_jobs=10)
    parsl.load(parsl_config)
else:
    executor = coffea.processor.iterative_executor

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor)

# Save cutflows
coffea.util.save(output["cutflows"], "cutflows.coffea")

# Save histograms
os.makedirs("hists", exist_ok=True)
for cut, hist in output["hists"].items():
    fname = '_'.join(cut).replace('/', '_') + ".root"
    roothists = pepper.misc.export_with_sparse(hist)
    if len(roothists) == 0:
        continue
    with uproot.recreate(os.path.join("hists", fname)) as f:
        for key, subhist in roothists.items():
            key = "_".join(key).replace("/", "_")
            f[key] = subhist
