#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser

import uproot
import coffea.hist

from pepper.misc import HistCollection


parser = ArgumentParser(
    description="Generate a ROOT file containing efficiency histograms needed "
    "for computing jet pile-up ID scale factors. This requires a specific "
    "histogram to have been computed, see the jet_pu_id_eff histogram in "
    "example/hist_config.json")
parser.add_argument("histsfile", help="A JSON file specifying the histograms, "
                                      "e.g. 'hists.json'")
parser.add_argument("output", help="Output ROOT file")
parser.add_argument(
    "--cut", default="Z window", help="Name of the cut before the jet PU ID "
                                      "requirement. (Default 'Z window')")
parser.add_argument(
    "--histname", default="jet_pu_id_eff", help="Name of the jet PU ID "
    "efficiency histogram. (Default 'jet_pu_id_eff')")
args = parser.parse_args()

if os.path.exists(args.output):
    if input(f"{args.output} exists. Overwrite [y/n] ") != "y":
        sys.exit(1)

with open(args.histsfile) as f:
    hists = HistCollection.from_json(f)

with uproot.recreate(args.output) as f:
    hist = hists.load(
        {"cut": args.cut, "hist": args.histname, "variation": None})
    if isinstance(hist, coffea.hist.Hist):
        hist = hist.to_hist()
    hist = hist.project("pt", "eta", "has_gen_jet", "pass_pu_id")
    eff_hist = hist[{"has_gen_jet": "yes"}]
    f["eff"] = (eff_hist[{"pass_pu_id": "yes"}]
                / eff_hist[{"pass_pu_id": sum}].values())
    mis_hist = hist[{"has_gen_jet": "no"}]
    f["mis"] = (mis_hist[{"pass_pu_id": "yes"}]
                / mis_hist[{"pass_pu_id": sum}].values())
