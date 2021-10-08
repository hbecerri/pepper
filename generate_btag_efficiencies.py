#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser

import uproot

import pepper
from pepper.misc import HistCollection


parser = ArgumentParser(
    description="Generate a ROOT file containing efficiency histograms needed "
    "for computing b-tagging scale factors. This requires a specific "
    "histogram to have been computed, see the btageff histogram in "
    "example/hist_config.json")
parser.add_argument("histsfile", help="A JSON file specifying the histograms, "
                                      "e.g. 'hists.json'")
parser.add_argument("output", help="Output ROOT file")
parser.add_argument(
    "--cut", default="Has jet(s)", help="Name of the cut before the b-tag "
                                        "requirement. (Default 'Has jet(s)')")
parser.add_argument(
    "--histname", default="btageff", help="Name of the b-tagging efficiency "
                                          "histogram. (Default 'btageff')")
parser.add_argument(
    "--central", action="store_true",
    help="Only output the central efficiency and no additional ones for "
         "systematic variations")
args = parser.parse_args()

if os.path.exists(args.output):
    if input(f"{args.output} exists. Overwrite [y/n] ") != "y":
        sys.exit(1)

with open(args.histsfile) as f:
    hists = HistCollection.from_json(f)

with uproot.recreate(args.output) as f:
    for key, histpath in hists[dict(cut=args.cut, hist=args.histname)].items():
        hist = pepper.misc.coffeahist2hist(hists.load(key))
        hist = hist[{"dataset": sum, "channel": sum}]
        eff = hist[{"btagged": "yes"}] / hist[{"btagged": sum}].values()
        if key.variation is None:
            f["central"] = eff
        elif args.central:
            continue
        elif any(key.variation.endswith(x) for x in (
                "XS_down", "XS_up", "lumi_down", "lumi_up")):
            # These aren't shape uncertainties and also do not have much effect
            # on the efficiency
            continue
        else:
            f[key[2]] = eff
