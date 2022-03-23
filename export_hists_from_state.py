#!/usr/bin/env python3

import os
import pepper
import coffea.util
from argparse import ArgumentParser


parser = ArgumentParser(
    description="Save all histograms contained in a Pepper processor state, "
    "even if processing of all data hasn't been finished yet")
parser.add_argument(
    "statedata", help="Path to the state file, for example "
    "'pepper_state.coffea'")
parser.add_argument(
    "format", choices=["hist", "root", "coffea"],
    help="Histogram output format")
parser.add_argument("outputdir", help="Output directory for the histograms")
args = parser.parse_args()

os.makedirs(args.outputdir, exist_ok=True)

state = coffea.util.load(args.statedata)
output = state["accumulator"]["out"]
pepper.Processor.postprocess(output)
pepper.Processor.save_histograms(args.format, output, args.outputdir)
