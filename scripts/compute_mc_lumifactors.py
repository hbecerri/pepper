#!/usr/bin/env python3

import os
from argparse import ArgumentParser
import json
from collections import defaultdict
import awkward as ak
import uproot
from itertools import chain
from tqdm import tqdm

import pepper


def update_counts(f, counts, process_name, geskey, lhesskey, lhepdfskey):
    runs = f["Runs"]
    gen_event_sumw = runs[geskey].array()[0]
    lhe_scale_sumw = runs[lhesskey].array()[0]
    has_lhe = len(lhe_scale_sumw) != 0
    if has_lhe:
        lhe_scale_sumw = lhe_scale_sumw * gen_event_sumw
        if lhepdfskey is not None:
            lhe_pdf_sumw = runs[lhepdfskey].array()[0] * gen_event_sumw

    counts[process_name] += gen_event_sumw
    if has_lhe:
        counts[process_name + "_LHEScaleSumw"] =\
            counts[process_name + "_LHEScaleSumw"] + lhe_scale_sumw
        if lhepdfskey is not None:
            counts[process_name + "_LHEPdfSumw"] =\
                counts[process_name + "_LHEPdfSumw"] + lhe_pdf_sumw
    return counts


parser = ArgumentParser(description="Compute factors from luminosity and "
                                    "cross sections to scale MC")
parser.add_argument(
    "config", help="Path to the JSON config file containing the MC dataset "
    "names, luminosity and cross sections. Latter two should be in 1/fb and "
    "fb respectively")
parser.add_argument("out", help="Path to the output file")
parser.add_argument(
    "-s", "--skip", action="store_true",
    help="If out exists, skip the data sets that are already in the output "
         "file")
parser.add_argument(
    "-p", "--pdfsumw", action="store_true", help="Add PdfSumw to the output "
    "file")
args = parser.parse_args()

if args.skip and os.path.exists(args.out):
    with open(args.out) as f:
        factors = json.load(f)
else:
    factors = {}

config = pepper.ConfigBasicPhysics(args.config)
lumi = config["luminosity"]
crosssections = config["crosssections"]
store, datasets = config[["store", "mc_datasets"]]
datasets = {k: v for k, v in datasets.items()
            if v is not None and k not in factors}
datasets = pepper.datasets.expand_datasetdict(datasets, store)[0]
if "dataset_for_systematics" in config:
    dsforsys = config["dataset_for_systematics"]
else:
    dsforsys = {}
counts = defaultdict(int)
num_files = len(list(chain(*datasets.values())))
i = 0
for process_name, proc_datasets in tqdm(datasets.items(), desc="Total"):
    if process_name not in crosssections and process_name not in dsforsys:
        print(f"Could not find crosssection for {process_name}")
        continue
    for path in tqdm(proc_datasets, desc="Per dataset"):
        with uproot.open(path) as f:
            counts = update_counts(
                f, counts, process_name, "genEventSumw", "LHEScaleSumw",
                "LHEPdfSumw" if args.pdfsumw else None)
        i += 1
    print("\033[F\033[F")  # Workaround for tqdm adding a new line
print("")
for key in counts.keys():
    if key.endswith("_LHEScaleSumw") or key.endswith("_LHEPdfSumw"):
        dsname = key.rsplit("_", 1)[0]
    else:
        dsname = key
    if dsname in dsforsys:
        xs = crosssections[dsforsys[dsname][0]]
    else:
        xs = crosssections[dsname]
    factor = xs * lumi / counts[key]
    if key.endswith("_LHEScaleSumw") or key.endswith("_LHEPdfSumw"):
        factor = counts[dsname] / counts[key]
    if isinstance(factor, ak.Array):
        factor = list(factor)
    factors[key] = factor
    if key == dsname:
        print(f"{key}: {xs} fb, {counts[key]} events, factor of "
              f"{factors[key]:.3e}")

with open(args.out, "w") as f:
    json.dump(factors, f, indent=4)
