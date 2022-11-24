#!/usr/bin/env python3

import os
from argparse import ArgumentParser
import json
from collections import defaultdict
import awkward as ak
from tqdm import tqdm
import parsl
from parsl import python_app
import concurrent.futures

import pepper


def get_counts(path, geskey, lhesskey, lhepdfskey):
    import awkward as ak
    import uproot
    import pepper
    with uproot.open(path, timeout=pepper.misc.XROOTDTIMEOUT) as f:
        runs = f["Runs"]
        gen_event_sumw = runs[geskey].array()[0]
        lhe_scale_sumw = runs[lhesskey].array()[0]
        lhe_pdf_sumw = ak.Array([])
        has_lhe = len(lhe_scale_sumw) != 0
        if has_lhe:
            lhe_scale_sumw = lhe_scale_sumw * gen_event_sumw
            if lhepdfskey is not None:
                lhe_pdf_sumw = runs[lhepdfskey].array()[0] * gen_event_sumw
    return path, gen_event_sumw, lhe_scale_sumw, lhe_pdf_sumw


def add_counts(counts, datasets, lhepdfskey, result):
    path, gen_event_sumw, lhe_scale_sumw, lhe_pdf_sumw = result
    process_name = datasets[path]
    has_lhe = len(lhe_scale_sumw) != 0
    counts[process_name] += gen_event_sumw
    if has_lhe:
        counts[process_name + "_LHEScaleSumw"] =\
            counts[process_name + "_LHEScaleSumw"] + lhe_scale_sumw
        if lhepdfskey is not None:
            counts[process_name + "_LHEPdfSumw"] =\
                counts[process_name + "_LHEPdfSumw"] + lhe_pdf_sumw


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
parser.add_argument(
    "-c", "--condor", type=int, metavar="simul_jobs",
    help="Number of HTCondor jobs to launch")
parser.add_argument(
    "-r", "--retries", type=int, help="Number of times to retry if there is "
    "exception in an HTCondor job. If not given, retry infinitely."
)
parser.add_argument(
    "-i", "--condorinit",
    help="Shell script that will be sourced by an HTCondor job after "
    "starting. This can be used to setup environment variables, if using "
    "for example CMSSW. If not provided, the local content of the "
    "environment variable PEPPER_CONDOR_ENV will be used as path to the "
    "script instead.")
parser.add_argument(
    "--condorsubmit",
    help="Text file containing additional parameters to put into the "
    "HTCondor job submission file that is used for condor_submit"
)
args = parser.parse_args()

if args.condorinit is not None:
    with open(args.condorinit) as f:
        condorinit = f.read()
else:
    condorinit = None
if args.condorsubmit is not None:
    with open(args.condorsubmit) as f:
        condorsubmit = f.read()
else:
    condorsubmit = None

if args.skip and os.path.exists(args.out):
    with open(args.out) as f:
        factors = json.load(f)
else:
    factors = {}

lhepdfskey = "LHEPdfSumw" if args.pdfsumw else None

config = pepper.ConfigBasicPhysics(args.config)
lumi = config["luminosity"]
crosssections = config["crosssections"]
procs, datasets = config.get_datasets(
    dstype="mc", return_inverse=True, exclude=factors.keys())
if "dataset_for_systematics" in config:
    dsforsys = config["dataset_for_systematics"]
else:
    dsforsys = {}
for process_name in procs.keys():
    if process_name not in crosssections and process_name not in dsforsys:
        raise ValueError(f"Could not find crosssection for {process_name}")

counts = defaultdict(int)

if args.condor is None:
    for filepath, process_name in tqdm(datasets.items()):
        result = get_counts(filepath, "genEventSumw", "LHEScaleSumw",
                            lhepdfskey)
        add_counts(counts, datasets, lhepdfskey, result)
else:
    get_counts = python_app(get_counts)
    parsl_config = pepper.misc.get_parsl_config(
        args.condor,
        condor_submit=condorsubmit,
        condor_init=condorinit,
        retries=args.retries)
    parsl.load(parsl_config)

    futures = set()
    for filepath, process_name in datasets.items():
        futures.add(get_counts(filepath, "genEventSumw", "LHEScaleSumw",
                               lhepdfskey))
    for future in tqdm(concurrent.futures.as_completed(futures),
                       total=len(futures)):
        add_counts(counts, datasets, lhepdfskey, future.result())

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
