import os
import json
from argparse import ArgumentParser
import copy
import logging

import uproot
import coffea
import numpy as np

import pepper
from pepper.misc import HistCollection


def safe_div(num, denom):
    return num / np.where(num > 0, denom, np.inf)


def hist_div(num, denom):
    ret = copy.copy(num)
    vals = safe_div(num.values(), denom.values())
    variances = safe_div(num.variances() * denom.values() ** 2
                         + denom.variances() * num.values() ** 2,
                         denom.values() ** 4)
    ret[:, :, :] = np.stack([vals, variances], axis=-1)
    return ret


def calculate_sf(data_hist, mc_hist):
    data_eff = hist_div(data_hist[{"dilep triggers": "yes"}],
                        data_hist[{"dilep triggers": sum}])
    mc_eff = hist_div(mc_hist[{"dilep triggers": "yes"}],
                      mc_hist[{"dilep triggers": sum}])
    return hist_div(data_eff, mc_eff)


parser = ArgumentParser(
    description="")
parser.add_argument("config", help="Json config")
parser.add_argument("histsfile", help="A JSON file specifying the histograms, "
                                      "e.g. 'hists.json'")
parser.add_argument("output", help="Output ROOT file")
parser.add_argument(
    "--cut", default="Req MET", help="Name of the cut after which to calculate"
    "the trigger SFs. (Default 'Req MET')")
parser.add_argument(
    "--histname", default="trigger_sf_constructor", help="Name of the trigger "
    "SF efficiency histogram. (Default 'trigger_sf_constructor')")
parser.add_argument("-v", "--verbose", action="store_true", help="Print SFs "
                    "and fractional size of all uncertainties")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
if args.verbose:
    logger.setLevel(logging.DEBUG)

with open(args.config) as f:
    config = json.load(f)

with open(args.histsfile) as f:
    hists = HistCollection.from_json(f)
hist = pepper.misc.coffeahist2hist(coffea.util.load(os.path.join(
    hists._path,
    hists[{"cut": args.cut, "hist": args.histname, "variation": None}])))

sel = {"njet": sum, "nPV": sum, "MET": sum, "MET triggers": "yes"}
data_sel = {"dataset": list(config["MET_trigger_datasets"].keys())}
mc_sel = {"dataset": "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"}
data_hist = hist[sel][data_sel][{"dataset": sum}]
mc_hist = hist[sel][mc_sel]
sf = calculate_sf(data_hist, mc_hist)
sf_variance = sf.variances()

logger.debug(f"SF values: {sf.values()}")
logger.debug(f"Stat. unc.: {safe_div(sf_variance ** 0.5, sf.values())}")

# Independent region systematics:
for ax in ["njet", "nPV", "MET"]:
    sel[ax] = 0
    data_hist_down = hist[sel][data_sel][{"dataset": sum}]
    mc_hist_down = hist[sel][mc_sel]
    sel[ax] = 1
    data_hist_up = hist[sel][data_sel][{"dataset": sum}]
    mc_hist_up = hist[sel][mc_sel]
    sel[ax] = sum

    sf_down = calculate_sf(data_hist_down, mc_hist_down)
    sf_up = calculate_sf(data_hist_up, mc_hist_up)
    diff = np.maximum(abs(sf.values() - sf_down.values()),
                      abs(sf.values() - sf_up.values()))
    sf_variance += diff ** 2
    logger.debug(f"Unc. for {ax} variation: {safe_div(diff, sf.values())}")

# Era-by-era systematics:
diffs = np.empty([len(config["MET_trigger_datasets"]), *sf.shape])
for i, era in enumerate(config["MET_trigger_datasets"].keys()):
    data_hist = hist[sel][{"dataset": era}]
    era_sf = calculate_sf(data_hist, mc_hist)
    diffs[i] = abs(sf.values() - era_sf.values())
    logger.debug(f"Unc. for era {era}: {safe_div(diffs[i], sf.values())}")
sf_variance += np.amax(diffs, axis=0) ** 2

# Trigger correlation systematics:
sel = {"njet": sum, "nPV": sum, "MET": sum}
mc_hist = hist[sel][mc_sel]
denom = mc_hist[{"dilep triggers": sum, "MET triggers": sum}].values()
eff_mt_only = safe_div(
    mc_hist[{"dilep triggers": sum, "MET triggers": "yes"}].values(), denom)
eff_dlt_only = safe_div(
    mc_hist[{"dilep triggers": "yes", "MET triggers": sum}].values(), denom)
eff_both = safe_div(
    mc_hist[{"dilep triggers": "yes", "MET triggers": "yes"}].values(), denom)
alpha = safe_div(eff_mt_only * eff_dlt_only, eff_both)
sf_variance += ((1 - alpha) * sf.values()) ** 2
logger.debug(f"Trigger correlation unc.: {np.where(sf.values()>0, 1 - alpha, 0)}")

sf[:, :, :] = np.stack([sf.values(), sf_variance], axis=-1)
logger.debug(
    f"Total uncertainty: {safe_div(sf.variances() ** 0.5, sf.values())}")

with uproot.recreate(args.output) as f:
    for ch in ["is_ee", "is_em", "is_mm"]:
        f["SF_" + ch + "_pt1_pt2"] = sf[{"channel": ch}]
