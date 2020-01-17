import os
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
import numpy as np

from utils.config import Config
from utils.misc import get_event_files, montecarlo_iterate, expdata_iterate


def get_channel_masks(data):
    p0 = abs(data["Lepton_pdgId"][:, 0])
    p1 = abs(data["Lepton_pdgId"][:, 1])
    return {
        "ee": (p0 == 11) & (p1 == 11),
        "mm": (p0 == 13) & (p1 == 13),
        "em": p0 != p1,
    }


parser = ArgumentParser()
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "eventdir", help="Path to the directory with the events. Event files need "
    "to be in subdirectories named by their dataset")
parser.add_argument(
    "--eventext", default=".hdf5", help="File extension of the event files. "
    "Defaults to \".hdf5\"")
parser.add_argument(
    "cuts", type=int, help="Only events that have the given number as "
    "their entry in the cut_arrays are considered. This must not include"
    "the Z window cut.")
args = parser.parse_args()

config = Config(args.config)

exp_datasets = get_event_files(
    args.eventdir, args.eventext, config["exp_datasets"])
mc_datasets = get_event_files(
    args.eventdir, args.eventext, config["mc_datasets"])
montecarlo_factors = config["mc_lumifactors"]
m_min = config["z_boson_window_start"]
m_max = config["z_boson_window_end"]
branches = [
    "Lepton_pdgId",
    "cutflags",
    "mll",
]

counts_in_mc = defaultdict(float)
counts_out_mc = defaultdict(float)
for name, weight, data in montecarlo_iterate(mc_datasets,
                                             montecarlo_factors,
                                             branches):
    pass_cuts = (data["cutflags"] & args.cuts) == args.cuts
    in_zwindow = (m_min < data["mll"]) & (data["mll"] < m_max)
    masks = get_channel_masks(data)

    print("{}: {} events passing cuts, {} total".format(name,
                                                        pass_cuts.sum(),
                                                        data.size))

    for chan, mask in masks.items():
        consider = mask & pass_cuts
        n_in = weight[consider & in_zwindow].sum()
        n_out = weight[consider & ~in_zwindow].sum()
        counts_in_mc[chan] += n_in
        counts_out_mc[chan] += n_out
        if name.startswith("DYJetsTo"):
            counts_in_mc["DY_" + chan] += n_in
            counts_out_mc["DY_" + chan] += n_out

print("Counts_in_mc = \\")
pprint(dict(counts_in_mc))
print("Counts_out_mc = \\")
pprint(dict(counts_out_mc))

counts_in_data = defaultdict(int)
counts_out_data = defaultdict(int)
for name, data in expdata_iterate(exp_datasets, branches):
    pass_cuts = (data["cutflags"] & args.cuts) == args.cuts
    in_zwindow = (m_min < data["mll"]) & (data["mll"] < m_max)
    masks = get_channel_masks(data)

    print("{}: {} events passing cuts, {} total".format(name,
                                                        pass_cuts.sum(),
                                                        data.size))

    for chan, mask in masks.items():
        consider = mask & pass_cuts
        counts_in_data[chan] += (consider & in_zwindow).sum()
        counts_out_data[chan] += (consider & ~in_zwindow).sum()

print("Counts_in_data = \\")
pprint(dict(counts_in_data))
print("Counts_out_data = \\")
pprint(dict(counts_out_data))

k_ee_mc = np.sqrt(counts_in_mc["ee"] / counts_in_mc["mm"])
k_mm_mc = 1 / k_ee_mc
k_ee_data = np.sqrt(counts_in_data["ee"] / counts_in_data["mm"])
k_mm_data = 1 / k_ee_data
print(f"k_ee_mc={k_ee_mc:.2f}; k_ee_data={k_ee_data:.2f}")
print(f"k_mm_mc={k_mm_mc:.2f}; k_mm_data={k_mm_data:.2f}")

routin_ee = counts_out_mc["DY_ee"] / counts_in_mc["DY_ee"]
routin_mm = counts_out_mc["DY_mm"] / counts_in_mc["DY_mm"]
print(f"routin_ee={routin_ee:.2f}")
print(f"routin_mm={routin_mm:.2f}")

diff_ee_mc = counts_in_mc["ee"] - 0.5 * counts_in_mc["em"] * k_ee_mc
diff_mm_mc = counts_in_mc["mm"] - 0.5 * counts_in_mc["em"] * k_mm_mc
diff_ee_data = counts_in_data["ee"] - 0.5 * counts_in_data["em"] * k_ee_data
diff_mm_data = counts_in_data["mm"] - 0.5 * counts_in_data["em"] * k_mm_data
print(f"diff_ee_mc={diff_ee_mc:.2f}; diff_ee_data={diff_ee_data:.2f}")
print(f"diff_mm_mc={diff_mm_mc:.2f}; diff_mm_data={diff_mm_data:.2f}")

nesti_ee_mc = routin_ee * diff_ee_mc
nesti_mm_mc = routin_mm * diff_mm_mc
nesti_ee_data = routin_ee * diff_ee_data
nesti_mm_data = routin_mm * diff_mm_data
print(f"nesti_ee_mc={nesti_ee_mc:.2f}; nesti_ee_data={nesti_ee_data:.2f}")
print(f"nesti_mm_mc={nesti_mm_mc:.2f}; nesti_mm_data={nesti_mm_data:.2f}")

sf_ee = nesti_ee_data / nesti_ee_mc
sf_mm = nesti_mm_data / nesti_mm_mc
sf_em = np.sqrt(sf_ee * sf_mm)
print(f"sf_ee={sf_ee}")
print(f"sf_mm={sf_mm}")
print(f"sf_em={sf_em}")
