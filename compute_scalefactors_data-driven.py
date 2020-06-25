import os
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
import numpy as np
from glob import glob
import h5py
import awkward
from tqdm import tqdm

import pepper


def iterate(paths):
    for path in paths:
        with h5py.File(path, "r") as f:
            a = awkward.hdf5(f)
            weight = a["weight"]
            data = a["events"]
            cutflags = a["cutflags"]
            cutnames = a["cutnames"]
        if "mll" in data.columns:
            mll = data["mll"]
        else:
            mll = data["Lepton"].sum().mass
        if weight is None:
            yield data, cutflags, cutnames, mll
        else:
            yield weight, data, cutflags, cutnames, mll


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
args = parser.parse_args()

config = pepper.Config(args.config)

exp_datasets = [p for ds in config["exp_datasets"].keys()
                for p in glob(os.path.join(args.eventdir, ds, "*.hdf5"))]
mc_datasets = [p for ds in config["mc_datasets"].keys()
               if ds.startswith("DYJets")
               for p in glob(os.path.join(args.eventdir, ds, "*.hdf5"))]
m_min = config["z_boson_window_start"]
m_max = config["z_boson_window_end"]

counts_in_dymc = defaultdict(float)
counts_out_dymc = defaultdict(float)
for weight, data, cutflags, cutnames, mll in tqdm(iterate(mc_datasets),
                                                  total=len(mc_datasets)):
    zcutindex = cutnames.index("Z window")
    pass_cuts = cutflags & ((1 << zcutindex) - 1) == ((1 << zcutindex) - 1)
    in_zwindow = (m_min < mll) & (mll < m_max)
    for chan, mask in get_channel_masks(data).items():
        consider = mask & pass_cuts
        counts_in_dymc[chan] += weight[consider & in_zwindow].sum()
        counts_out_dymc[chan] += weight[consider & ~in_zwindow].sum()
print("counts_in_dymc = \\")
pprint(dict(counts_in_dymc))
print("counts_out_dymc = \\")
pprint(dict(counts_out_dymc))

counts_in_data = defaultdict(int)
counts_out_data = defaultdict(int)
for data, cutflags, cutnames, mll in tqdm(iterate(exp_datasets),
                                          total=len(exp_datasets)):
    zcutindex = cutnames.index("Z window")
    pass_cuts = cutflags & ((1 << zcutindex) - 1) == ((1 << zcutindex) - 1)
    in_zwindow = (m_min < mll) & (mll < m_max)
    for chan, mask in get_channel_masks(data).items():
        consider = mask & pass_cuts
        counts_in_data[chan] += (consider & in_zwindow).sum()
        counts_out_data[chan] += (consider & ~in_zwindow).sum()
print("counts_in_data = \\")
pprint(dict(counts_in_data))
print("Counts_out_data = \\")
pprint(dict(counts_out_data))

k_ee_data = np.sqrt(counts_in_data["ee"] / counts_in_data["mm"])
k_mm_data = 1 / k_ee_data
print(f"k_ee_data={k_ee_data:.2f}; k_mm_data={k_mm_data:.2f}")

routin_ee = counts_out_dymc["ee"] / counts_in_dymc["ee"]
routin_mm = counts_out_dymc["mm"] / counts_in_dymc["mm"]
print(f"routin_ee={routin_ee:.2f}; routin_mm={routin_mm:.2f}")

diff_ee_data = counts_in_data["ee"] - 0.5 * counts_in_data["em"] * k_ee_data
diff_mm_data = counts_in_data["mm"] - 0.5 * counts_in_data["em"] * k_mm_data
print(f"diff_ee_data={diff_ee_data:.2f}; diff_mm_data={diff_mm_data:.2f}")

nesti_ee_data = routin_ee * diff_ee_data
nesti_mm_data = routin_mm * diff_mm_data
print(f"nesti_ee_data={nesti_ee_data:.2f}; nesti_mm_data={nesti_mm_data:.2f}")

sf_ee = nesti_ee_data / counts_out_dymc["ee"]
sf_mm = nesti_mm_data / counts_out_dymc["mm"]
sf_em = np.sqrt(sf_ee * sf_mm)
print(f"sf_ee={sf_ee}")
print(f"sf_mm={sf_mm}")
print(f"sf_em={sf_em}")
