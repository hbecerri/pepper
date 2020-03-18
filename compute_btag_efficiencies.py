#!/usr/bin/env python3

import os
import numpy as np
import coffea
from argparse import ArgumentParser

from utils.config import Config
from utils.misc import get_event_files, montecarlo_iterate, export


def hist_divide(num, denom):
    """Return a histogram with bin heights = num / denum and errors set
    accordingly"""
    if not num.compatible(denom):
        raise ValueError("Cannot divide this histogram {} with histogram {} "
                         "of dissimilar dimensions".format(num, denom))
    hout = num.copy()
    hout.label = "Ratio"

    raxes = denom.sparse_axes()

    def div(a, b):
        out = np.zeros_like(a)
        nz = b != 0
        out[nz] = a[nz] / b[nz]
        return out

    def diverr2(a, b, da2, db2):
        out = np.zeros_like(a)
        nz = b != 0
        out[nz] = (da2[nz] * b[nz]**2 + db2[nz] * a[nz]**2) / b[nz]**4
        return out

    denomsumw2 = denom._sumw2 if denom._sumw2 is not None else denom._sumw
    for rkey in denom._sumw.keys():
        lkey = tuple(num.axis(rax).index(rax[ridx])
                     for rax, ridx in zip(raxes, rkey))
        if lkey in hout._sumw:
            hout._sumw2[lkey] = diverr2(hout._sumw[lkey],
                                        denom._sumw[rkey],
                                        hout._sumw2[lkey],
                                        denomsumw2[rkey])
            hout._sumw[lkey] = div(hout._sumw[lkey], denom._sumw[rkey])
        else:
            hout._sumw2[lkey] = np.zeros_like(denomsumw2[rkey])
            hout._sumw[lkey] = np.zeros_like(denom._sumw[rkey])
    return hout


def unpack_cuts(cutflags, num_cuts):
    byteview = cutflags[:, None].view(np.uint8)[:, ::-1]
    bits = np.unpackbits(byteview, axis=1)
    return bits[..., -num_cuts:].astype(bool)


parser = ArgumentParser()
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "eventdir", help="Path to the directory with the events. Event files need "
    "to be in subdirectories named by their dataset")
parser.add_argument(
    "cut", type=int, help="Cuts to check or ignore for efficiency evaluation. "
                          "The number is interpreted in base 2, where a 1 at "
                          "position n means that evens have to pass the n-th "
                          "cut while a 0 defines the corresponding cut to be "
                          "ignored.")
parser.add_argument("out", help="File name to write the resulting ROOT "
                                "histogram to")
parser.add_argument(
    "--eventext", default=".hdf5", help="File extension of the event files. "
    "Defaults to \".hdf5\"")
args = parser.parse_args()

config = Config(args.config)
num_cuts = args.cut.bit_length()
cut = unpack_cuts(np.array([args.cut], dtype=np.uint64), num_cuts)

mc_datasets = get_event_files(
    args.eventdir, args.eventext, config["mc_datasets"])
montecarlo_factors = config["mc_lumifactors"]

branches = ["Jet_hadronFlavour",
            "Jet_pt",
            "Jet_eta",
            "Jet_btag",
            "cutflags"]

flavbins = np.array([0, 4, 5, 6])
ptbins = (np.array([0, 20, 20, 30, 60, 60, 780]).cumsum()
          + config["btag_pt_min"])
etabins = np.array([0, 0.75, 1.5, 2.25, 3])
n_btag = coffea.hist.Hist("Counts",
                          coffea.hist.Bin("flav", "Flavor", flavbins),
                          coffea.hist.Bin("pt", "$p_{T}$", ptbins),
                          coffea.hist.Bin("abseta", r"$\left|\eta\right|$",
                                          etabins))
n_total = n_btag.copy(False)
for name, weight, data in montecarlo_iterate(mc_datasets,
                                             montecarlo_factors,
                                             branches,
                                             False):
    passing_cuts = (unpack_cuts(data["cutflags"], num_cuts)
                    | (~cut)).all(axis=1)
    data = data[passing_cuts]
    weight = weight[passing_cuts]
    print("{}: Considering {} events".format(name, passing_cuts.sum()))

    counts = data["Jet_hadronFlavour"].counts
    is_btagged = data["Jet_btag"]
    counts_btag = is_btagged.sum()
    n_total.fill(flav=data["Jet_hadronFlavour"].flatten(),
                 pt=data["Jet_pt"].flatten(),
                 abseta=abs(data["Jet_eta"].flatten()),
                 weight=np.repeat(weight, counts))
    n_btag.fill(flav=data["Jet_hadronFlavour"][is_btagged].flatten(),
                pt=data["Jet_pt"][is_btagged].flatten(),
                abseta=abs(data["Jet_eta"][is_btagged].flatten()),
                weight=np.repeat(weight, counts_btag))

efficiency = hist_divide(n_btag, n_total)
with uproot.recreate(args.out, compression=uproot.ZLIB(4)) as f:
    f["efficiency"] = export(efficiency)
