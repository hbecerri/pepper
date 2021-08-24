import numpy as np
import uproot3
from uproot3_methods.classes.TH1 import from_numpy as th1_from_numpy
from uproot3_methods.classes.TH2 import from_numpy as th2_from_numpy
from argparse import ArgumentParser


class TAxis(object):
    def __init__(self, edges, label):
        self._fNbins = len(edges) - 1
        self._fXmin = edges[0]
        self._fXmax = edges[-1]
        self._fXbins = edges
        self._fName = label
        self._fTitle = label


def create_root_hist(hist, sumw2, edges):
    if len(edges) == 1:
        out = th1_from_numpy([hist, list(edges)])
        sw2 = np.zeros(len(hist) + 2, dtype=sumw2.dtype)
        sw2[1:-1] = sumw2
        out._fSumw2 = (sw2).astype(">f8")
    elif len(edges) == 2:
        out = th2_from_numpy([hist, list(edges)])
        out._fSumw2 = np.pad((sumw2).astype(">f8").T, (1, 1),
                             mode='constant').flatten()
    return out


parser = ArgumentParser(
    description="Script to caluclate weighted average of two SF histograms, "
    "for instance for 2016, where the muon SFs differ before and after era G")
parser.add_argument("out_file", help="Path to file to save output histogram "
                    "in. NB: This file will be overwritten")
parser.add_argument("name", help="Name of ouput histogram")
parser.add_argument(
    "-i", "--input", nargs=3, metavar=("weight", "path", "histname"),
    action="append", help="Histograms to merge, specify once per histogram."
    " Expects three values: the weight for this histogram, the path to the "
    "ROOT file containing the histogram, and the name of this histogram")
args = parser.parse_args()

denom = 0
Numerator = None
for in_hist in args.input:
    with uproot3.open(in_hist[1]) as f:
        hist = f[in_hist[2]]
    edges = hist.edges
    dimlabels = []
    for member in ("_fXaxis", "_fYaxis", "_fZaxis")[:len(edges)]:
        dimlabels.append(getattr(getattr(hist, member), "_fName"))
    factors = hist.values
    sigmas = np.sqrt(hist.variances)
    if len(edges) != len(dimlabels):
        raise ValueError("Got {} dimenions but {} labels"
                         .format(len(edges), len(dimlabels)))
    weight = int(in_hist[0])
    denom += weight
    if Numerator is None:
        Numerator = factors * weight
        sigma_num = sigmas * weight
        dimlabels_orig = dimlabels
        edges_orig = edges
    elif dimlabels != dimlabels_orig:
        # Still need to check edges
        raise ValueError("All histograms must have same lables and edges!")
    else:
        Numerator += factors * weight
        sigma_num += sigmas * weight

out_file = uproot3.recreate(args.out_file)
out_file[args.name] = create_root_hist(Numerator / denom,
                                       (sigma_num / denom) ** 2, edges)
out_file.close()
