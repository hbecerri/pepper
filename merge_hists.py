import numpy as np
import uproot3
from uproot3_methods.classes.TH1 import Methods as TH1Methods
from uproot3_methods.classes.TH2 import Methods as TH2Methods
from uproot3_methods.classes.TH3 import Methods as TH3Methods
from argparse import ArgumentParser


class TH1(TH1Methods, list):
    pass


class TH2(TH2Methods, list):
    pass


class TH3(TH3Methods, list):
    pass


class TAxis(object):
    def __init__(self, edges, label):
        self._fNbins = len(edges) - 1
        self._fXmin = edges[0]
        self._fXmax = edges[-1]
        self._fXbins = edges
        self._fName = label
        self._fTitle = label


def create_root_hist(hist, sumw2, edges, labels, name):
    if len(edges) == 1:
        out_hist = TH1.__new__(TH1)
        out_hist._fXaxis = TAxis(edges[0], labels[0])
        out_hist._classname = b"TH1D"
    elif len(edges) == 2:
        out_hist = TH2.__new__(TH2)
        out_hist._fXaxis = TAxis(edges[0], labels[0])
        out_hist._fYaxis = TAxis(edges[1], labels[1])
        out_hist._classname = b"TH2D"
    elif len(edges) == 3:
        out_hist = TH3.__new__(TH3)
        out_hist._fXaxis = TAxis(edges[0], labels[0])
        out_hist._fYaxis = TAxis(edges[1], labels[1])
        out_hist._fZaxis = TAxis(edges[2], labels[2])
        out_hist._classname = b"TH3D"

    out_hist._fName = name
    out_hist._fTitle = name
    out_hist.extend((hist).astype(">f8"))
    out_hist._fSumw2 = (sumw2).astype(">f8")
    return out_hist


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
    factors = hist.allvalues
    sigmas = np.sqrt(hist.allvariances)
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
out_file[args.name] = create_root_hist(
    Numerator / denom, (sigma_num / denom) ** 2, edges, dimlabels, args.name)
out_file.close()
