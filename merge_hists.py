import uproot
from argparse import ArgumentParser


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
numerator = None
for in_hist in args.input:
    with uproot.open(in_hist[1]) as f:
        hist = f[in_hist[2]].to_hist()
    weight = int(in_hist[0])
    denom += weight
    if numerator is None:
        numerator = hist * weight
    else:
        numerator += hist * weight

with uproot.recreate(args.out_file) as out_file:
    out_file[args.name] = numerator / denom
