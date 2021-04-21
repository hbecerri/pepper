import json
from argparse import ArgumentParser


parser = ArgumentParser(
    description="Convert c++ header files for MET-xy corrections "
    "(as available from https://lathomas.web.cern.ch/lathomas/"
    "METStuff/XYCorrections) to json format used by pepper")
parser.add_argument(
    "year", help="Year to produce numbers for. If running on UL, "
    "add this to the start, e.g. UL2018")
parser.add_argument("file", help="Input .h file")
parser.add_argument("-o", "--output", help="Output file, default "
                    "MET_xy_shift.json", default="MET_xy_shift.json")
args = parser.parse_args()

xshifts = {}
yshifts = {}

with open(args.input) as f:
    # These files currently contain a number of blocks, depending on
    # whether one is using the v2 corrections (only recommended for
    # non-UL 2017) and puppi MET (we currently assume this is not the
    # case)
    readlines = False
    for line in f:
        if args.year != "2017":
            if "if(!usemetv2){" in line:
                readlines = True
        else:
            if "else {//these are the corrections for v2 MET recipe" in line:
                readlines = True
        if "}" in line:
            readlines = False

        if readlines:
            _, _, id_, nums = line.split("=")
            if ("y" + args.year) in id_:
                era = id_.split(")")[0][1:]
                n1, n2 = nums[3:-3].split("*npv")
                if id_.split(")")[1] == " METxcorr ":
                    xshifts[era] = [float(n1), float(n2)]
                elif id_.split(")")[1] == " METycorr ":
                    yshifts[era] = [float(n1), float(n2)]

with open(args.output, "w+") as f:
    json.dump({"METxcorr": xshifts, "METycorr": yshifts}, f, indent=4)
