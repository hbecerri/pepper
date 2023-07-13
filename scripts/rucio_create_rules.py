#!/usr/bin/env python3

from argparse import ArgumentParser
import sys
import itertools

import hjson
try:
    import rucio.client
except ImportError:
    print(
        "Could not import Rucio. Make sure your environment contains an "
        "installation\n"
        "You can run the following command to do so:\n"
        "source /cvmfs/cms.cern.ch/rucio/setup-py3.sh"
    )
    sys.exit(1)


parser = ArgumentParser(
    description="Creates Rucio rules for all data sets specified in a Pepper "
                "config. Once the rules are approved, the data sets will be"
                "transfered to the local site")
parser.add_argument("config")
parser.add_argument("site", help="The target site. For example T2_DE_DESY")
parser.add_argument(
    "-l", "--lifetime", type=int, default=15552000,
    help="Lifetime of the rule in seconds. If negative, do not specify a "
         "lifetime. Defaults to 6 months")
args = parser.parse_args()

with open(args.config) as f:
    config = hjson.load(f)
datasets = []
for dspaths in itertools.chain(
        config["exp_datasets"].values(), config["mc_datasets"].values()):
    for dspath in dspaths:
        if (dspath.count("/") == 3
            and dspath.startswith("/")
            and (dspath.endswith("/NANOAOD")
                 or dspath.endswith("/NANOAODSIM"))):
            datasets.append(dspath)
dids = [{"scope": "cms", "name": dataset} for dataset in datasets]
if args.lifetime < 0:
    lifetime = None
else:
    lifetime = args.lifetime
client = rucio.client.Client()
for dataset in datasets:
    print(dataset)
    try:
        client.add_replication_rule(
            dids=[{"scope": "cms", "name": dataset}],
            copies=1,
            rse_expression=args.site,
            lifetime=lifetime,
            ask_approval=True
        )
    except rucio.common.exception.DuplicateRule:
        pass
