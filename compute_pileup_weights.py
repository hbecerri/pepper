import os
import sys
import pepper
import coffea
import uproot
import parsl
from argparse import ArgumentParser


class Processor(pepper.Processor):
    def __init__(self, config):
        super().__init__(config, None)

    def setup_selection(self, data, dsname, is_mc, filler):
        # Ignore generate weights, as pileup is independent
        return pepper.Selector(data, on_update=filler.get_callbacks())

    def process_selection(self, selector, dsname, is_mc, filler):
        pass


def save_output(hist, datahist, datahistup, datahistdown, filename):
    with uproot.recreate(filename) as f:
        for idn in hist.identifiers("dataset"):
            dataset = idn.name
            hist_int = hist.integrate("dataset", dataset)
            hist_int.scale(1 / hist_int.project().values()[()])
            for datahist_i, suffix in [
                    (datahist, ""), (datahistup, "_up"),
                    (datahistdown, "_down")]:
                ratio = pepper.misc.hist_divide(datahist_i, hist_int)
                f[dataset + suffix] = pepper.misc.export(ratio)


parser = ArgumentParser(
    description="Create histograms needed for pileup reweighting")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "data_pu_hist", help="Root file containing a histogram named 'pileup' for "
    "data as outputted by the CMSSW script pileupCalc.py")
parser.add_argument("data_pu_hist_up", help="Up variation of data_pu_hist")
parser.add_argument("data_pu_hist_down", help="down variation of data_pu_hist")
parser.add_argument(
    "-o", "--output", help="Name of the output file. Defaults to pileup.root",
    default="pileup.root")
parser.add_argument(
    "--perdataset", help="Output dataset-dependent weights to this file. "
    "This should be used if not all datasets have the same pileup settings.",
    default=None)
parser.add_argument(
    "-c", "--condor", type=int, const=10, nargs="?", metavar="simul_jobs",
    help="Split and submit to HTCondor. By default 10 condor jobs are "
    "submitted. The number can be changed by supplying it to this option")
parser.add_argument(
    "--chunksize", type=int, default=500000, help="Number of events to "
    "process at once. Defaults to 5*10^5")
parser.add_argument(
    "-d", "--debug", action="store_true", help="Only process a small amount "
    "of files to make debugging feasible")
args = parser.parse_args()

if os.path.exists(args.output):
    a = input(f"Overwrite {args.output}? y/n ")
    if a != "y":
        sys.exit(1)
if args.perdataset is not None and os.path.exists(args.perdataset):
    a = input(f"Overwrite {args.perdataset}? y/n ")
    if a != "y":
        sys.exit(1)
config = pepper.Config(args.config)
datasets = config.get_datasets(dstype="mc")
if args.debug:
    print("Processing only one file per dataset because of --debug")
    datasets = {key: [val[0]] for key, val in datasets.items()}

with uproot.open(args.data_pu_hist) as f:
    datahist = pepper.misc.rootimport(f["pileup"])
with uproot.open(args.data_pu_hist_up) as f:
    datahistup = pepper.misc.rootimport(f["pileup"])
with uproot.open(args.data_pu_hist_down) as f:
    datahistdown = pepper.misc.rootimport(f["pileup"])
if len(datahist.axes()) != 1:
    print("data_pu_hist has invalid number of axes. Only one axis is allowed.")
    sys.exit(1)
if not datahist.compatible(datahistup):
    print("data_pu_hist_up does not have the same shape as data_pu_hist.")
    sys.exit(1)
if not datahist.compatible(datahistdown):
    print("data_pu_hist_down does not have the same shape as data_pu_hist.")
    sys.exit(1)

config["hists"] = {
    "pileup": pepper.HistDefinition({
        "bins": [
            {
                "name": datahist.axes()[0].name,
                "label": "True mean number interactions per bunch crossing",
                "n_or_arr": datahist.axes()[0].edges()
            }
        ],
        "fill": {
            datahist.axes()[0].name: [
                "Pileup_nTrueInt"
            ]
        }
    })
}
if "hists_to_do" in config:
    del config["hists_to_do"]
config["compute_systematics"] = False
# Treat all datasets as normal datasets, instead of using them as systematic
config["dataset_for_systematics"] = {}

processor = Processor(config)

os.makedirs(os.path.dirname(os.path.realpath(args.output)), exist_ok=True)

if args.condor is not None:
    executor = coffea.processor.parsl_executor
    # Load parsl config immediately instead of putting it into executor_args
    # to be able to use the same jobs for preprocessing and processing
    print("Spawning jobs. This can take a while")
    parsl.load(pepper.misc.get_parsl_config(args.condor))
    executor_args = {}
else:
    executor = coffea.processor.iterative_executor
    executor_args = {}

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args,
    chunksize=args.chunksize)

# Normalize data histograms
datahist.scale(1 / datahist.project().values()[()])
datahistup.scale(1 / datahistup.project().values()[()])
datahistdown.scale(1 / datahistdown.project().values()[()])

mchist = output["hists"][("Before cuts", "pileup")]
# Set underflow and 0 pileup bin to 0, which might be != 0 only for buggy
# reasons in MC
for idn in mchist.identifiers("dataset"):
    mchist._sumw[(idn,)][:2] = 0

new_ds_axis = coffea.hist.Cat("dataset", "Dataset")
mchist_allds = mchist.group(
    mchist.axis("dataset"), new_ds_axis,
    {"all_datasets": [i.name for i in mchist.identifiers("dataset")]})
save_output(mchist_allds, datahist, datahistup, datahistdown, args.output)
if args.perdataset is not None:
    save_output(mchist, datahist, datahistup, datahistdown, args.perdataset)
