#!/usr/bin/env python3

import os
import numpy as np
import awkward
import coffea
from functools import partial
import shutil
import parsl
from parsl.addresses import address_by_hostname
from argparse import ArgumentParser

import utils.config
from utils.datasets import expand_datasetdict
from processor import Processor


def get_channel_masks(data):
    p0 = abs(data["Lepton"]["pdgId"][:, 0])
    p1 = abs(data["Lepton"]["pdgId"][:, 1])
    return {
        "ee": (p0 == 11) & (p1 == 11),
        "mm": (p0 == 13) & (p1 == 13),
        "em": p0 != p1,
    }


def make_particle_hist(particle_name, data, dsname, is_mc, weight):
    """pt, eta, phi histogram. Due to the high dimensionality, a histogram has
    82080 bins, considering a single dataset source, so its memory intensive.
    """
    hist = coffea.hist.Hist(
        "Counts",
        coffea.hist.Cat("dsname", "Dataset name", "integral"),
        coffea.hist.Cat("chan", "Channel", "integral"),
        coffea.hist.Bin(
            "pt", "{} $p_{{\\mathrm{{T}}}}$ (GeV)".format(particle_name), 20,
            0, 200),
        coffea.hist.Bin(
            "eta", r"{} $\eta$".format(particle_name), 26, -2.6, 2.6),
        coffea.hist.Bin(
            "phi", r"{} $\varphi$".format(particle_name), 38, -3.8, 3.8)
    )
    if particle_name in data:
        for chan, mask in get_channel_masks(data).items():
            pt = data[particle_name].pt[mask].flatten()
            eta = data[particle_name].eta[mask].flatten()
            phi = data[particle_name].phi[mask].flatten()
            if weight is not None:
                chan_weight = np.repeat(weight[mask],
                                        data[particle_name][mask].counts)
                hist.fill(dsname=dsname, chan=chan, pt=pt, eta=eta, phi=phi,
                          weight=chan_weight)
            else:
                hist.fill(dsname=dsname, chan=chan, pt=pt, eta=eta, phi=phi)
    return hist


def make_onedim_hist(xlabel, binopts, datasource, data, dsname, is_mc, weight):
    hist = coffea.hist.Hist(
        "Counts",
        coffea.hist.Cat("dsname", "Dataset name", "integral"),
        coffea.hist.Cat("chan", "Channel", "integral"),
        coffea.hist.Bin("x", xlabel, *binopts)
    )
    if callable(datasource):
        x = datasource(data)
    elif isinstance(datasource, tuple):
        if datasource[0] not in data:
            x = None
        else:
            x = getattr(data[datasource[0]], datasource[1])
    else:
        if datasource not in data:
            x = None
        else:
            x = data[datasource]
    if x is not None:
        for chan, mask in get_channel_masks(data).items():
            x_masked = x[mask]
            if weight is not None:
                if isinstance(x_masked, awkward.JaggedArray):
                    chan_weight = np.repeat(weight[mask], x_masked.counts)
                    hist.fill(dsname=dsname, chan=chan, x=x_masked.flatten(),
                              weight=chan_weight)
                else:
                    hist.fill(dsname=dsname, chan=chan, x=x_masked,
                              weight=weight[mask])
            else:
                hist.fill(dsname=dsname, chan=chan, x=x_masked.flatten())
    return hist


def nbjets_datafunc(data):
    if "Jets" not in data:
        return None
    return data["Jets"]["btagged"].sum()


parser = ArgumentParser(description="Select events from nanoAODs")
parser.add_argument("config", help="Path to a configuration file")
parser.add_argument(
    "eventdir", help="Path to event destination output directory")
parser.add_argument(
    "histdir", help="Path to the histogram destination output directory")
parser.add_argument(
    "--dataset", nargs=2, action="append", metavar=("name", "path"),
    help="Can be specified multiple times. Ignore datasets given in "
    "config and instead process these")
parser.add_argument(
    "-c", "--condor", type=int, const=10, nargs="?", metavar="simul_jobs",
    help="Split and submit to HTCondor. By default 10 condor jobs are "
    "submitted. The number can be changed by supplying it to this option"
)
parser.add_argument(
    "--chunksize", type=int, default=500000, help="Number of events to "
    "process at once. Defaults to 5*10^5")
parser.add_argument(
    "--skip_sysds", action="store_true", help="Skip the datasets that are "
    "used for systematics calculation as given by datasets_for_systematics")
parser.add_argument(
    "--mc", action="store_true", help="Only process MC files")
parser.add_argument(
    "-d", "--debug", action="store_true", help="Only process a small amount "
    "of files to make debugging feasible")
args = parser.parse_args()


config = utils.config.Config(args.config)
store = config["store"]


datasets = {}
if args.dataset is None:
    datasets = {}
    if not args.mc:
        datasets.update(config["exp_datasets"])
    duplicate = set(datasets.keys()) & set(config["mc_datasets"])
    if len(duplicate) > 0:
        print("Got duplicate dataset names: {}".format(", ".join(duplicate)))
        exit(1)
    datasets.update(config["mc_datasets"])
else:
    datasets = {}
    for dataset in args.dataset:
        if dataset[0] in datasets:
            datasets[dataset[0]].append(dataset[1])
        else:
            datasets[dataset[0]] = [dataset[1]]
if args.skip_sysds:
    for sysds in config["datasets_for_systematics"].keys():
        if sysds in datasets:
            del datasets[sysds]

datasets, paths2dsname = expand_datasetdict(datasets, store)
if args.dataset is None:
    num_files = len(paths2dsname)
    num_mc_files = sum(len(datasets[dsname])
                       for dsname in config["mc_datasets"].keys())

    print("Got a total of {} files of which {} are MC".format(num_files,
                                                              num_mc_files))

if args.debug:
    print("Processing only one file because of --debug")
    key = next(iter(datasets.keys()))
    datasets = {key: datasets[key][:1]}

nonempty = []
for dsname in datasets.keys():
    try:
        next(os.scandir(os.path.join(args.eventdir, dsname)))
    except (FileNotFoundError, StopIteration):
        pass
    else:
        nonempty.append(dsname)
if len(nonempty) != 0:
    print("Non-empty output directories: {}".format(", ".join(nonempty)))
    while True:
        answer = input("Delete? y/n ")
        if answer == "y":
            for dsname in nonempty:
                shutil.rmtree(os.path.join(args.eventdir, dsname))
            break
        elif answer == "n":
            break

hist_dict = {
    "Leptonpt": partial(
        make_onedim_hist, "Lepton $p_{{\\mathrm{{T}}}}$ (GeV)", (20, 0, 200),
        ("Lepton", "pt")),
    "Leptoneta": partial(
        make_onedim_hist, r"Lepton $\eta$", (26, -2.6, 2.6),
        ("Lepton", "eta")),
    "Leptonphi": partial(
        make_onedim_hist, r"Lepton $\varphi$", (38, -3.8, 3.8),
        ("Lepton", "phi")),
    "Jetpt": partial(
        make_onedim_hist, "Jet $p_{{\\mathrm{{T}}}}$ (GeV)", (20, 0, 200),
        ("Jet", "pt")),
    "Jeteta": partial(
        make_onedim_hist, r"Jet $\eta$", (26, -2.6, 2.6), ("Jet", "eta")),
    "Jetphi": partial(
        make_onedim_hist, r"Jet $\varphi$", (38, -3.8, 3.8), ("Jet", "phi")),
    "METpt": partial(
        make_onedim_hist, "MET $p_{{\\mathrm{{T}}}}$ (GeV)", (20, 0, 200),
        ("MET", "pt")),
    "METphi": partial(
        make_onedim_hist, r"MET $\varphi$", (38, -3.8, 3.8), ("MET", "phi")),
    "mll": partial(make_onedim_hist, "$M_{ll}$ (GeV)", (20, 0, 200), "mll"),
    "njet": partial(
        make_onedim_hist, "Number of jets", (np.arange(10),),
        ("Jet", "counts")),
    "nbjet": partial(
        make_onedim_hist, "Number of b-tagged jets", (np.arange(10),),
        nbjets_datafunc),
}

processor = Processor(config, os.path.realpath(args.eventdir), hist_dict)
if args.condor is not None:
    executor = coffea.processor.parsl_executor
    conor_config = ("requirements = (OpSysAndVer == \"SL6\" || OpSysAndVer =="
                    " \"CentOS7\")")
    # Need to unset PYTHONPATH because of cmssw setting it incorrectly
    # Need to put own directory into PYTHONPATH for unpickling processor to
    # work. Should be unncessecary, once we have correct module structure.
    # Need to extend PATH to be able to execute the main parsl script
    condor_init = """
source /cvmfs/cms.cern.ch/cmsset_default.sh
if lsb_release -r | grep -q 7\\.; then
cd /cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
else
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
fi
eval `scramv1 runtime -sh`
cd -
export PYTHONPATH={}
export PATH=~/.local/bin:$PATH
""".format(os.path.dirname(os.path.abspath(__file__)))
    provider = parsl.providers.CondorProvider(
        init_blocks=args.condor,
        max_blocks=args.condor,
        scheduler_options=conor_config,
        worker_init=condor_init
        )
    parsl_executor = parsl.executors.HighThroughputExecutor(
        label="HTCondor",
        address=address_by_hostname(),
        max_workers=1,
        provider=provider,
    )
    parsl_config = parsl.config.Config(
        executors=[parsl_executor],
        retries=100000,
    )

    # Load config now instead of putting it into executor_args to be able to
    # use the same jobs for preprocessing and processing
    print("Spawning jobs. This can take a while")
    parsl.load(parsl_config)
    executor_args = {}
else:
    executor = coffea.processor.iterative_executor
    executor_args = {}

output = coffea.processor.run_uproot_job(
    datasets, "Events", processor, executor, executor_args,
    chunksize=args.chunksize)

os.makedirs(args.histdir, exist_ok=True)
for key, hist in output["sel_hists"].items():
    if hist.values() == {}:
        continue
    fname = "_".join(key) + ".coffea"
    coffea.util.save(hist, os.path.join(args.histdir, fname))
