from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray as JCA
from coffea.util import awkward, numpy
from coffea.util import numpy as np
from awkward import JaggedArray
import coffea.processor
from coffea.processor import parsl_executor
import uproot
import matplotlib
import matplotlib.pyplot as plt
from parsl import load, python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import CondorProvider
from parsl.addresses import address_by_hostname
from parsl.channels import LocalChannel
import h5py
import mplhep

from collections import defaultdict
import os

import config_utils
from processor import Processor

matplotlib.interactive(True)

wrk_init = """
export PATH=/afs/desy.de/user/s/stafford/.local/bin:$PATH
export PYTHONPATH=\
/afs/desy.de/user/s/stafford/.local/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=\/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea:$PYTHONPATH
"""

nproc = 1
condor_cfg = """
Requirements = OpSysAndVer == "CentOS7"
RequestMemory = %d
RequestCores = %d
""" % (2048*nproc, nproc)

parsl_config = Config(
        executors=[HighThroughputExecutor(label="HTCondor",
                   address=address_by_hostname(),
                   prefetch_capacity=0,
                   cores_per_worker=1,
                   max_workers=nproc,
                   provider=CondorProvider(
                        channel=LocalChannel(),
                        init_blocks=800,
                        min_blocks=5,
                        max_blocks=1000,
                        nodes_per_block=1,
                        parallelism=1,
                        scheduler_options=condor_cfg,
                        worker_init=wrk_init))],
        lazy_errors=False
)
#dfk = load(parsl_config)

config = config_utils.Config("example/config.json")
store = config["store"]
fileset, _ = config_utils.expand_datasetdict(config["mc_datasets"], store)
smallfileset = {"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8":
                ["/pnfs/desy.de/cms/tier2/store/mc/RunIIAutumn18NanoAODv5/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/Nano1June2019_102X_upgrade2018_realistic_v19-v1/250000/14933F79-95FB-354D-A917-E19B5C005037.root"]}
destdir="/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea/selected_columns"
output = coffea.processor.run_uproot_job(
    smallfileset,
    treename="Events",
    processor_instance=Processor(config, "selected_columns"),
    executor=coffea.processor.iterative_executor,
    executor_args={"workers": 4},
    chunksize=100000)
"""
output = coffea.processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=Processor(config, destdir),
    executor=parsl_executor,
    chunksize=500000)"""

plot_config = config_utils.Config("example/plot_config.json")
labels = plot_config["labels"]
colours = plot_config["colours"]
xsecs = plot_config["cross-sections"]

#plt.style.use(mplhep.cms.style.ROOT)

cutvalues = dict((k, np.zeros(
    len(output["cutflow"]["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"])))
    for k in set(labels.values()))
cuteffs = dict((k, np.zeros(len(
    output["cutflow"]["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]) - 1))
    for k in set(labels.values()))
# currently assumes one always runs over a dilepton sample-
# might be nice to relax this
lumifactors = defaultdict(int)
for dataset in fileset.keys():
    cutvals = np.array(list(output["cutflow"][dataset].values()))
    if len(cutvals) == 0:
        eff = 0
        lumifactors[dataset] = 0
    else:
        eff = cutvals[-1]/cutvals[0]
        lumifactors[dataset] = 0.05 * xsecs[dataset]/cutvals[0]
    print(dataset, "efficiency:", eff*100)
    if len(cutvals > 0):
        cutvalues[labels[dataset]] += cutvals

labelsset = list(set(labels.values()))
nlabels = len(labelsset)
ax = plt.gca()
for n, label in enumerate(labelsset):
    cuteffs[label] = 100*cutvalues[label][1:]/cutvalues[label][:-1]
    ax.bar(np.arange(len(cuteffs[label])) + (2*n-nlabels)*0.4/nlabels,
           cuteffs[label], 0.8/nlabels, label=label, color=colours[label])

ax.set_xticks(np.arange(len(
    cuteffs[labels["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]])))
ax.set_xticklabels(np.array(list(
    (output["cutflow"]["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]).keys()))[1:])
ax.set_ylabel("Efficiency")

handles, labs = ax.get_legend_handles_labels()
# https://stackoverflow.com/questions/43348348/pyplot-legend-index-error-tuple-index-out-of-range
leghandles = []
leglabs = []
for i, h in enumerate(handles):
    if len(h):
        leghandles.append(h)
        leglabs.append(labs[i])
ax.legend(leghandles, leglabs)

plt.show(block=True)

output["Mttbar"].scale(lumifactors, axis="dataset")
labelmap = defaultdict(list)
for key, val in labels.items():
    cutvals = np.array(list(output["cutflow"][key].values()))
    if len(cutvals) > 0 and cutvals[-1] > 0:
        labelmap[val].append(key)

sortedlabels = sorted(labelsset, key=(
    lambda x: sum([(output["Mttbar"].integrate("Mttbar")).values()[(y,)]
                  for y in labelmap[x]])))
for key in sortedlabels:
    labelmap[key] = labelmap.pop(key)
    colours[key] = colours.pop(key)

labels_axis = hist.Cat("labels", "", sorting="placement")
mttbar = output["Mttbar"].group("dataset", labels_axis, labelmap)

ax = hist.plot1d(mttbar, overlay="labels", stack=True)

plt.show(block=True)
