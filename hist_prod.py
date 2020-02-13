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
import json

import utils.config as config_utils
import utils.datasets as dataset_utils
from processor import Processor


class hist_set():
    def __init__(self, hist_dict):
        self.accumulator = coffea.processor.dict_accumulator({})
        self.hist_dict = hist_dict
        self.sys = []

    def set_ds(self, dsname, is_mc):
        self.dsname = dsname
        self.is_mc = is_mc

    def __call__(self, data, cut):
        for hist, fill_func in self.hist_dict.items():
            self.accumulator[(cut, hist)] = \
                fill_func(data, self.dsname, self.is_mc)
            for sys in self.sys:
                self.accumulator[(cut, hist, sys)] = \
                    fill_func(data, self.dsname, self.is_mc, sys)


def fill_MET(data, dsname, is_mc):
    dataset_axis = hist.Cat("dataset", "")
    MET_axis = hist.Bin("MET", "MET [GeV]", 100, 0, 400)
    MET_hist = hist.Hist("Counts", dataset_axis, MET_axis)
    if is_mc:
        MET_hist.fill(dataset=dsname, MET=data["MET_pt"],
                      weight=data["genWeight"])
    else:
        MET_hist.fill(dataset=dsname, MET=data["MET_pt"])
    return MET_hist


def fill_Mll(data, dsname, is_mc):
    dataset_axis = hist.Cat("dataset", "")
    Mll_axis = hist.Bin("Mll", "Mll [GeV]", 100, 0, 400)
    Mll_hist = hist.Hist("Counts", dataset_axis, Mll_axis)
    if "Mll" in data:
        if is_mc:
            Mll_hist.fill(dataset=dsname, Mll=data["Mll"],
                          weight=data["genWeight"])
        else:
            Mll_hist.fill(dataset=dsname, Mll=data["Mll"])
    return Mll_hist


# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


wrk_init = """
export PATH=/afs/desy.de/user/s/stafford/.local/bin:$PATH
export PYTHONPATH=\
/afs/desy.de/user/s/stafford/.local/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=\
/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea:$PYTHONPATH
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
                        init_blocks=20,
                        min_blocks=5,
                        max_blocks=500,
                        nodes_per_block=1,
                        parallelism=0.5,
                        scheduler_options=condor_cfg,
                        worker_init=wrk_init))],
        retries=10
)
dfk = load(parsl_config)

config = config_utils.Config("example/config.json")
store = config["store"]
mc_fileset, _ = dataset_utils.expand_datasetdict(config["mc_datasets"], store)
data_fileset, _ = dataset_utils.expand_datasetdict(config["exp_datasets"],
                                                   store)
data_fileset.update(mc_fileset)
fileset = data_fileset
'''fileset = {"WW_TuneCP5_13TeV-pythia8": fileset["WW_TuneCP5_13TeV-pythia8"],
           "WZ_TuneCP5_13TeV-pythia8": fileset["WZ_TuneCP5_13TeV-pythia8"],
           "ZZ_TuneCP5_13TeV-pythia8": fileset["ZZ_TuneCP5_13TeV-pythia8"]}
{"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8":
    fileset["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"]}'''
# smallfileset, _ = \
#    dataset_utils.expand_datasetdict(config["testdataset"], store)
smallfileset = {key: [val[0]] for key, val in fileset.items()}
destdir = \
    "/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea/selected_columns"

selector_hist_set = hist_set({"MET": fill_MET,
                              "Mll": fill_Mll})

"""output = coffea.processor.run_uproot_job(
    smallfileset,
    treename="Events",
    processor_instance=Processor(config, "None", selector_hist_set),
    executor=coffea.processor.iterative_executor,
    executor_args={"workers": 4},
    chunksize=100000)
"""
output = coffea.processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=Processor(config, "None", selector_hist_set),
    executor=parsl_executor,
    executor_args={"tailtimeout": None},
    chunksize=500000)

coffea.util.save(output, "out_hists/output.coffea")
