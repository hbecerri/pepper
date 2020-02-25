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
from utils.hist_defns import create_hist_dict


wrk_init = """
export PATH=/afs/desy.de/user/s/stafford/.local/bin:$PATH
export PYTHONPATH=\
/afs/desy.de/user/s/stafford/.local/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=\
/nfs/dust/cms/user/stafford/coffea/desy-ttbarbsm-coffea:$PYTHONPATH
"""

hist_dict = create_hist_dict("ttDM_config")

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

config = config_utils.Config("ttDM_config/config.json")
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

"""output = coffea.processor.run_uproot_job(
    smallfileset,
    treename="Events",
    processor_instance=Processor(config, "None", hist_dict),
    executor=coffea.processor.iterative_executor,
    executor_args={"workers": 4},
    chunksize=100000)
"""
output = coffea.processor.run_uproot_job(
    smallfileset,
    treename="Events",
    processor_instance=Processor(config, "None", hist_dict),
    executor=parsl_executor,
    executor_args={"tailtimeout": None},
    chunksize=500000)

print("saving")
coffea.util.save(output, "out_hists/output.coffea")
print("saved")
