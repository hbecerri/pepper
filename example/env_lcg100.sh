#!/bin/bash

unset HDF5_PLUGIN_PATH
source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_100 x86_64-centos7-gcc10-opt
export PATH=~/.local/bin:$PATH
export PEPPER_CONDOR_ENV=$(realpath $0)
