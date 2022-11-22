#!/bin/bash

# Load LCG 100
source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_102 x86_64-centos7-gcc11-opt
# Make sure python libs installed in the user directory are prefered over system-wide ones
export PYTHONPATH=`python3 -c 'import site; print(site.getusersitepackages())'`:$PYTHONPATH
# Parsl installs some of it's commands into ~/.local/bin if installed as user
export PATH=~/.local/bin:$PATH
# On DESY NAF, old HDF5 plugins that are installed system-wide break HDF5 functionality. Disable
unset HDF5_PLUGIN_PATH
# Use this script also as environment script when running Pepper on HTCondor
if test -n "$BASH_VERSION"; then
    export PEPPER_CONDOR_ENV="$(realpath $BASH_SOURCE)"
elif test -n "$ZSH_VERSION"; then
    export PEPPER_CONDOR_ENV="$(realpath ${(%):-%N})"
fi
