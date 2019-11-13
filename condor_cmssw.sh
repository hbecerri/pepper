#!/bin/bash

OLD_PWD=$(pwd)
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
eval `scramv1 runtime -sh`
cd $OLD_PWD
exec $1 ${@:2}
