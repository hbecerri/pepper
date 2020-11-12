# Pepper - ParticlE Physics ProcEssoR
A python framework for analyzing NanoAODs. Easy to use and highly configurable.

Currently focusing on <img src="https://latex.codecogs.com/gif.latex?\mathrm{t\bar{t}}\rightarrow\mathrm{ll\nu\nu}" />.

## Installation
Requires Python3 and the Python packages `coffea`, `parsl` and `h5py`

> pip3 install --user coffea parsl h5py

Unfortunately the naf nodes seem to have a problem with the LLVM bindings, so it is necessary to activate a CMSSW release before doing this install; the recommended release is CMSSW_10_2_4_patch1.

The features of the framework are implemented as a Python package, which is inside the `pepper` directory. To use it, you can add the path to where you downloaded the repository to the `PYTHONPATH` variable

> git clone \<repository url\> pepper

> export PYTHONPATH=\`pwd\`/pepper:$PYTHONPATH



## Running
The main directory contains numerous scripts that can be evoked:

 - `calculate_DY_SFs.py`: Generate scale factors for DY reweighting
 - `compute_kinreco_hists.py`: Generate histograms needed for top-quark kinematic reconstruction
 - `compute_mc_lumifactors.py`: Compute <img align="top" src="https://latex.codecogs.com/gif.latex?{\cal L}\sigma/\sum w_{\mathrm{gen}}" />, the factors needed to scale MC to data
 - `compute_pileup_weights.py`: Compute scale factors for pileup reweighting
 - `delete_duplicate_cols.py`: Check for duplication in the per event data produced by select_events.py, and delete any duplicates
 - `generate_btag_efficiencies.py`: Generate a ROOT file containing efficiency histograms needed for b-tagging scale factors
 - `plot_control.py`: Create control plots from histogram output generated by `select_events.py`
 - `select_events.py`: Run the main analysis procedure, outputting histograms and per event data



## Configuration
Configuration is done via JSON files. Examples can be found in the `example` directory. Additional data needed for configuration, for example scale factors and cross sections, can be found in a separate data repository here https://gitlab.cern.ch/desy-ttbarbsm-coffea/data
After downloading it, make sure to set the config variable "datadir" to the path where the data repository was downloaded.
