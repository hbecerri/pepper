# Pepper - ParticlE Physics ProcEssoR
A python framework for analyzing NanoAODs. Easy to use and highly configurable.

Currently focusing on <img src="https://latex.codecogs.com/gif.latex?\mathrm{t\bar{t}}\rightarrow\mathrm{ll\nu\nu}" />.

## Installation
Requires Python3 and the Python packages `coffea`, `awkward`, `parsl`, `h5py`, `hdf5plugin` and `hjson`

```sh
python3 -m pip install --user --upgrade pip
python3 -m pip install --user coffea awkward parsl h5py hdf5plugin hjson
```

Pepper has been tested on and is recommended to be used with `CMSSW_11_1_0_pre5_PY3`.

The features of the framework are implemented as a Python package, which is inside the `pepper` directory. To use it, you can add the path to where you downloaded the repository to the `PYTHONPATH` variable

```sh
git clone <repository url> pepper
export PYTHONPATH=`pwd`/pepper:$PYTHONPATH
```

## Installation through pip3
Alternatively, Pepper can be installed as a python package as follows:
```sh
git clone <repository url> pepper
cd pepper
python3 -m pip install . --user
```
Use the option `-e` to make the installed package editable.
Now, `pepper` can be imported as any other python package from any location.



## Running
The main directory contains numerous scripts that can be evoked:

 - `calculate_DY_SFs.py`: Calculate scale factors for DY reweighting from the output of produce_DY_numbers.py
 - `compute_kinreco_hists.py`: Generate histograms needed for top-quark kinematic reconstruction
 - `compute_mc_lumifactors.py`: Compute <img align="top" src="https://latex.codecogs.com/gif.latex?{\cal L}\sigma/\sum w_{\mathrm{gen}}" />, the factors needed to scale MC to data
 - `compute_pileup_weights.py`: Compute scale factors for pileup reweighting
 - `delete_duplicate_outputs.py`: Check for duplication in the per event data produced by select_events.py, and move or delete any duplicates
 - `generate_btag_efficiencies.py`: Generate a ROOT file containing efficiency histograms needed for b-tagging scale factors
 - `merge_hists.py`: Caluclate weighted average of two SF histograms
 - `plot_control.py`: Create control plots from histogram output generated by `select_events.py`
 - `produce_DY_numbers.py`: Produce the numbers needed for DY SF calculation
 - `select_events.py`: Run the main analysis procedure, outputting histograms and per event data



## Configuration
Configuration is done via JSON files. Examples can be found in the `example` directory. Additional data needed for configuration, for example scale factors and cross sections, can be found in a separate data repository here https://gitlab.cern.ch/pepper/data
After downloading it, make sure to set the configuration variable "datadir" to the path where the data repository was downloaded.
For a detailed explanation on the configuration variables, see `config_ttbarll_documentation.md`.
