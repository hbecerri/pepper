# Pepper - ParticlE Physics ProcEssoR
A python framework for analyzing NanoAODs. Easy to use and highly configurable.

Currently focusing on <img src="https://latex.codecogs.com/gif.latex?\mathrm{t\bar{t}}\rightarrow\mathrm{ll\nu\nu}" />.



## Installation
It is recommended to use a proper environment with Pepper. An example environment setup for DESY NAF can be found [here](example/environment.sh), which can be sourced after cloning the repository.
Pepper can be installed as a python package as follows:
```sh
git clone <repository url> pepper
cd pepper
source example/environment.sh
python3 -m pip install . --user
```
Use the option `-e` to make the installed package editable.
Now, `pepper` can be imported as any other python package from any location.



## Usage

### Getting started

In Pepper an analysis is implemented as a Processor class. A short example of such a Processor with many explanatory comments can be found in [here](example/example_processor.py). This processor can be run by executing `python3 -m pepper.runproc example_processor.py example_config.json` (when inside the example directory). Also running `python -m pepper.runproc -h` will show the available command line options.

### Running on HTCondor
In order to run on HTCondor using `pepper.runproc`, one only has to specify the `--condor` parameter followed by the number of jobs desired. Events will be split across jobs as evenly as possible.
To control which environment is employed on the HTCondor node, the parameter `--condorinit` can be used. `--condorinit` should point to a Shell script that can be sourced setting up the environment. If `--condorinit` is not present, Pepper will instead use the script that is pointed at by the local environment variable `PEPPER_CONDOR_ENV`. If this is also not set, the jobs will be run in the default environment of your HTCondor system. For an example environment script setting up LCG100 on CentOS7 see [here](example/env_lcg100.sh).

When running on HTCondor, the local process, that has started also the jobs, needs to be kept open until everything has finished. In order to not accidentally kill the process by an unstable connection or similar, it is recommended to run it inside a `byobu`, `tmux` or `screen` session, or to prepend the command `nohup`.

### Main scripts
The main directory of this repository contains several optional scripts to obtain inputs and plot outputs:

 - `calculate_DY_SFs.py`: Calculate scale factors for DY reweighting from the output of produce_DY_numbers.py
 - `calculate_triggerSFs.py`: Calculate SFs for ttbar dileptonic triggers using the output of produce_triggerSF_numbers.py by cross-trigger method
 - `calulate_stitching_factors.py`: Calculate factors to stitch MC samples from an already produced histogram (currently only 1D histograms supported)
 - `compute_kinreco_hists.py`: Generate histograms needed for top-quark kinematic reconstruction
 - `compute_mc_lumifactors.py`: Compute <img align="top" src="https://latex.codecogs.com/gif.latex?{\cal L}\sigma/\sum w_{\mathrm{gen}}" />, the factors needed to scale MC to data
 - `compute_pileup_weights.py`: Compute scale factors for pileup reweighting
 - `delete_duplicate_outputs.py`: Check for duplication in the per event data produced by select_events.py, and move or delete any duplicates
 - `generate_btag_efficiencies.py`: Generate a ROOT file containing efficiency histograms needed for b-tagging scale factors
 - `merge_hists.py`: Caluclate weighted average of two SF histograms
 - `plot_control.py`: Create control plots from histogram output generated by `select_events.py`
 - `produce_DY_numbers.py`: Produce the numbers needed for DY SF calculation
 - `produce_triggerSF_numbers.py`: Produce the numbers needed for trigger SF calculation
 - `produce_met_xy_nums.py`: Convert MET-xy correction numbers from the C++ headers provided centrally to json files
 - `select_events.py`: Run the main analysis procedure, outputting histograms and per event data



## Configuration
Configuration is done via JSON files. Examples can be found in the `example` directory. Additional data needed for configuration, for example scale factors and cross sections, can be found in a separate data repository here https://gitlab.cern.ch/pepper/data
After downloading it, make sure to set the configuration variable "datadir" to the path where the data repository was downloaded.
For a detailed explanation on the configuration variables, see `config_documentation.md`.



## Contributing
Feel free to submit merge requests to have your code included in this repository! Your code must comply with pep8. You can check this by running the following inside the Pepper directory:
```sh
python3 -m pip install .[dev] --user
python3 -m flake8
```
