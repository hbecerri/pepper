import os
import json
from argparse import ArgumentParser

import parsl
from parsl import python_app

import pepper


@python_app
def process_dir(sample_dir, d):
    import awkward
    import h5py
    import glob
    import os
    deleted_files = []
    processed_chunks = []
    for in_file in glob.glob(os.path.join(sample_dir, d, "*.hdf5")):
        remove = False
        try:
            with h5py.File(in_file, "r") as f:
                g = awkward.hdf5(f)
                if g['identifier'] in processed_chunks:
                    remove = True
                processed_chunks.append(g['identifier'])
            if remove:
                os.remove(in_file)
                deleted_files.append(
                    in_file[len(os.path.join(sample_dir, d)) + 1:])
        except OSError:
            pass
    return(deleted_files)


parser = ArgumentParser(
    description="Check if any of the samples in the event directory produced"
    " by select_events.py are duplicated, and if so, delete the duplicates")
parser.add_argument("event_dir", help="Directory to check")
parser.add_argument(
    "-c", "--condor", type=int, default=100, nargs="?", metavar="simul_jobs",
    help="Number of HTCondor jobs to launch, default 100")
parser.add_argument(
    "-r", "--retries", type=int, help="Number of times to retry if there is "
    "exception in an HTCondor job. If not given, retry infinitely."
)
parser.add_argument(
    "-p", "--parsl_config", help="JSON file holding a dictionary with the "
    "keys condor_init and condor_config. Former overwrites the enviroment "
    "script that is executed at the start of a Condor job. Latter is appended "
    "to the job submit file.")
args = parser.parse_args()


sample_dir = os.path.realpath(args.event_dir)

if args.parsl_config is not None:
    with open(args.parsl_config) as f:
        parsl_config = json.load(f)
    parsl_config = pepper.misc.get_parsl_config(
        args.condor,
        condor_submit=parsl_config["condor_config"],
        condor_init=parsl_config["condor_init"],
        retries=args.retries)
else:
    parsl_config = pepper.misc.get_parsl_config(
        args.condor, retries=args.retries)
parsl.load(parsl_config)

dirs = next(os.walk(sample_dir))[1]
print("Number of sample directories to run over: ", len(dirs))

results = {}
for d in dirs:
    results[d] = process_dir(sample_dir, d)

print("Directories deleted: ", {d: res.result() for d, res in results.items()})
