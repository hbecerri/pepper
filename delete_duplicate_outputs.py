import os
import json
from argparse import ArgumentParser
from tqdm import tqdm
from pprint import pprint

import parsl
from parsl import python_app

import pepper


@python_app
def process_dir(dir, delete=False):
    import awkward
    import h5py
    import glob
    import os
    duplicate_files = []
    corrupted_files = []
    processed_chunks = set()
    for in_file in glob.glob(os.path.join(dir, "*.hdf5")):
        try:
            with h5py.File(in_file, "r") as f:
                g = awkward.hdf5(f)
                identifier = g["identifier"]
        except OSError:
            if delete:
                os.remove(in_file)
            else:
                os.rename(in_file, in_file + ".corrupted")
            corrupted_files.append(os.path.relpath(in_file, dir))
            continue
        if identifier in processed_chunks:
            if delete:
                os.remove(in_file)
            else:
                os.rename(in_file, in_file + ".duplicate")
            duplicate_files.append(os.path.relpath(in_file, dir))
        processed_chunks.add(identifier)
    return duplicate_files, corrupted_files


parser = ArgumentParser(
    description="Check if any of the samples in the event directory produced"
    " by select_events.py are duplicated or corrupted, and if so, rename or "
    "delete them")
parser.add_argument("event_dir", help="Directory to check")
parser.add_argument(
    "-c", "--condor", type=int, default=100, nargs="?", metavar="simul_jobs",
    help="Number of HTCondor jobs to launch, default 100")
parser.add_argument(
    "-r", "--retries", type=int, help="Number of times to retry if there is "
    "exception in an HTCondor job. If not given, retry infinitely."
)
parser.add_argument(
    "-o", "--offset", type=int, help="Skip the first <offset> directories",
    default=0)
parser.add_argument(
    "-d", "--delete", action="store_true", help="Delete duplicate or corrupt "
    "files instead of renaming them")
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

dirs = next(os.walk(sample_dir))[1][args.offset:]
print(f"Number of sample directories to run over: {len(dirs)}")

futures = {}
for d in dirs:
    futures[d] = process_dir(os.path.join(sample_dir, d), args.delete)
results = {}
for d, future in tqdm(futures.items()):
    results[d] = future.result()

print("Duplicate or corrupted files: ")
pprint(results)
