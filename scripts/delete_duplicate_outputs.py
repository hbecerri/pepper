import os
from argparse import ArgumentParser
from tqdm import tqdm

import parsl
from parsl import python_app
import concurrent.futures

import pepper


def process_dir(dir, delete=False):
    import pepper
    import os
    duplicate_files = []
    corrupted_files = []
    processed_chunks = set()
    for in_file in os.listdir(dir):
        if not in_file.endswith(".hdf5") and not in_file.endswith(".h5"):
            continue
        in_file = os.path.join(dir, in_file)
        try:
            with pepper.HDF5File(in_file, "r") as f:
                identifier = f["identifier"]
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
    return dir, (duplicate_files, corrupted_files)


parser = ArgumentParser(
    description="Check if any of the samples in the event directory produced"
    " by select_events.py are duplicated or corrupted, and if so, rename or "
    "delete them")
parser.add_argument(
    "eventdir", help="Directory containing sample directories. Should be the "
    "same as the eventdir argument when running a processor")
parser.add_argument(
    "-c", "--condor", type=int, metavar="simul_jobs",
    help="Number of HTCondor jobs to launch")
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
    "-i", "--condorinit",
    help="Shell script that will be sourced by an HTCondor job after "
    "starting. This can be used to setup environment variables, if using "
    "for example CMSSW. If not provided, the local content of the "
    "environment variable PEPPER_CONDOR_ENV will be used as path to the "
    "script instead.")
parser.add_argument(
    "--condorsubmit",
    help="Text file containing additional parameters to put into the "
    "HTCondor job submission file that is used for condor_submit"
)
args = parser.parse_args()


if args.condorinit is not None:
    with open(args.condorinit) as f:
        condorinit = f.read()
else:
    condorinit = None
if args.condorsubmit is not None:
    with open(args.condorsubmit) as f:
        condorsubmit = f.read()
else:
    condorsubmit = None

sample_dir = os.path.realpath(args.eventdir)

dirs = next(os.walk(sample_dir))[1][args.offset:]

results = {}
if args.condor is None:
    for d in tqdm(dirs):
        dir, result = process_dir(os.path.join(sample_dir, d), args.delete)
        results[dir] = result
else:
    process_dir = python_app(process_dir)

    parsl_config = pepper.misc.get_parsl_config(
        args.condor,
        condor_submit=condorsubmit,
        condor_init=condorinit,
        retries=args.retries)
    parsl.load(parsl_config)

    futures = set()
    for d in dirs:
        futures.add(process_dir(os.path.join(sample_dir, d), args.delete))
    for future in tqdm(concurrent.futures.as_completed(futures),
                       total=len(futures)):
        dir, result = future.result()
        results[dir] = result

print("Duplicate or corrupted files: ")
have_moved = False
for d, (dups, corrupt) in results.items():
    moved_files = dups + corrupt
    if len(moved_files) == 0:
        continue
    print(f"{d}: {', '.join(moved_files)}")
    have_moved = True
if not have_moved:
    print("None")
