#!/usr/bin/env python3

import os
from glob import glob
import logging
from collections import defaultdict
import urllib.request
import ssl
import json


logger = logging.getLogger(__name__)
__cernrootcert = None


def dataset_to_lfn_local(dataset, store, ext=".root"):
    """Get the logical file names of the files belonging to a dataset by
    checking the directory contents of the path pointed to by `store`.

    Parameters:
    dataset -- name of the dataset
    store -- Path to the store directory, e.g. /pnfs/desy.de/cms/tier2/store/
    ext -- File extension the files have

    Returns a list of paths as strings
    """
    primary, processed, tier = dataset.split("/")[1:]
    if tier == "USER":
        username, tag = processed.split("-", 1)
        tag, counter = tag.rsplit("-", 1)
        pat = "{}/user/{}/{}/{}/*/*/*{}".format(
            store, username, primary, tag, ext)
    elif tier.startswith("NANOAOD"):
        campaign, version = processed.split("-", 1)
        datatype = "mc" if tier.endswith("SIM") else "data"
        pat = "{}/{}/{}/{}/{}/{}/*/*{}".format(
            store, datatype, campaign, primary, tier, version, ext)
    else:
        raise ValueError("Unknown tier in dataset, must be NANOAOD, "
                         f"NANOAODSIM or USER: {dataset}")
    return set(["/store/" + os.path.relpath(p, store) for p in glob(pat)])


def lfn_to_local_path(lfn, store):
    len_store = 7  # = len("/store/")
    return os.path.normpath(os.path.join(store, lfn[len_store:]))


def _get_cernca_rootcert():
    global __cernrootcert
    CERNCAPATH = "/cvmfs/grid.cern.ch/etc/grid-security/certificates"
    CERNCAURL = ("https://cafiles.cern.ch/cafiles/certificates/CERN%20Root%20C"
                 "ertification%20Authority%202.crt")

    if os.path.exists(CERNCAPATH):
        verify_locations = {"capath": CERNCAPATH}
    else:
        if __cernrootcert is None:
            logger.debug("Downloading CERN Root Certification certificate")
            with urllib.request.urlopen(CERNCAURL) as f:
                dercert = f.read()
            __cernrootcert = ssl.DER_cert_to_PEM_cert(dercert)
        verify_locations = {"cadata": __cernrootcert}

    return verify_locations


def _get_cert_vomsproxy():
    if "X509_USER_KEY" in os.environ and "X509_USER_CERT" in os.environ:
        return {"certfile": os.environ["X509_USER_CERT"],
                "keyfile": os.environ["X509_USER_KEY"]}
    path = os.environ.get("X509_USER_PROXY", f"/tmp/x509up_u{os.getuid()}")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Could not find certificate file from VOMS proxy at '{path}'. "
            "Did you run voms-proxy-init?")
    return {"certfile": path}


def dataset_to_lfn_dbs(dataset):
    """Get the URLs of the files for a dataset for xrootd using DBS. Either
    needs the environment variables X509_USER_KEY and X509_USER_CERT set to
    user's grid certificate paths or have the VOMS proxy initialized. If the
    proxy certificate file isn't at its usual location (/tmp/x509up_uNNN), it
    can be set via the environment variable X509_USER_PROXY."""

    DBSURL = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader/files?dataset="
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(**_get_cernca_rootcert())
    context.load_cert_chain(**_get_cert_vomsproxy())
    with urllib.request.urlopen(DBSURL + dataset, context=context) as f:
        dbs_reply = f.read()
    dbs_files = json.loads(dbs_reply.decode())
    return set([file["logical_file_name"] for file in dbs_files])


def lfn_to_xrootd_path(lfn, xrootddomain):
    return f"root://{xrootddomain}/" + lfn


def resolve_lfn(lfn, store=None, xrootddomain=None):
    """Get paths and URLs for a file given a specific logical file name (LFN).

    Parameters:
    lfn -- LFN of the file, normally starts with "/store/". This function
           ccepts the additional prefix "cmslfn://store/".
    store -- See expand_datasetdict
    xrootddomain -- See expand_datasetdict

    Returns a list of valid paths and URLs for the given LFN.
    """
    pfns = []
    if lfn.startswith("cmslfn://"):
        lfn = lfn.split("cmslfn:/", 1)[1]
    if store is not None:
        path = lfn_to_local_path(lfn, store)
        if os.path.exists(path):
            pfns.append(path)
    if xrootddomain is not None:
        import XRootD.client
        # Same as xrdfs <xrootddomain> locate -h <lfn>
        client = XRootD.client.FileSystem("root://" + xrootddomain)
        # The flag PrefName (to get domain names instead of IP addresses) does
        # not exist in the Python bidings. However, MAKEPATH has the same value
        locations = client.locate(lfn, XRootD.client.flags.OpenFlags.MAKEPATH)
        domains = [r.address for r in locations[1]]
        pfns.extend(f"root://{d}/{lfn}" for d in domains)
    return pfns


def dataset_to_lfns(dataset, store=None, ext=".root", mode="local"):
    """Get the logical file names (LFN) of the files belonging to a dataset.

    Parameters:
    dataset -- name of the dataset
    store -- See expand_datasetdict
    ext -- See expand_datasetdict
    mode -- See expand_datasetdict

    Returns a list of LFNs as strings
    """

    if mode not in ("local", "xrootd", "local+xrootd"):
        raise ValueError(f"Invalid mode {mode}")
    lfns = set()
    if mode == "local" or mode == "local+xrootd":
        if store is None:
            raise ValueError("Must provide store in local mode")
        lfns |= set(dataset_to_lfn_local(dataset, store, ext))
    if mode == "xrootd" or mode == "local+xrootd":
        lfns |= set(dataset_to_lfn_dbs(dataset))
    lfns = ["cmslfn:/" + lfn for lfn in sorted(lfns)]
    return lfns


def read_paths(source, store=None, ext=".root", mode="local"):
    """Get all file names of a dataset, which can be interpreted from a
    source

    Parameters:
    source -- A glob pattern, dataset name or a path to a text file containing
              any of the afore mentioned (one per line). If it ends with ext,
              it will be considered as a glob pattern.
    store -- See expand_datasetdict
    ext -- See expand_datasetdict
    mode -- See expand_datasetdict

    Returns a list of paths as strings
    """
    paths = []
    if source.endswith(ext):
        paths = glob(source)
    elif (source.count("/") == 3
            and (source.endswith("NANOAOD")
                 or source.endswith("NANOAODSIM")
                 or source.endswith("USER"))):
        paths.extend(dataset_to_lfns(source, store, ext, mode))
    else:
        with open(source) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith(store):
                    paths_from_line = glob(line)
                else:
                    paths_from_line = dataset_to_lfns(line, store, ext, mode)
                num_files = len(paths_from_line)
                if num_files == 0:
                    logger.warning("No files found for \"{}\"".format(line))
                else:
                    logger.info("Found {} file{} for \"{}\"".format(
                        num_files, "s" if num_files > 1 else "", line))
                    paths.extend(paths_from_line)
    return paths


def expand_datasetdict(datasets, store=None, ignore_path=None, ext=".root",
                       mode="local"):
    """Interpred a dict of dataset names or paths

    Parameters:
    datasets -- A dict whose values are lists of glob patterns, dataset names
                or files containing any of the afore mentioned
    store -- Path to the store directory, e.g. /pnfs/desy.de/cms/tier2/store/
    ignore_path -- Callable of the form file path -> bool. If it evaluates not
                   to True, the file path is skipped for the output. If None,
                   no files are skipped
    ext -- File extension the files have
    mode -- One of 'local', 'xrootd', 'local+xrootd'. If datasets contain
            dataset names, this defines how to handle them. With 'local' they
            will evaluate to local file paths. If 'xrootd' they will evaluate
            to xrootd URLs. If 'local+xrootd' only files that are not present
            locally are returned with an xrootd URL, otherwise local file paths
            are returned.

    Returns a tuple of two dicts. The first one is a dict mapping the keys of
    `datasets` to lists of paths for the corresponding files. The second one is
    the inverse mapping.
    """
    paths2dsname = {}
    datasetpaths = defaultdict(list)
    for key in datasets.keys():
        paths = list(dict.fromkeys([
            a for b in datasets[key] for a in read_paths(
                b, store, ext, mode)]))
        if ignore_path:
            processed_paths = []
            for path in paths:
                if not ignore_path(path):
                    processed_paths.append(path)
            paths = processed_paths

        for path in paths:
            if path in paths2dsname:
                raise RuntimeError(
                    f"Path '{path}' is found to belong to more than one "
                    f"dataset: {key} and {paths2dsname[path]}")
            datasetpaths[key].append(path)
            paths2dsname[path] = key

    return datasetpaths, paths2dsname
