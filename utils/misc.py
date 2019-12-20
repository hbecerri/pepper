import os
from glob import glob
import h5py
import uproot
from collections import defaultdict
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray as Jca
import awkward
import numpy as np


def concatenate(arr1, arr2):
    keys = ["pt", "eta", "phi", "mass"]
    arr_dict = {}
    offsets = awkward.concatenate([arr1.pt, arr2.pt], axis=1).offsets
    arr_dict["pt"] = \
        awkward.concatenate([arr1.pt, arr2.pt], axis=1).flatten()
    arr_dict["eta"] = \
        awkward.concatenate([arr1.eta, arr2.eta], axis=1).flatten()
    arr_dict["phi"] = \
        awkward.concatenate([arr1.phi, arr2.phi], axis=1).flatten()
    arr_dict["mass"] = \
        awkward.concatenate([arr1.mass, arr2.mass], axis=1).flatten()
    if "pdgId" in arr1.flatten().contents:
        arr_dict["pdgId"] = awkward.concatenate([arr1["pdgId"], arr2["pdgId"]],
                                                axis=1).flatten()
    return Jca.candidatesfromoffsets(offsets, **arr_dict)


def pairswhere(condition, x, y):
    counts = np.where(condition, x.counts, y.counts)
    pt0 = np.empty(counts.sum(), dtype=float)
    eta0 = np.empty(counts.sum(), dtype=float)
    phi0 = np.empty(counts.sum(), dtype=float)
    mass0 = np.empty(counts.sum(), dtype=float)

    pt1 = np.empty(counts.sum(), dtype=float)
    eta1 = np.empty(counts.sum(), dtype=float)
    phi1 = np.empty(counts.sum(), dtype=float)
    mass1 = np.empty(counts.sum(), dtype=float)

    offsets = awkward.JaggedArray.counts2offsets(counts)
    starts, stops = offsets[:-1], offsets[1:]

    working_array = \
        np.zeros(counts.sum()+1, dtype=awkward.JaggedArray.INDEXTYPE)
    xstarts = starts[condition]
    xstops = stops[condition]
    not_empty = xstarts != xstops
    working_array[xstarts[not_empty]] += 1
    working_array[xstops[not_empty]] -= 1
    mask = np.array(np.cumsum(working_array)[:-1],
                    dtype=awkward.JaggedArray.MASKTYPE)

    pt0[mask] = x[condition].i0.pt.flatten()
    pt0[~mask] = y[~condition].i0.pt.flatten()
    eta0[mask] = x[condition].i0.eta.flatten()
    eta0[~mask] = y[~condition].i0.eta.flatten()
    phi0[mask] = x[condition].i0.phi.flatten()
    phi0[~mask] = y[~condition].i0.phi.flatten()
    mass0[mask] = x[condition].i0.mass.flatten()
    mass0[~mask] = y[~condition].i0.mass.flatten()
    out0 = Jca.candidatesfromcounts(
            counts, pt=pt0, eta=eta0, phi=phi0, mass=mass0)

    pt1[mask] = x[condition].i1.pt.flatten()
    pt1[~mask] = y[~condition].i1.pt.flatten()
    eta1[mask] = x[condition].i1.eta.flatten()
    eta1[~mask] = y[~condition].i1.eta.flatten()
    phi1[mask] = x[condition].i1.phi.flatten()
    phi1[~mask] = y[~condition].i1.phi.flatten()
    mass1[mask] = x[condition].i1.mass.flatten()
    mass1[~mask] = y[~condition].i1.mass.flatten()
    out1 = Jca.candidatesfromcounts(
            counts, pt=pt1, eta=eta1, phi=phi1, mass=mass1)
    return out0, out1


def jaggedlike(j, content):
    return awkward.JaggedArray(j.starts, j.stops, content)


def get_trigger_paths_for(dataset, is_mc, trigger_paths, trigger_order=None):
    """Get trigger paths needed for the specific dataset.

    Arguments:
    dataset -- Name of the dataset
    trigger_paths -- dict mapping dataset names to their triggers
    trigger_order -- List of datasets to define the order in which the triggers
                     are applied.

    Returns a tuple of lists (pos_triggers, neg_triggers) describing trigger
    paths to include and to exclude respectively.
    """
    pos_triggers = []
    neg_triggers = []
    if is_mc:
        for paths in trigger_paths.values():
            pos_triggers.extend(paths)
    else:
        for dsname in trigger_order:
            if dsname == dataset:
                break
            neg_triggers.extend(trigger_paths[dsname])
        pos_triggers = trigger_paths[dataset]
    return list(dict.fromkeys(pos_triggers)), list(dict.fromkeys(neg_triggers))


def jagged_reduce(jarr):
    """Remove any unused content from a JaggedArray, making it ready for
    saving"""
    if isinstance(jarr, np.ndarray):
        return jarr
    cls = awkward.array.objects.Methods.maybemixin(type(jarr),
                                                   awkward.JaggedArray)
    return cls.fromcounts(jarr.counts, jarr.flatten())


def get_event_files(eventdir, eventext, datasets):
    out = {}
    for dsname in datasets:
        out[dsname] = glob(os.path.join(eventdir, dsname, "*" + eventext))
    return out


def treeopen(path, treepath, branches):
    treedata = {}
    if path.endswith(".root"):
        f = uproot.open(path)
        tree = f[treepath]
        for branch in branches:
            treedata[branch] = tree[branch].array()
    elif path.endswith(".hdf5") or path.endswith(".h5"):
        f = h5py.File(path, "r")
        tree = awkward.hdf5(f)
        for branch in branches:
            treedata[branch] = tree[branch]
    else:
        raise RuntimeError("Cannot open {}. Unknown extension".format(path))
    return awkward.Table(treedata)


def montecarlo_iterate(datasets, factors, branches, treepath="Events"):
    for group, group_paths in datasets.items():
        chunks = defaultdict(list)
        weight_chunks = []
        weight = np.array([])
        for path in group_paths:
            tree = treeopen(path, treepath, list(branches) + ["weight"])
            for branch in branches:
                if tree[branch].size == 0:
                    continue
                chunks[branch].append(tree[branch])
            weight_chunks.append(tree["weight"] * factors[group])
        if len(weight_chunks) == 0:
            continue
        data = {}
        for branch in chunks.keys():
            data[branch] = awkward.concatenate(chunks[branch])

        yield group, np.concatenate(weight_chunks), awkward.Table(data)


def expdata_iterate(datasets, branches, treepath="Events"):
    for dsname, paths in datasets.items():
        chunks = defaultdict(list)
        for path in paths:
            data = treeopen(path, treepath, branches)
            if data.size == 0:
                continue
            for branch in branches:
                chunks[branch].append(data[branch])
        if len(chunks) == 0:
            continue
        data = {}
        for branch in branches:
            data[branch] = awkward.concatenate(chunks[branch])
        yield dsname, awkward.Table(data)
