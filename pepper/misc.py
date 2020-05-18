import os
import sys
from glob import glob
import h5py
import uproot
from collections import defaultdict
from coffea.analysis_objects import JaggedCandidateArray as Jca
import awkward
import numpy as np
import parsl
import parsl.addresses


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
            if factors is not None:
                weight_chunks.append(tree["weight"] * factors[group])
            else:
                weight_chunks.append(tree["weight"])
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


def export(hist):
    """Export a one, two or three dimensional `Hist` to a ROOT histogram"""
    d = hist.dense_dim()
    if d > 3:
        raise ValueError("export() only supports up to three dense dimensions")
    if hist.sparse_dim() != 0:
        raise ValueError("export() expects zero sparse dimensions")

    axes = hist.axes()

    if d == 1:
        from uproot_methods.classes.TH1 import Methods
    elif d == 2:
        from uproot_methods.classes.TH2 import Methods
    else:
        from uproot_methods.classes.TH3 import Methods

    class TH(Methods, list):
        pass

    class TAxis(object):
        def __init__(self, fNbins, fXmin, fXmax):
            self._fNbins = fNbins
            self._fXmin = fXmin
            self._fXmax = fXmax

    out = TH.__new__(TH)
    axisattrs = ["_fXaxis", "_fYaxis", "_fZaxis"][:d]

    values_of = hist.values(sumw2=True, overflow="all")  # with overflow
    values_noof = hist.values()  # no overflow
    if len(values_of) == 0:
        sumw_of = sumw2_of = np.zeros(tuple(axis.size - 1 for axis in axes))
        sumw_noof = np.zeros(tuple(axis.size - 3 for axis in axes))
    else:
        sumw_of, sumw2_of = values_of[()]
        sumw_noof = values_noof[()]
    centers = []
    for axis, axisattr in zip(axes, axisattrs):
        edges = axis.edges(overflow="none")

        taxis = TAxis(len(edges) - 1, edges[0], edges[-1])
        taxis._fName = axis.name
        taxis._fTitle = axis.label
        if not axis._uniform:
            taxis._fXbins = edges.astype(">f8")
        setattr(out, axisattr, taxis)
        centers.append((edges[:-1] + edges[1:]) / 2.0)

    out._fEntries = out._fTsumw = out._fTsumw2 = sumw_noof.sum()

    projected_x = sumw_noof.sum((1, 2)[:d - 1])
    out._fTsumwx = (projected_x * centers[0]).sum()
    out._fTsumwx2 = (projected_x * centers[0]**2).sum()
    if d >= 2:
        projected_y = sumw_noof.sum((0, 2)[:d - 1])
        projected_xy = sumw_noof.sum((2,)[:d - 2])
        out._fTsumwy = (projected_y * centers[1]).sum()
        out._fTsumwy2 = (projected_y * centers[1]**2).sum()
        out._fTsumwxy = ((projected_xy * centers[1]).sum(1) * centers[0]).sum()
    if d == 3:
        projected_z = sumw_noof.sum((0, 1)[:d - 1])
        projected_xz = sumw_noof.sum((1,)[:d - 2])
        projected_yz = sumw_noof.sum((0,)[:d - 2])
        out._fTsumwz = (projected_z * centers[2]).sum()
        out._fTsumwz2 = (projected_z * centers[2]**2).sum()
        out._fTsumwxz = ((projected_xz * centers[2]).sum(1) * centers[0]).sum()
        out._fTsumwyz = ((projected_yz * centers[2]).sum(1) * centers[1]).sum()

    out._fName = "histogram"
    out._fTitle = hist.label

    if d == 1:
        out._classname = b"TH1D"
    elif d == 2:
        out._classname = b"TH2D"
    else:
        out._classname = b"TH3D"
    out.extend(sumw_of.astype(">f8").transpose().flatten())
    out._fSumw2 = sumw2_of.astype(">f8").transpose().flatten()

    return out


def hist_divide(num, denom):
    """Return a histogram with bin heights = num / denum and errors set
    accordingly"""
    if not num.compatible(denom):
        raise ValueError("Cannot divide this histogram {} with histogram {} "
                         "of dissimilar dimensions".format(num, denom))
    hout = num.copy()
    hout.label = "Ratio"

    raxes = denom.sparse_axes()

    def div(a, b):
        out = np.zeros_like(a)
        nz = b != 0
        out[nz] = a[nz] / b[nz]
        return out

    def diverr2(a, b, da2, db2):
        out = np.zeros_like(a)
        nz = b != 0
        out[nz] = (da2[nz] * b[nz]**2 + db2[nz] * a[nz]**2) / b[nz]**4
        return out

    denomsumw2 = denom._sumw2 if denom._sumw2 is not None else denom._sumw
    for rkey in denom._sumw.keys():
        lkey = tuple(num.axis(rax).index(rax[ridx])
                     for rax, ridx in zip(raxes, rkey))
        if lkey in hout._sumw:
            hout._sumw2[lkey] = diverr2(hout._sumw[lkey],
                                        denom._sumw[rkey],
                                        hout._sumw2[lkey],
                                        denomsumw2[rkey])
            hout._sumw[lkey] = div(hout._sumw[lkey], denom._sumw[rkey])
        else:
            hout._sumw2[lkey] = np.zeros_like(denomsumw2[rkey])
            hout._sumw[lkey] = np.zeros_like(denom._sumw[rkey])
    return hout


def jcafromjagged(**fields):
    """Create JaggedCandidateArray from JaggedArrays
    This eliminates the need to flatten every JaggedArray.
    """
    counts = None
    flattened = {}
    for key, val in fields.items():
        if counts is None:
            counts = val.counts
        elif (counts != val.counts).any():
            raise ValueError("Got JaggedArrays of different sizes "
                             "({counts} and {val.counts})")
        flattened[key] = val.flatten()
    return Jca.candidatesfromcounts(counts, **flattened)


def jaggeddepth(arr):
    """Get the number of jagged dimensions of a JaggedArray"""
    depth = 0
    while not isinstance(arr, (awkward.Table, np.ndarray)):
        depth += 1
        arr = arr.content
    return depth


def sortby(table, field, ascending=False):
    """Sort a table by a field or attribute"""
    try:
        return table[table[field].argsort(ascending=ascending)]
    except KeyError:
        return table[getattr(table, field).argsort(ascending=ascending)]


def hist_counts(hist):
    """Get the number of entries in a histogram, including all overflow and
    nan
    """
    values = hist.sum(*hist.axes(), overlow="allnan").values()
    if len(values) == 0:
        return 0
    return next(iter(values.values()))


def get_parsl_config(num_jobs, runtime=3*60*60, hostname=None):
    """Get a parsl config for a host.

    Arguments:
    num_jobs -- Number of jobs/processes to run in parallel
    runtime -- Requested runtime in seconds. If None, do not request a runtime
    hostname -- hostname of the machine to submit from. If None, use current
    """
    if hostname is None:
        hostname = parsl.addresses.address_by_hostname()
    scriptdir = sys.path[0]
    pythonpath = os.environ["PYTHONPATH"]
    condor_config = ("requirements = (OpSysAndVer == \"SL6\" || OpSysAndVer =="
                     " \"CentOS7\")\n")
    if runtime is not None:
        if hostname.endswith(".desy.de"):
            condor_config += f"+RequestRuntime = {runtime}\n"
        elif hostname.endswith(".cern.ch"):
            condor_config += f"+MaxRuntime = {runtime}\n"
        else:
            raise NotImplementedError(f"runtime on unknown host {hostname}")
    # Need to unset PYTHONPATH because of DESY NAF setting it incorrectly
    condor_init = """
source /cvmfs/cms.cern.ch/cmsset_default.sh
if lsb_release -r | grep -q 7\\.; then
cd /cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
else
cd /cvmfs/cms.cern.ch/slc6_amd64_gcc700/cms/cmssw-patch/CMSSW_10_2_4_patch1/src
fi
eval `scramv1 runtime -sh`
cd -
"""
    # Need to put own directory into PYTHONPATH for unpickling to work.
    # Need to extend PATH to be able to execute the main parsl script.
    condor_init += f"export PYTHONPATH={scriptdir}:{pythonpath}\n"
    condor_init += "PATH=~/.local/bin:$PATH"
    provider = parsl.providers.CondorProvider(
        init_blocks=num_jobs,
        max_blocks=num_jobs,
        scheduler_options=condor_config,
        worker_init=condor_init
    )
    parsl_executor = parsl.executors.HighThroughputExecutor(
        label="HTCondor",
        address=hostname,
        max_workers=1,
        provider=provider,
    )
    parsl_config = parsl.config.Config(
        executors=[parsl_executor],
        # Set retries to a large number to retry infinitely
        retries=100000,
    )
    return parsl_config
