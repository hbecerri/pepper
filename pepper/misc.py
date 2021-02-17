import os
import sys
from glob import glob
import awkward as ak
import coffea
import numpy as np
import parsl
import parsl.addresses
from functools import wraps
import inspect
import gc


def normalize_trigger_path(path):
    if path.startswith("HLT_"):
        path = path[4:]
    return path


def get_trigger_paths_for(dataset, is_mc, trigger_paths, trigger_order=None,
                          normalize=True):
    """Get trigger paths needed for the specific dataset.

    Arguments:
    dataset -- Name of the dataset
    trigger_paths -- dict mapping dataset names to their triggers
    trigger_order -- list of datasets to define the order in which the triggers
                     are applied.
    normalize -- bool, whether to remove HLT_ from the beginning

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
    pos_triggers = list(dict.fromkeys(pos_triggers))
    neg_triggers = list(dict.fromkeys(neg_triggers))
    if normalize:
        pos_triggers = [normalize_trigger_path(t) for t in pos_triggers]
        neg_triggers = [normalize_trigger_path(t) for t in neg_triggers]
    return pos_triggers, neg_triggers


def get_event_files(eventdir, eventext, datasets):
    out = {}
    for dsname in datasets:
        out[dsname] = glob(os.path.join(eventdir, dsname, "*" + eventext))
    return out


def export(hist):
    """Export a one, two or three dimensional `Hist` to a uproot3 histogram"""
    d = hist.dense_dim()
    if d > 3:
        raise ValueError("export() only supports up to three dense dimensions")
    if hist.sparse_dim() != 0:
        raise ValueError("export() expects zero sparse dimensions")

    axes = hist.axes()

    if d == 1:
        from uproot3_methods.classes.TH1 import Methods
    elif d == 2:
        from uproot3_methods.classes.TH2 import Methods
    else:
        from uproot3_methods.classes.TH3 import Methods

    class TH(Methods, list):
        pass

    class TAxis(object):
        def __init__(self, fnbins, fxmin, fxmax):
            self._fNbins = fnbins
            self._fXmin = fxmin
            self._fXmax = fxmax

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


def export_with_sparse(hist):
    ret = {}
    for key in hist.values().keys():
        hist_integrated = hist
        for sparse_axis, keypart in zip(hist.sparse_axes(), key):
            hist_integrated = hist_integrated.integrate(sparse_axis, keypart)
        ret[key] = export(hist_integrated)
    return ret


def rootimport(uproothist, enconding="utf-8"):
    """The inverse of export. Takes an uproot3 histogram and converts it to a
    coffea histogram. `encoding` is the coding with which the labels of the
    uproot histogram are decoded."""
    axes = []
    edges = uproothist.edges
    if not isinstance(edges, tuple):
        edges = (edges,)
    axes = []
    attrs = ("_fXaxis", "_fYaxis", "_fZaxis")
    for i, (attr, edges_per_axis) in enumerate(zip(attrs, edges)):
        uprootaxis = getattr(uproothist, attr)
        title = uprootaxis._fTitle
        if isinstance(title, bytes):
            title = title.decode(enconding)
        name = uprootaxis._fName
        if isinstance(name, bytes):
            name = name.decode(enconding)
        axis = coffea.hist.Bin(name, title, edges_per_axis)
        axes.append(axis)
    title = uproothist.title
    if isinstance(title, bytes):
        title = title.decode(enconding)
    hist = coffea.hist.Hist(title, *axes)
    ndim = len(edges)
    # Add NaN bins
    values = np.pad(uproothist.allvalues, ((0, 1),) * ndim).astype(float)
    sumw2 = np.pad(uproothist.allvariances, ((0, 1),) * ndim).astype(float)

    hist._sumw = {tuple(): values}
    hist._sumw2 = {tuple(): sumw2}
    return hist


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


def hist_counts(hist):
    """Get the number of entries in a histogram, including all overflow and
    nan
    """
    values = hist.sum(*hist.axes(), overlow="allnan").values()
    if len(values) == 0:
        return 0
    return next(iter(values.values()))


def get_parsl_config(num_jobs, runtime=3*60*60, memory=None, retries=None,
                     hostname=None, *, condor_submit=None, condor_init=None):
    """Get a parsl config for a host.

    Arguments:
    num_jobs -- Number of jobs/processes to run in parallel
    runtime -- Requested runtime in seconds. If None, do not request a runtime
    memory -- Request memory in MB. If None, do not request memory
    retries -- The number of times to retry a failed task. If None, the task is
               retried until it stops failing
    hostname -- hostname of the machine to submit from. If None, use current
    condor_submit -- String that gets appended to the Condor submit file
    condor_init -- Overwrite default environment setup on Condor node. This
                   needs to be a string containing Bash commands
    """
    if hostname is None:
        hostname = parsl.addresses.address_by_hostname()
    if retries is None:
        # Actually parsl doesn't support infinite retries so set it very high
        retries = 1000000
    scriptdir = os.path.realpath(sys.path[0])
    if "PYTHONPATH" in os.environ:
        pythonpath = os.environ["PYTHONPATH"]
    else:
        pythonpath = ""
    condor_config = ""
    if runtime is not None:
        if hostname.endswith(".desy.de"):
            condor_config += f"+RequestRuntime = {runtime}\n"
        elif hostname.endswith(".cern.ch"):
            condor_config += f"+MaxRuntime = {runtime}\n"
        else:
            raise NotImplementedError(
                    f"runtime on unknown host {hostname}")
    if memory is not None:
        condor_config += f"RequestMemory = {memory}\n"
    if condor_submit is not None:
        condor_config += condor_submit
    if condor_init is None:
        condor_init = """
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_1_0_pre5_PY3/src
eval `scramv1 runtime -sh`
cd -
"""
        # Need to put own directory into PYTHONPATH for unpickling to work.
        condor_init += f"export PYTHONPATH={scriptdir}:{pythonpath}\n"
    provider = parsl.providers.CondorProvider(
        init_blocks=min(5, num_jobs),
        max_blocks=num_jobs,
        parallelism=1,
        scheduler_options=condor_config,
        worker_init=condor_init
    )
    launch_cmd = ("python3 "
                  "~/.local/bin/process_worker_pool.py "
                  "{debug} "
                  "{max_workers} "
                  "-a {addresses} "
                  "-p {prefetch_capacity} "
                  "-c {cores_per_worker} "
                  "-m {mem_per_worker} "
                  "--poll {poll_period} "
                  "--task_port={task_port} "
                  "--result_port={result_port} "
                  "--logdir={logdir} "
                  "--block_id={{block_id}} "
                  "--hb_period={heartbeat_period} "
                  "{address_probe_timeout_string} "
                  "--hb_threshold={heartbeat_threshold} ")
    parsl_executor = parsl.executors.HighThroughputExecutor(
        label="HTCondor",
        launch_cmd=launch_cmd,
        address=hostname,
        max_workers=1,
        provider=provider,
    )
    parsl_config = parsl.config.Config(
        executors=[parsl_executor],
        # Set retries to a large number to retry infinitely
        retries=retries,
    )
    return parsl_config


def chunked_calls(array_param, chunksize, returns_multiple=False):
    """A decorator that will split a function call into multiple calls on
    smaller chunks of data. Only arguments that have the attribute shape and
    have the same size in their first dimension as the value of the parameter
    named by array_param will get chunked.
    The return values of each call will be concatenated.

    Arguments:
    array_param -- Parameter that will defninitely be a chunkable argument
    chunksize -- Maximum chunk size to call the function on
    returns_multiple -- Needs to be set to true if the function returns more
                        than one variable, e.g. as a tuple or list
    """

    def concatenate(arrays):
        if isinstance(arrays[0], np.ndarray):
            return np.concatenate(arrays)
        else:
            return ak.concatenate(arrays)

    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs = sig.bind(*args, **kwargs).arguments
            rows = len(kwargs[array_param])
            if rows <= chunksize:
                # Nothing to chunk, just return whatever func returns
                return func(**kwargs)
            array_parameters = {array_param}
            for param, arg in kwargs.items():
                if hasattr(arg, "__len__") and len(arg) == rows:
                    array_parameters.add(param)
            starts = np.arange(0, rows, chunksize)
            stops = np.r_[starts[1:], rows]
            ret_chunks = []
            for start, stop in zip(starts, stops):
                chunked_kwargs = kwargs.copy()
                for param in array_parameters:
                    chunked_kwargs[param] = kwargs[param][start:stop]
                ret_chunk = func(**chunked_kwargs)
                if ret_chunk is None:
                    return None
                ret_chunks.append(ret_chunk)
                # Force clean up of memory to keep usage low
                gc.collect()
            if len(ret_chunks) == 1:
                concated = ret_chunks[0]
            elif returns_multiple:
                # Have to transpose ret_chunks
                ret_chunks_t = [[] for _ in range(len(ret_chunks[0]))]
                for ret_chunk in ret_chunks:
                    for ret_split, chunk_val in zip(ret_chunk, ret_chunks_t):
                        chunk_val.append(ret_split)
                concated = tuple(concatenate(v) for v in ret_chunks_t)
            else:
                concated = concatenate(ret_chunks)
            return concated

        return wrapper
    return decorator


def onedimeval(func, array, tonumpy=True):
    flattened = array
    counts = []
    for i in range(flattened.ndim - 1):
        if isinstance(flattened.type.type, ak.types.RegularType):
            counts.append(flattened.type.type.size)
        else:
            counts.append(ak.num(flattened))
        flattened = ak.flatten(flattened)
    res = func(np.asarray(flattened) if tonumpy else flattened)
    for count in reversed(counts):
        res = ak.unflatten(res, count)
    for name, val in ak.parameters(array):
        res = ak.with_parameter(res, name, val)
    res.behavior = array.behavior
    return res


def akstriparray(array):
    """Make an awkward array use as little memory as possible.

    This is a workaround until ak.packed in implemented"""
    if len(ak.fields(array)) != 0:
        values = {f: akstriparray(array[f]) for f in ak.fields(array)}
        parameters = ak.parameters(array)
        behavior = array.behavior
        try:
            res = ak.zip(values, parameters=parameters, behavior=behavior)
        except ValueError:
            # Can not be broadcasted, values have differing offsets
            res = ak.Array(values, behavior=behavior)
            for name, val in ak.parameters(array):
                res = ak.with_parameter(res, name, val)
    else:
        res = onedimeval(lambda a: ak.flatten(a, axis=None), array, False)
    return res


def vecdot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def lorvecp3unit(lv):
    rho = lv.rho
    mask = rho == 0
    rho = ak.where(mask, 1, rho)
    v = ak.zip({
        "x": ak.where(mask, 0, lv.x / rho),
        "y": ak.where(mask, 0, lv.y / rho),
        "z": ak.where(mask, 0, lv.z / rho),
    })
    v.behavior = lv.behavior
    v = ak.with_name(v, "ThreeVector")
    return v


def lorvecboost(lv, boost):
    # Compute boost three vector
    mask = boost.t == 0
    t = ak.where(mask, 1, boost.t)
    boost = ak.zip({
        "x": ak.where(mask, 0, - boost.x / t),
        "y": ak.where(mask, 0, - boost.y / t),
        "z": ak.where(mask, 0, - boost.z / t)})
    boost.behavior = lv.behavior
    boost = ak.with_name(boost, "ThreeVector")

    # Apply boost
    b2 = boost.rho2
    gamma = (1 - b2) ** (-0.5)
    mask = b2 == 0
    b2 = ak.where(mask, 1, b2)
    gamma2 = ak.where(mask, 0, (gamma - 1) / b2)
    del mask

    bp = lv.x * boost.x + lv.y * boost.y + lv.z * boost.z
    v = gamma2 * bp * boost + lv.t * gamma * boost

    out = ak.with_name(ak.zip({
        "x": lv.x + v.x,
        "y": lv.y + v.y,
        "z": lv.z + v.z,
        "t": gamma * (lv.t + bp)}),
        "LorentzVector")
    out.behavior = lv.behavior

    return out
