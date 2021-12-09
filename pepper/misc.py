import os
from glob import glob
import inspect
import gc
import json
from collections import namedtuple
from collections.abc import Mapping
from functools import wraps, partial
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import awkward as ak
import coffea
import hist as hi
import parsl
import parsl.addresses


def normalize_trigger_path(path):
    # Remove HLT and L1 prefixes
    if path.startswith("HLT_"):
        path = path[4:]
    elif path.startswith("L1_"):
        path = path[3:]
    # Remove _v suffix
    if path.endswith("_v"):
        path = path[:-2]
    return path


def get_trigger_paths_for(dataset, is_mc, trigger_paths, trigger_order=None,
                          normalize=True, era=None):
    """Get trigger paths needed for the specific dataset.

    Arguments:
    dataset -- Name of the dataset
    trigger_paths -- dict mapping dataset names to their triggers
    trigger_order -- list of datasets to define the order in which the triggers
                     are applied.
    normalize -- bool, whether to remove HLT_ from the beginning
    era -- None or a string. If not None and if <name>_era, where name is any
           dataset name, is present in `trigger_paths`, it will be used over
           just <name>. This can be used to define per era triggers.

    Returns a tuple of lists (pos_triggers, neg_triggers) describing trigger
    paths to include and to exclude respectively.
    """
    if isinstance(trigger_order, dict):
        if era in trigger_order.keys():
            trigger_order = trigger_order[era]
        else:
            trigger_order = trigger_order["other"]
    pos_triggers = []
    neg_triggers = []
    if is_mc:
        for key in trigger_order:
            pos_triggers.extend(trigger_paths[key])
    else:
        for key in trigger_order:
            if (key == dataset) or (
                era is not None and key == dataset + "_" + era
            ):
                break
            neg_triggers.extend(trigger_paths[key])
        else:
            raise ValueError(f"Dataset {dataset} not in trigger_order")
        pos_triggers = trigger_paths[key]
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


def coffeahist2hist(hist):
    warnings.warn(
        "coffeahist2hist is deprecated, use coffea.hist.Hist.to_hist instead",
        DeprecationWarning
    )
    axes = []
    cat_pos = []
    for i, axis in enumerate(hist.axes()):
        if isinstance(axis, coffea.hist.Cat):
            identifiers = [idn.name for idn in axis.identifiers()]
            axes.append(hi.axis.StrCategory(identifiers, name=axis.name,
                                            label=axis.label))
            cat_pos.append(i)
        elif isinstance(axis, coffea.hist.Bin):
            edges = axis.edges()
            is_uniform = np.unique(np.diff(edges)).size == 1
            if is_uniform:
                axes.append(hi.axis.Regular(
                    edges.size - 1, edges[0], edges[-1], name=axis.name,
                    label=axis.label, overflow=True, underflow=True))
            else:
                axes.append(hi.axis.Variable(
                    edges, name=axis.name, label=axis.label, overflow=True,
                    underflow=True))
        else:
            raise ValueError(f"Axis of unknown type: {axis}")
    ret = hi.Hist(*axes, storage=hi.storage.Weight())
    idx = [slice(None)] * len(axes)
    for key, (sumw, sumw2) in hist.values(
            overflow="allnan", sumw2=True).items():
        for i, idn in zip(cat_pos, key):
            idx[i] = idn
        # Sum NaN bins to overflow bins
        for i in range(len(axes) - len(cat_pos)):
            sumw[(np.s_[:],) * i + (-2,)] += sumw[(np.s_[:],) * i + (-1,)]
            sumw2[(np.s_[:],) * i + (-2,)] += sumw2[(np.s_[:],) * i + (-1,)]
        # Remove NaN bins
        sumw = sumw[(np.s_[:-1],) * sumw.ndim]
        sumw2 = sumw2[(np.s_[:-1],) * sumw2.ndim]

        ret[tuple(idx)] = np.stack([sumw, sumw2], axis=-1)
    return ret


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
    """Get a parsl HTCondor config for a host.

    Arguments:
    num_jobs -- Number of jobs/processes to run in parallel
    runtime -- Requested runtime in seconds. If None, do not request a runtime
    memory -- Request memory in MB. If None, do not request memory
    retries -- The number of times to retry a failed task. If None, the task is
               retried until it stops failing
    hostname -- hostname of the machine to submit from. If None, use current
    condor_submit -- String that gets appended to the Condor submit file
    condor_init -- String containing Shell commands to be executed by every job
                   upon startup to setup an envrionment. If None, try to read
                   the file pointed at by the local environment variable
                   PEPPER_CONDOR_ENV and its contents instead. If
                   PEPPER_CONDOR_ENV is also not set, no futher environment
                   will be set up.
    """
    if num_jobs > 450:
        raise ValueError(
            "Due to technical limitations only up to 450 jobs are possible "
            "right now")
    if hostname is None:
        hostname = parsl.addresses.address_by_hostname()
    if retries is None:
        # Actually parsl doesn't support infinite retries so set it very high
        retries = 1000000
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
    if condor_init is None and "PEPPER_CONDOR_ENV" in os.environ:
        with open(os.environ["PEPPER_CONDOR_ENV"]) as f:
            condor_init = f.read()
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
                  "--hb_threshold={heartbeat_threshold} "
                  "--cpu-affinity {cpu_affinity} ")
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


def chunked_calls(array_param, returns_multiple=False, chunksize=10000,
                  num_threads=1):
    """A decorator that will split a function call into multiple calls on
    smaller chunks of data. This can be used to reduce peak memory usage or
    to paralellzie the call.
    Only arguments that have the attribute shape and have the same size in
    their first dimension as the value of the parameter named by array_param
    will get chunked.
    The return values of each call will be concatenated.
    The resulting functions will have two additional parameters, chunksize and
    num_threads. For a description see below.

    Arguments:
    array_param -- Parameter that will defninitely be a chunkable argument
    returns_multiple -- Needs to be set to true if the function returns more
                        than one variable, e.g. as a tuple or list
    chunksize -- Default maximum chunk size to call the function on. The
                 chunksize can be adjusted by using the keyword argument
                 `chunksize` of the resulting function.
    num_threads -- Number of simultaneous threads. Each thread processes one
                   chunk at a time, allowing to process multiple chunks in
                   parallel and on multiple cores. The number of threads can be
                   adjusted by using the keyword argument `num_threads` of the
                   resulting function.
    """

    def concatenate(arrays):
        if isinstance(arrays[0], np.ndarray):
            return np.concatenate(arrays)
        else:
            return ak.concatenate(arrays)

    def decorator(func):
        sig = inspect.signature(func)

        def do_work(kwargs, array_parameters, start, stop):
            chunked_kwargs = kwargs.copy()
            for param in array_parameters:
                chunked_kwargs[param] = kwargs[param][start:stop]
            return func(**chunked_kwargs)

        @wraps(func)
        def wrapper(*args, chunksize=chunksize, num_threads=num_threads,
                    **kwargs):
            kwargs = sig.bind(*args, **kwargs).arguments
            rows = len(kwargs[array_param])
            if rows <= chunksize:
                # Nothing to chunk, just return whatever func returns
                return func(**kwargs)
            if num_threads > 1:
                pool = ThreadPoolExecutor(max_workers=num_threads)
            array_parameters = {array_param}
            for param, arg in kwargs.items():
                if hasattr(arg, "__len__") and len(arg) == rows:
                    array_parameters.add(param)
            starts = np.arange(0, rows, chunksize)
            stops = np.r_[starts[1:], rows]
            result_funcs = []
            for start, stop in zip(starts, stops):
                if num_threads > 1:
                    result_funcs.append(pool.submit(
                        do_work, kwargs, array_parameters, start, stop).result)
                else:
                    result_funcs.append(partial(
                        do_work, kwargs, array_parameters, start, stop))
            ret_chunks = []
            for result_func in result_funcs:
                ret_chunk = result_func()
                if ret_chunk is None:
                    if num_threads > 1:
                        pool.shutdown(cancel_futures=True)
                    return None
                ret_chunks.append(ret_chunk)
                # Force clean up of memory to keep usage low
                gc.collect()
            if num_threads > 1:
                pool.shutdown()
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


def onedimeval(func, *arrays, tonumpy=True, output_like=0):
    """Evaluate the callable `func` on the flattened versions of arrays. These
    are converted into numpy arrays if `tonumpy` is true. The return value is
    the result of `func` converted into an awkward array, unflattened and with
    the parameters and behavior of the array at position `output_like`.
    """
    counts_all_arrays = []
    flattened_arrays = []
    for array in arrays:
        flattened = array
        counts = []
        for i in range(flattened.ndim - 1):
            if isinstance(flattened.type.type, ak.types.RegularType):
                counts.append(flattened.type.type.size)
            else:
                counts.append(ak.num(flattened))
            flattened = ak.flatten(flattened)
        if tonumpy:
            flattened = np.asarray(flattened)
        counts_all_arrays.append(counts)
        flattened_arrays.append(flattened)
    res = func(*flattened_arrays)
    for count in reversed(counts_all_arrays[output_like]):
        res = ak.unflatten(res, count)
    for name, val in ak.parameters(arrays[output_like]).items():
        res = ak.with_parameter(res, name, val)
    res.behavior = arrays[output_like].behavior
    return res


def akremask(array, mask):
    if ak.sum(mask) != len(array):
        raise ValueError(f"Got array of length {len(array)} but mask needs "
                         f"{ak.sum(mask)}")
    if len(array) == 0:
        return ak.pad_none(array, len(mask), axis=0)
    offsets = np.cumsum(np.asarray(mask)) - 1
    return ak.mask(array[offsets], mask)


class VirtualArrayCopier:
    """Create a shallow copy of the an awkward Array such as NanoEvents
    while trying to not make virtual subarrays load their contents.
    """
    def __init__(self, array):
        self.data = {f: array[f] for f in ak.fields(array)}
        self.behavior = array.behavior

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __delitem__(self, key):
        del self.data[key]

    def get(self):
        array = ak.Array(self.data)
        array.behavior = self.behavior
        return array

    def wrap_with_copy(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(self.get(), *args, **kwargs)
        return wrapper


def akismasked(arr):
    """Return true if arr is masked on any axis, otherwise false"""
    t = arr
    while hasattr(t, "type"):
        if isinstance(t.type, ak.types.OptionType):
            return True
        t = t.type
    return False


class HistCollection(dict):
    class Key(namedtuple(
            "HistCollectionKeyBase", ["cut", "hist", "variation"])):
        def __new__(cls, cut=None, hist=None, variation=None):
            return cls.__bases__[0].__new__(cls, cut, hist, variation)

        def fitsto(self, **kwargs):
            for key, value in kwargs.items():
                if getattr(self, key) != value:
                    return False
            else:
                return True

    def __init__(self, *args, **kwargs):
        self._path = kwargs["path"]
        del kwargs["path"]
        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, fileobj):
        data = json.load(fileobj)
        path = os.path.dirname(os.path.realpath(fileobj.name))
        return cls({cls.Key(*k): v for k, v in zip(*data)}, path=path)

    def __getitem__(self, key):
        if isinstance(key, self.Key):
            return super().__getitem__(key)
        elif isinstance(key, Mapping):
            ret = self.__class__({k: v for k, v in self.items()
                                  if k.fitsto(**key)}, path=self._path)
            if len(ret) == 0:
                raise KeyError(key)
            elif len(key) == len(self.Key._fields):
                ret = next(iter(ret.values()))
            return ret
        elif isinstance(key, tuple):
            return self[dict(zip(self.Key._fields, key))]
        else:
            return self[{self.Key._fields[0]: key}]

    def load(self, key):
        return coffea.util.load(os.path.join(self._path, self[key]))
