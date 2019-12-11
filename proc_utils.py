from collections import OrderedDict
from copy import copy

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray as Jca
from coffea.util import awkward
from coffea.util import numpy as np
import uproot_methods
from coffea.processor import PackedSelection
from coffea.processor import AccumulatorABC


class PackedSelectionAccumulator(PackedSelection, AccumulatorABC):
    def add_cut(self, name, selection):
        return super(PackedSelectionAccumulator, self).add(name, selection)

    def identity(self):
        return PackedSelectionAccumulator(dtype=self._dtype)

    def add(self, other):
        if len(other.names) == 0:
            return self
        elif len(self.names) == 0:
            self._mask = other._mask
            self._names = other._names
            return self
        if self.names != other.names:
            if len(self.names) > len(other.names):
                other = self._extend(other, self)
            elif len(other.names) < len(self.names):
                self = self._extend(self, other)
            else:
                raise ValueError("Names of cuts do not match")
        self._mask = np.concatenate((self._mask, other._mask))
        return self

    @staticmethod
    def _extend(short_names, long_names):
        if long_names.names[:len(short_names.names)] == short_names.names:
            for name in long_names.names[len(short_names.names):]:
                short_names.add_cut(name,
                                    np.full(len(short_names._mask), True))
        else:
            raise ValueError("Names of cuts do not match")
        return short_names

    def mask_self(self, cuts):
        masked = copy(self)
        masked._mask = masked._mask[cuts]
        return masked

    @property
    def mask(self):
        return self._mask


class ArrayAccumulator(AccumulatorABC):
    def __init__(self):
        self._empty = None
        self._value = None

    def __repr__(self):
        return "ArrayAccumulator(%r)" % self.value

    def identity(self):
        return array_accumulator(self._empty)

    def add(self, other):
        if self._value is None:
            self._empty = other._empty
            self.value = other.value
        elif other._value is None:
            pass
        elif (isinstance(self.value, np.ndarray) &
              isinstance(other.value, np.ndarray)):
            if other._empty.shape != self._empty.shape:
                raise ValueError(
                    "Cannot add two np arrays of dissimilar shape (%r vs %r)"
                    % (self._empty.shape, other._empty.shape))
            self.value = np.concatenate((self.value, other.value))
        elif (isinstance(self.value, awkward.JaggedArray) &
              isinstance(other.value, awkward.JaggedArray)):
            if other._empty.shape != self._empty.shape:
                raise ValueError(
                  "Cannot add two JaggedArrays of dissimilar shape (%r vs %r)"
                  % (self._empty.shape, other._empty.shape))
            self.value = awkward.JaggedArray.concatenate(
                                    (self.value, other.value))
        else:
            raise ValueError("Cannot add %r to %r"
                             % (type(other._value), type(self._value)))

    @property
    def value(self):
        '''The current value of the column
        Returns a JaggedArray where the first dimension is the column dimension
        '''
        return self._value

    @value.setter
    def value(self, val):
        if (self._value is not None) & (type(self.value) != type(val)):
            raise ValueError("Cannot change type of value once set!")
        elif isinstance(val, np.ndarray):
            self._empty = np.zeros(dtype=val.dtype, shape=(0,) + val.shape[1:])
            self._value = val
        elif isinstance(val, awkward.JaggedArray):
            self._empty = np.zeros(dtype=val.dtype, shape=(0,) + val.shape[1:])
            self._value = val
        else:
            raise ValueError(
                "Must set value with either a np array or JaggedArray, not %r"
                % type(val))


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
