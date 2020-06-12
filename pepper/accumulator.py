#!/usr/bin/env python3

from coffea.processor import PackedSelection, AccumulatorABC
import numpy as np
from copy import copy
import awkward


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
        return ArrayAccumulator()

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
