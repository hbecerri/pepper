import numpy as np
import awkward
from copy import copy
from collections.abc import MutableMapping


class LazyTable(MutableMapping):

    # load lazily if a basket is loaded with less than 75% probability
    LAZYPROB = 0.75

    def __init__(self, tree, entrystart=None, entrystop=None, flatten=False,
                 slic=None):
        self._loaded = {}
        self._loaded_lazily = set()
        self._overwritten = {}
        self._tree = tree
        self._available = {k.decode() for k in self._tree.keys()}
        self._branchargs = {"flatten": flatten}
        if entrystart is None:
            entrystart = 0
        if entrystop is None:
            entrystop = len(tree)
        self._branchargs["entrystart"] = entrystart
        self._branchargs["entrystop"] = entrystop
        self._slice = slic
        self._available = {k.decode() for k in self._tree.keys()}

    @classmethod
    def from_lazydf(cls, df):
        return cls(
            df._tree, df._branchargs["entrystart"],
            df._branchargs["entrystop"], flatten=df._branchargs["flatten"])

    def _mergeslice(self, slic):
        new = copy(self)
        if self._slice is not None:
            new._slice = self._slice[slic]
        else:
            new._slice = slic
        return new

    def _maybeslice(self, arr):
        if self._slice is not None:
            arr = arr[self._slice]
        return arr

    def _get_basket_probability(self, branch):
        """Get the probablility of a basket being required to load given the
        current slice"""
        start = self.entrystart
        stop = self.entrystop
        num_baskets = 0
        for i in range(branch.numbaskets):
            if (start <= branch.basket_entrystop(i)
                    and stop >= branch.basket_entrystart(i)):
                num_baskets += 1
        num_selected = len(self)
        return 1 - (1 - 1 / num_baskets) ** num_selected

    @staticmethod
    def _lazycut(chunked, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = chunked.size
        if start >= stop:
            return awkward.ChunkedArray([], [])
        chunks = chunked.chunks
        chunksizes = chunked.chunksizes
        startid = chunked.global2chunkid(start)
        startlocal = chunked.global2local(start)[1]
        stopid = chunked.global2chunkid(stop - 1)
        stoplocal = chunked.global2local(stop - 1)[1] + 1
        if stopid == startid:
            stoplocal -= startlocal
        chunks = chunks[startid:stopid + 1]
        chunksizes = chunksizes[startid:stopid + 1]
        if startlocal != 0:
            chunks[0] = awkward.VirtualArray(
                lambda x: x, args=(chunks[0][startlocal:],))
            chunks[0].materialize()
            chunksizes[0] = chunksizes[0] - startlocal
        if stoplocal != chunksizes[-1]:
            chunks[-1] = awkward.VirtualArray(
                lambda x: x, args=(chunks[-1][:stoplocal],))
            chunks[-1].materialize()
            chunksizes[-1] = stoplocal
        return awkward.ChunkedArray(chunks, chunksizes)

    def _load_lazily(self, key, idx):
        args = self._branchargs.copy()
        args["entrystart"] = None
        args["entrystop"] = None
        chunked = self._tree[key].lazyarray(**args)
        chunked = self._lazycut(chunked, self.entrystart, self.entrystop)
        arr = awkward.concatenate([c for c in chunked[idx].chunks])
        cloaded = [c.ismaterialized for c in chunked.chunks]
        if all(cloaded):
            self._loaded[key] = awkward.concatenate(
                [c.array for c in chunked.chunks])
        else:
            self._loaded[key] = chunked
            self._loaded_lazily.add(key)
        return arr

    def __contains__(self, key):
        return key in self._overwritten or key in self._tree

    def __getitem__(self, key):
        if isinstance(key, slice):
            idx = np.arange(*key.indices(self.size))
            ret = self._mergeslice(idx)
        elif isinstance(key, np.ndarray):
            if key.ndim > 1:
                raise IndexError("too many indices for table")
            if key.dtype == np.bool:
                if key.size > self.size:
                    raise IndexError("boolean index too long")
                idx = np.nonzero(key)[0]
            elif key.dtype == np.int:
                outofbounds = key > self.size
                if any(outofbounds):
                    where = key[np.argmax(outofbounds)]
                    raise IndexError(f"index {where} is out of bounds")
                idx = key
            else:
                raise IndexError("numpy arrays used as indices must be "
                                 "of intger or boolean type")
            ret = self._mergeslice(idx)
        elif key in self._overwritten:
            ret = self._maybeslice(self._overwritten[key])
        elif key in self._loaded:
            if key in self._loaded_lazily:
                chunked = self._loaded[key]
                if self._slice is None:
                    ret = awkward.concatenate(
                        [c.array for c in chunked.chunks])
                    fullarr = ret
                else:
                    ret = awkward.concatenate(
                        [c for c in chunked[self._slice].chunks])
                    if all(c.ismaterialized for c in chunked.chunks):
                        fullarr = awkward.concatenate(
                            [c.array for c in chunked.chunks])
                    else:
                        fullarr = None
                if fullarr is not None:
                    self._loaded_lazily.remove(key)
                    self._loaded[key] = fullarr
            else:
                ret = self._maybeslice(self._loaded[key])
        elif key in self._tree:
            branch = self._tree[key]
            if (self._slice is None
                    or self._get_basket_probability(branch) >= self.LAZYPROB):
                ret = branch.array(**self._branchargs)
                self._loaded[key] = ret
                ret = self._maybeslice(ret)
            else:
                ret = self._load_lazily(key, self._slice)
        else:
            raise KeyError(key)
        return ret

    def __setitem__(self, key, value):
        if self._slice is not None:
            raise RuntimeError("Can not set a column when sliced")
        self._overwritten[key] = value

    def __delitem__(self, key):
        if key not in self.columns:
            raise KeyError(key)
        if key in self._overwritten:
            del self._overwritten[key]
        if key in self.loaded:
            del self._loaded[key]
        self._loaded_lazily.discard(key)

    def __len__(self):
        if self._slice is None:
            return self.entrystop - self.entrystart
        else:
            return self._slice.size

    def __iter__(self):
        for key in self.columns:
            yield key

    def __copy__(self):
        # Tweak copy a bit to have setitem behavior like if copying a dict
        cls = self.__class__
        newinstance = cls.__new__(cls)
        newinstance.__dict__.update(self.__dict__)
        newinstance._overwritten = self._overwritten.copy()
        return newinstance

    @property
    def materialized(self):
        return (set(self._loaded.keys()) | set(self._overwritten.keys()))

    @property
    def columns(self):
        return self.materialized | self._available

    @property
    def size(self):
        return len(self)

    @property
    def entrystart(self):
        return self._branchargs["entrystart"]

    @property
    def entrystop(self):
        return self._branchargs["entrystop"]
