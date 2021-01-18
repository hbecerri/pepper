import numpy as np
import awkward as ak
from collections import defaultdict
import logging
from copy import copy


logger = logging.getLogger(__name__)


class Selection:
    def __init__(self):
        self.names = []
        self.cuts = ak.Array({})

    def all(self, names=None):
        if names is None:
            names = self.names
        total = None
        for name in names:
            if total is None:
                total = np.array(self.cuts[name])
            else:
                total = total * self.cuts[name]
        return total

    def add_cut(self, name, accept):
        self.cuts[name] = accept
        self.names.append(name)

    def clear(self):
        self.names = []
        self.cuts = ak.Array({})

    def __len__(self):
        return len(self.names)

    def __copy__(self):
        s = self.__class__.__new__(self.__class__)
        s.__dict__.update(self.__dict__)
        s.names = copy(self.names)
        s.cuts = copy(self.cuts)
        return s


class Selector:
    """Keeps track of the current event selection and data"""

    def __init__(self, data, weight=None, on_update=None, applying_cuts=True):
        """Create a new Selector

        Arguments:
        data -- An `ak.Array` holding the events' data
        weight -- A 1d array of size equal to `data` size, describing
                  the events' weight or None
        on_update -- callable or list of callables that get called after a
                     call to `add_cut` or `set_column`. The callable should
                     accept the keyword argument data, systematics and cut.
        applying_cuts -- bool, wether to apply cuts added with `add_cut`. If
                         False, cuts will be kept at `unapplied_cuts`
        """
        self.data = data
        if on_update is None:
            self.on_update = []
        elif isinstance(on_update, list):
            self.on_update = on_update.copy()
        else:
            self.on_update = [on_update]

        if weight is not None:
            self.systematics = ak.Array({"weight": weight})
        else:
            self.systematics = None

        self.cutnames = []
        self.unapplied_cuts = Selection()
        self.cut_systematic_map = defaultdict(list)

        self._applying_cuts = True
        self.add_cut("Before cuts", np.full(self.num, True))
        self._applying_cuts = applying_cuts

        # Workaround for adding lazy column not working
        self.set_column(
            "__lazyworkaround", lambda data: np.empty(len(data)),
            no_callback=True, lazy=True)

    @property
    def applying_cuts(self):
        return self._applying_cuts

    @applying_cuts.setter
    def applying_cuts(self, val):
        if val and not self._applying_cuts and len(self.unapplied_cuts) > 0:
            self.apply_all_cuts()
        self._applying_cuts = val

    @property
    def unapplied_product(self):
        if len(self.unapplied_cuts) == 0:
            return np.full(self.num, 1.)
        else:
            return self.unapplied_cuts.all()

    @property
    def final(self):
        """Data of events which have passed all cuts (including unapplied ones)
        """
        if len(self.unapplied_cuts) == 0:
            return self.data
        else:
            return ak.mask(
                self.data, ak.values_astype(self.unapplied_product, bool))

    @property
    def final_systematics(self):
        """Systematics of events which have passed all cuts
        (including unapplied ones)"""
        if len(self.unapplied_cuts) == 0:
            return self.systematics
        else:
            unapplied = self.unapplied_product
            masked = ak.mask(self.systematics, unapplied.astype(bool))
            masked["weight"] = masked["weight"] * unapplied
            return masked

    @property
    def num(self):
        "Number of events passing the applied cuts"
        return len(self.data)

    @property
    def num_final(self):
        "Number of selected events passing all cuts (including unapplied ones)"
        if len(self.unapplied_cuts) == 0:
            return self.num
        else:
            return (self.unapplied_product != 0).sum()

    def add_cut(self, name, accept, systematics=None, no_callback=False):
        logger.info(f"Adding cut '{name}'"
                    + (" (no callback)" if no_callback else ""))
        if callable(accept):
            accept = accept(self.data)
            if isinstance(accept, tuple):
                accept, systematics = accept
        self.cutnames.append(name)
        if not isinstance(accept, np.ndarray):
            accept = np.array(accept)
        if accept.dtype == bool:
            accept_weighted = accept.astype(float)
        else:
            accept_weighted = accept
            accept = accept != 0
        self.unapplied_cuts.add_cut(name, accept_weighted)
        if self.applying_cuts:
            self.apply_all_cuts()
        if systematics is not None:
            for sysname, values in systematics.items():
                if isinstance(values, list):
                    raise ValueError("Multiple systematic variations need to "
                                     "be given as tuple, not list")
                if not isinstance(values, tuple):
                    values = (values,)
                if not self.applying_cuts:
                    values_old = values
                    values = []
                    n = self.num
                    for value_old in values_old:
                        value = np.empty(n)
                        value[accept] = value_old
                        values.append(ak.mask(value, accept))
                self.set_systematic(sysname, *values, cut=name)
        if not no_callback:
            for cb in self.on_update:
                cb(data=self.final, systematics=self.final_systematics,
                   cut=name)

    def apply_all_cuts(self):
        weighted = self.unapplied_product
        mask = weighted != 0
        self.data = self.data[mask]
        if self.systematics is not None:
            self.systematics["weight"] = self.systematics["weight"] * weighted
            self.systematics = self.systematics[mask]
        self.unapplied_cuts.clear()

    def set_systematic(self, name, *values, scheme=None, cut=None):
        if name == "weight":
            raise ValueError("The name of a systematic can't be 'weight'")
        if len(values) == 0 and callable(values[0]):
            values = values(self.data)
        if scheme is None:
            if len(values) == 1:
                scheme = "single"
            elif len(values) == 2:
                scheme = "updown"
            else:
                scheme = "numeric"
        if scheme == "single":
            names = [name]
        elif scheme == "updown":
            names = [f"{name}_up", f"{name}_down"]
        elif scheme == "numeric":
            names = [f"{name}_{i}" for i in range(len(values))]
        else:
            raise ValueError("scheme needs to be either 'updown', 'numeric',"
                             f"'single' or None, got {scheme}")
        for name, value in zip(names, values):
            self.systematics[name] = value
            if cut is not None:
                self.cut_systematic_map[cut].append(name)

    def set_column(self, column_name, column, all_cuts=False,
                   no_callback=False, lazy=False):
        if lazy:
            column = ak.virtual(column, (self.data,), cache={},
                                length=self.num)
        elif callable(column):
            column = column(self.final if all_cuts else self.data)
        self.data[column_name] = column
        if not no_callback:
            for cb in self.on_update:
                cb(data=self.final, systematics=self.final_systematics,
                   cut=self.cutnames[-1])

    def set_multiple_columns(self, columns, no_callback=False):
        if callable(columns):
            columns = columns(self.data)
        if isinstance(columns, ak.Array):
            for name in ak.fields(columns):
                self.set_column(name, columns[name], no_callback=True)
        else:
            for name, column in columns.items():
                self.set_column(name, column, no_callback=True)
        if not no_callback:
            for cb in self.on_update:
                cb(data=self.final, systematics=self.final_systematics,
                   cut=self.cutnames[-1])

    def __copy__(self):
        s = self.__class__.__new__(self.__class__)
        s.__dict__.update(self.__dict__)
        # copy(self.data) loads all fields. Workaround
        s.data = ak.Array({})
        for field in self.data.fields:
            s.data[field] = self.data[field]
        s.data.behavior = self.data.behavior
        if self.systematics is not None:
            s.systematics = copy(self.systematics)
        s.cutnames = copy(self.cutnames)
        s.unapplied_cuts = copy(self.unapplied_cuts)
        s.cut_systematic_map = copy(self.cut_systematic_map)
        return s

    def copy(self):
        return copy(self)
