import numpy as np
import awkward as ak
from collections import defaultdict
import logging
from copy import copy
import pepper.misc


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

    def __init__(self, data, weight=None, on_update=None, applying_cuts=True,
                 rng_seed=None):
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
        rng_seed -- int or tuple of ints to seed the random number generator
                    with. If None, a random seed will be used. For defailts
                    see the parameter of numpy.random.default_rng().
        """
        self.data = data
        if hasattr(self.data, "metadata"):
            self.metadata = self.data.metadata
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
        self.done_steps = set()

        self.rng = np.random.default_rng(rng_seed)

        self._applying_cuts = True
        self.add_cut("Before cuts", np.full(self.num, True))
        self._applying_cuts = applying_cuts

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
        """An array giving the effect on the event weight of all cuts added
        after the last cut was applied."""
        if len(self.unapplied_cuts) == 0:
            return np.full(self.num, 1.)
        else:
            return self.unapplied_cuts.all()

    @property
    def final(self):
        """Data of events which have passed all cuts (including unapplied ones)
        """
        return ak.mask(
            self.data, ak.values_astype(self.unapplied_product, bool))

    @property
    def final_systematics(self):
        """Systematics of events which have passed all cuts
        (including unapplied ones)"""
        if self.systematics is None:
            return None
        unapplied = self.unapplied_product
        masked = ak.mask(
            self.systematics, ak.values_astype(unapplied, bool))
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
            return ak.sum(self.unapplied_product != 0)

    def _invoke_callbacks(self):
        data = self.final
        systematics = self.final_systematics
        for cb in self.on_update:
            cb(data=data, systematics=systematics, cut=self.cutnames[-1],
               done_steps=self.done_steps)

    def _get_category_mask(self, categories):
        num = self.num
        mask = np.full(num, True)
        for cat, regs in categories.items():
            cat_mask = np.full(num, False)
            for reg in regs:
                cat_mask = cat_mask | self.data[reg]
            mask = mask & cat_mask
        return mask

    def add_cut(self, name, accept, systematics=None, no_callback=False,
                categories=None):
        """Adds a cut and applies it if `self.applying_cuts` is True, otherwise
        the cut will be stored in `self.unapplied_cuts`. Applying in this
        context means that rows of `self.data` are discarded accordingly.

        Argument:
        name -- Name of the cut
        accept -- An array of bools or floats or a tuple of the former and a
                  systematics dict or a callable returning any of the former.
                  In the array a value of 0 or `False` means that the
                  event corresponding to the row is discarded. Any other value
                  will get multiplied into the event weight. Here a value of
                  `True` corresponds to a 1.
                  The systematics dict is a mapping of systematics name ->
                  values, where name and values have the same meaning as in
                  `self.set_systematic`. The values arrays of lengths equal
                  `self.num` either before or after the cut is applied.
                  Systematics given for cut evets are ignored.
                  In case this is a callable, the callable will be called and
                  its return value will be used as the new value for this
                  parameter.
        systematics -- A dict of name and values and has the same effect has
                       calling `self.set_systematic` on every item. Will be
                       ignored if a systematics dict is given with `accept`.
        no_callback -- A bool whether not to call the callbacks, which usually
                       fill histograms etc.
        categories -- If not None, ignore events that are not part of any of
                      the specified categorizations. This is done by specifying
                      a dict of lists. A key gives the name of the
                      categorization, while the list contains field names
                      of `self.data`. These fields needs to be flat bool
                      arrays. An event is considered to be part of a
                      categorization if any of the fields are True.
        """
        def pad_cats(mask, arr):
            full_arr = np.full(self.num, 1, dtype=arr.dtype)
            full_arr[mask] = arr
            return full_arr

        logger.info(f"Adding cut '{name}'"
                    + (" (no callback)" if no_callback else ""))
        if categories is not None:
            cats_mask = self._get_category_mask(categories)
            if callable(accept):
                accept = accept(self.data[cats_mask])
        elif callable(accept):
            accept = accept(self.data)
        if isinstance(accept, tuple):
            accept, systematics = accept
        # Allow accept to be masked. Treat masked values as False
        accept = ak.fill_none(accept, 0)
        self.cutnames.append(name)
        if not isinstance(accept, np.ndarray):
            accept = np.array(accept)
        if categories is not None:
            accept = pad_cats(cats_mask, accept)
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
                values_old = values
                values = []
                n = self.num
                for value_old in values_old:
                    if categories is not None:
                        value_old = pad_cats(cats_mask, value_old)
                    if len(value_old) != n:
                        if self.applying_cuts:
                            value = value_old[accept]
                        else:
                            value = np.empty(n)
                            value[accept] = value_old
                            value = ak.mask(value, accept)
                    elif not self.applying_cuts:
                        value = ak.mask(value_old, accept)
                    else:
                        value = value_old
                    values.append(value)
                self.set_systematic(sysname, *values, cut=name)
        self.done_steps.add("cut:" + name)
        if not no_callback:
            self._invoke_callbacks()

    def apply_all_cuts(self):
        """Applies all unapplied cuts, discarding rows of `self.data` where the
        resulting weight is 0 and modifies the event weight accordingly."""
        weighted = self.unapplied_product
        mask = weighted != 0
        self.data = self.data[mask]
        if self.systematics is not None:
            self.systematics["weight"] = self.systematics["weight"] * weighted
            self.systematics = self.systematics[mask]
        self.unapplied_cuts.clear()

    def set_systematic(self, name, *values, scheme=None, cut=None):
        """Set the systematic variation for an uncertainty. These will be
        found in the `self.systematics`.

        Arguments:
        name -- Name of the systematic to set.
        values -- Arrays. Each array gives the ratio of a systematic
                  variation and the central value of the event weight.
        scheme -- One of 'updown', 'numeric', 'single' or None. Determines the
                  column (of the systematics table) the values will appear in.
                  'updown': Requires `values` to have length 2. Column will
                  be `name` + _up and _down.
                  'numeric' column will be `name` + _i where i is determined by
                  enumerating `values`
                  'single': Requires `values` to have length 1. Column will be
                  `name`
                  None: scheme will be decided based on `values` length, where
                  numeric will be used for lengths > 2.
        cut -- Name of the cut after which the systematic needs to be accounted
               for. If not None, a corresponding item will be found in
               `self.cut_systematic_map`.
        """

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

    def _mask(self, column, mask):
        num_events = ak.sum(mask)
        if len(column) != num_events:
            raise ValueError(f"Column length exptected to be {num_events} but "
                             f"got {len(column)}")
        if len(column) > 0:
            return ak.mask(column[np.cumsum(np.asarray(mask)) - 1], mask)
        else:
            return ak.pad_none(column, len(mask), axis=0)

    def set_column(self, column_name, column, all_cuts=False,
                   no_callback=False, lazy=False, categories=None):
        """Sets a column of `self.data`.

        Arguments:
        column_name -- The name of the column to set
        column -- Column data or a callable that returns it.
                  The callable that will be called with `self.data` as argument
        all_cuts -- The callable from `column` will be called only on events
                    passing all cuts (including unapplied cuts).
        no_callback -- A bool whether not to call the callbacks, which usually
                       fill histograms etc.
        lazy -- If True, column must be a callable and the column will be
                inserted as a virtual array, making the callable only called
                when the data array determines it has to.
        categories -- If not None, ignore events that are not part of any of
                      the specified categorizations. This is done by specifying
                      a dict of lists. A key gives the name of the
                      categorization, while the list contains field names
                      of `self.data`. These fields needs to be flat bool
                      arrays. An event is considered to be part of a
                      categorization if any of the fields are True.
        """

        logger.info(
            f"Setting column {column_name}" + (" lazily" if lazy else ""))
        if all_cuts and not self.applying_cuts:
            data = self.final
            mask = ~ak.is_none(data)
            data = ak.flatten(self.final, axis=0)
        else:
            data = self.data
            mask = None

        if categories is not None:
            cats_mask = self._get_category_mask(categories)
            data = data[cats_mask]

        if callable(column):
            if lazy:
                column = pepper.misc.VirtualArrayCopier(data).wrap_with_copy(
                    column)
                column = ak.virtual(column, cache={}, length=len(data))
            else:
                column = column(data)
        if categories is not None:
            column = self._mask(column, cats_mask)
        if mask is not None:
            column = self._mask(column, mask)
        if lazy:
            data_copier = pepper.misc.VirtualArrayCopier(self.data)
            data_copier[column_name] = column
            self.data = data_copier.get()
        else:
            self.data[column_name] = column
        self.done_steps.add("column:" + column_name)
        if not no_callback:
            self._invoke_callbacks()

    def set_multiple_columns(self, columns, all_cuts=False, no_callback=False,
                             categories=None):
        """Sets multiple columns of `self.data` at once.

        Arguments:
        columns -- A dict of column names and data or a callable returning
                   the former. The callable will be called with `self.data`
                   as argument.
        all_cuts -- The callables from `columns` will be called only on events
                    passing all cuts (including unapplied cuts).
        no_callback -- A bool whether not to call the callbacks, which usually
                       fill histograms etc.
        categories -- If not None, ignore events that are not part of any of
                      the specified categorizations. This is done by specifying
                      a dict of lists. A key gives the name of the
                      categorization, while the list contains field names
                      of `self.data`. These fields needs to be flat bool
                      arrays. An event is considered to be part of a
                      categorization if any of the fields are True.
        """
        if callable(columns):
            if all_cuts and not self.applying_cuts:
                data = self.final
                data = ak.flatten(self.final, axis=0)
            else:
                data = self.data
            if categories is not None:
                cats_mask = self._get_category_mask(categories)
                data = data[cats_mask]
            columns = columns(data)
        if isinstance(columns, ak.Array):
            columns = {k: columns[k] for k in ak.fields(columns)}
        for name, column in columns.items():
            self.set_column(name, column, all_cuts, no_callback=True,
                            categories=categories)
        if not no_callback:
            self._invoke_callbacks()

    def get_cuts(self):
        """Returns a tuple a list of all cut names and an array containing the
           currently unapplied cuts"""
        return self.cutnames, self.unapplied_cuts.cuts

    def __copy__(self):
        """Makes a shallow copy excluding some constituents. This allows to
        work with the copy without modifying the original"""
        s = self.__class__.__new__(self.__class__)
        s.__dict__.update(self.__dict__)
        # copy(self.data) loads all fields. Workaround
        s.data = pepper.misc.VirtualArrayCopier(self.data).get()
        if self.systematics is not None:
            s.systematics = copy(self.systematics)
        s.cutnames = copy(self.cutnames)
        s.unapplied_cuts = copy(self.unapplied_cuts)
        s.cut_systematic_map = copy(self.cut_systematic_map)
        s.done_steps = copy(self.done_steps)
        return s

    def copy(self):
        """Shorthard for `copy.copy(self)``, see `__copy__`"""
        return copy(self)
