import numpy as np
import awkward
import copy
import logging

import pepper
from pepper import PackedSelectionAccumulator


logger = logging.getLogger(__name__)


class Selector():
    """Keeps track of the current event selection and data"""

    def __init__(self, table, weight=None, on_update=None):
        """Create a new Selector

        Arguments:
        table -- An `awkward.Table` or `LazyTable` holding the events' data
        weight -- A 1d numpy array of size equal to `table` size, describing
                  the events' weight or None
        on_update -- callable or list of callables that get called after a
                     call to `add_cut` or `set_column`. The callable should
                     accept the keyword argument data, systematics and cut.
        """
        self.table = table
        self._cuts = PackedSelectionAccumulator()
        self._current_cuts = []
        self._frozen = False
        if on_update is None:
            self.on_update = []
        elif isinstance(on_update, list):
            self.on_update = on_update.copy()
        else:
            self.on_update = [on_update]

        if weight is not None:
            self.systematics = awkward.Table({"weight": weight})
        else:
            self.systematics = None

        # Add a dummy cut to inform about event number and circumvent error
        # when calling all or require before adding actual cuts
        self.add_cut(np.full(self.table.size, True), "Before cuts")

    @property
    def masked(self):
        """Get currently selected events

        Returns an `awkward.Table` of the currently selected events
        """
        if len(self._current_cuts) > 0:
            return self.table[self._cur_sel]
        else:
            return self.table

    @property
    def weight(self):
        """Get the event weights for the currently selected events
        """
        if self.systematics is None:
            return None
        weight = self.systematics["weight"].flatten()
        if len(self._current_cuts) > 0:
            return weight[self._cur_sel]
        else:
            return weight

    @property
    def masked_systematics(self):
        """Get the systematics for the currently selected events

        Returns an `awkward.Table`, where "weight" maps to the event weight.
        All other columns are named by the scale factor they belong to.
        """
        if self.systematics is None:
            return None
        if len(self._current_cuts) > 0:
            return self.systematics[self._cur_sel]
        else:
            return self.systematics

    @property
    def final(self):
        """Get events which have passed all cuts
        (both those before and after freeze_selection)
        """
        return self.table[self._final_sel]

    @property
    def final_systematics(self):
        """Get the systematics for the events which have passed all cuts
        """
        if self.systematics is None:
            return None
        return self.systematics[self._final_sel]

    def freeze_selection(self):
        """Freezes the selection

        After a call to this method, additional cuts wont effect the current
        selection anymore.
        """

        self._frozen = True

    @property
    def _cur_sel(self):
        """Get a bool mask describing the current selection"""
        return self._cuts.all(*self._current_cuts)

    @property
    def _final_sel(self):
        """Get a bool mask describing the final selection"""
        return self._cuts.all(*self._cuts.names)

    @property
    def num_selected(self):
        return self._cur_sel.sum()

    def add_cut(self, accept, name, no_callback=False):
        """Adds a cut

        Cuts control what events get fed into later cuts, get saved and are
        given by `masked`.

        Arguments:
        accept -- An array of bools or a tuple or a function, that returns the
                  former. The tuple contains first mentioned array and a dict.
                  The function that will be called with a table of the
                  currently selected events.
                  The array has the same length as the table and indicates if
                  an event is not cut (True).
                  The dict maps names of scale factors to either the SFs or
                  tuples of the form (sf, up, down) where sf, up and down are
                  arrays of floats giving central, up and down variation for a
                  scale factor for each event, thus making only sense in case
                  of MC. In case of no up/down variations (sf, None) is a valid
                  value. `up` and `down` must be given relative to sf.
                  accept does not get called if num_selected is already 0.
        name -- A label to assoiate wit the cut
        no_callback -- Do not call on_update after the cut is added
        """
        if name in self._cuts.names:
            raise ValueError("A cut with name {} already exists".format(name))
        logger.info(f"Adding cut '{name}'")
        if callable(accept):
            accepted = accept(self.masked)
        else:
            accepted = accept
        if isinstance(accepted, tuple):
            accepted, weight = accepted
        else:
            weight = {}
        if len(self._current_cuts) > 0:
            cut = np.full(self.table.size, False)
            cut[self._cur_sel] = accepted
        else:
            cut = accepted
        self._cuts.add_cut(name, cut)
        if not self._frozen:
            self._current_cuts.append(name)
            mask = None
        else:
            mask = accepted
        for weightname, factors in weight.items():
            if isinstance(factors, tuple):
                factor = factors[0]
                updown = (factors[1], factors[2])
            else:
                factor = factors
                updown = None
            self.modify_weight(weightname, factor, updown, mask)
        if not no_callback:
            for cb in self.on_update:
                cb(data=self.final, systematics=self.final_systematics,
                   cut=name)

    def _pad_npcolumndata(self, data, defaultval=None, mask=None):
        padded = np.empty(self.table.size, dtype=data.dtype)
        if defaultval:
            padded[:] = defaultval
        if mask is not None:
            total_mask = self._cur_sel
            total_mask[self._cur_sel] = mask
            padded[total_mask] = data
        else:
            padded[self._cur_sel] = data
        return padded

    def set_systematic(self, name, up, down, mask=None):
        """Set the systematic up/down variation for a systematic given by
        `name`. `mask` is an array of bools and indicates, which events the
        systematic applies to. If `None`, the systematic applies to all events.
        `up` and `down` must be given relative to the central value.
        """
        if name == "weight":
            raise ValueError("The name of a systematic can't be 'weight'")
        self.systematics[name + "_up"] = self._pad_npcolumndata(up, 1, mask)
        self.systematics[name + "_down"] = self._pad_npcolumndata(
            down, 1, mask)

    def modify_weight(self, name, factor=None, updown=None, mask=None):
        """Modify the event weight. The weight will be multiplied by `factor`.
        `name` gives the name of the factor and is important to keep track of
        the systematics supplied by `updown`. If updown is not None, it should
        be a tuple of up and down variation factors relative to `factor`.
        `mask` is an array of bools and indicates, which events the
        systematic applies to.
        """
        if factor is not None:
            factor = self._pad_npcolumndata(factor, 1, mask)
            self.systematics["weight"] = self.systematics["weight"] * factor
        if updown is not None:
            self.set_systematic(name, updown[0], updown[1], mask)

    def set_column(self, column, column_name, all_cuts=False,
                   no_callback=False):
        """Sets a column of the table

        Arguments:
        column -- Column data or a function that returns it.
                  The function that will be called with a table of the
                  currently selected events. Does not get called if
                  `num_selected` is 0 already.
                  The column data must be a numpy array or an
                  awkward.JaggedArray with a size of `num_selected`.
        column_name -- The name of the column to set
        all_cuts -- The column function will be called only on events passing
                    all cuts (including after freezing). The function must
                    return a JaggedArray in this case.
        no_callback -- Do not call on_update after the cut is added
        """
        if not isinstance(column_name, str):
            raise ValueError("column_name needs to be string")
        logger.info(f"Adding column '{column_name}'")
        if callable(column):
            if all_cuts:
                data = column(self.final)
            else:
                data = column(self.masked)
        else:
            data = column

        # Convert data to appropriate type if possible
        if isinstance(data, awkward.ChunkedArray):
            data = awkward.concatenate(data.chunks)

        # Move data into the table with appropriate padding (important!)
        if isinstance(data, np.ndarray):
            if all_cuts:
                raise ValueError("Got numpy array but all_cuts was specified")
            unmasked_data = self._pad_npcolumndata(data)
        elif isinstance(data, awkward.JaggedArray):
            counts = np.zeros(self.table.size, dtype=int)
            if all_cuts:
                counts[self._final_sel] = data.counts
            else:
                counts[self._cur_sel] = data.counts
            cls = awkward.Methods.maybemixin(type(data), awkward.JaggedArray)
            unmasked_data = cls.fromcounts(counts, data.flatten())
        else:
            raise TypeError("Unsupported column type {}".format(type(data)))
        self.table[column_name] = unmasked_data

        if not no_callback:
            cut_name = self._cuts.names[-1]
            for cb in self.on_update:
                cb(data=self.final, systematics=self.final_systematics,
                   cut=cut_name)

    def unset_column(self, column):
        logger.info("Removing column '{column}'")
        del self.table[column]

    def set_multiple_columns(self, columns, no_callback=False):
        """Sets multiple columns of the table

        Arguments:
        columns -- A dict of columns, with keys determining the column names.
                   For requirements to the values, see `column` parameter of
                   `set_column`.
        no_callback -- Do not call on_update after the cut is added
        """
        if callable(columns):
            columns = columns(self.masked)
        for name, column in columns.items():
            self.set_column(column, name, no_callback=True)
        if not no_callback:
            cut_name = self._cuts.names[-1]
            for cb in self.on_update:
                cb(data=self.final, systematics=self.final_systematics,
                   cut=cut_name)

    def get_columns(self, part_props={}, other_cols=set(), cuts="Current",
                    prefix=""):
        """Get columns of events passing cuts

        Arguments:
        part_props- A dictionary of particles, followed by a list of properties
                     one wishes to save for those particles- "p4" will add all
                     components of the 4-momentum
        other_cols- The other columns one wishes to save
        cuts      - "Current", "All" or a list of cuts - the list of cuts to
                     apply before saving- The default, "Current", only applies
                     the cuts before freeze_selection
        prefix    - A string that gets prepended to every key in return_dict

        Returns:
        return_dict - A dict containing JaggedArrays or numpy arrays of the
                      columns
        """
        if cuts == "Current":
            cuts = self._current_cuts
        elif cuts == "All":
            cuts = self._cuts.names
        elif not isinstance(cuts, list):
            raise ValueError("cuts needs to be one of 'Current', 'All' or a "
                             "list")
        data = self.table[self._cuts.all(*cuts)]
        return_dict = {}
        for part in part_props.keys():
            props = set(part_props[part])
            if "p4" in props:
                props |= {"pt", "eta", "phi", "mass"}
            props -= {"p4"}
            for prop in props:
                if not hasattr(data[part], prop):
                    continue
                arr = pepper.misc.jagged_reduce(getattr(data[part], prop))
                return_dict[part + "_" + prop] = arr
        for col in other_cols:
            if col not in data:
                continue
            return_dict[prefix + col] = pepper.misc.jagged_reduce(data[col])
        return return_dict

    def get_columns_from_config(self, to_save, prefix=""):
        return self.get_columns(to_save["part_props"],
                                to_save["other_cols"],
                                to_save["cuts"],
                                prefix=prefix)

    def get_cuts(self, cuts="Current"):
        """Get information on what events pass which cuts

        Arguments:
        cuts -- "Current", "All" or a list of cuts - the list of cuts to
                apply before saving- The default, "Current", only applies
                the cuts before freeze_selection
        """
        if cuts == "Current":
            cuts = self._current_cuts
        elif cuts == "All":
            cuts = self._cuts.names
        elif not isinstance(cuts, list):
            raise ValueError("cuts needs to be one of 'Current', 'All' or a "
                             "list")
        return self._cuts.mask[self._cuts.all(*cuts)]

    def copy(self):
        """Create a copy of the Selector instance, containing shallow copies
        of most of its constituents.
        This is indended to be used if one wants to fork the selection to, for
        example, repeat particular steps with different settings.
        Already read or set columns are being handled memory-efficiently,
        meaning a call to copy won't double the memory usage for present
        columns."""
        s = self.__class__(copy.copy(self.table))
        s._cuts = copy.deepcopy(self._cuts)
        s._current_cuts = copy.copy(self._current_cuts)
        s._frozen = self._frozen
        if self.on_update is None:
            s.on_update = []
        else:
            s.on_update = copy.copy(self.on_update)
        if self.systematics is None:
            s.systematics = None
        else:
            s.systematics = copy.copy(self.systematics)
        return s
