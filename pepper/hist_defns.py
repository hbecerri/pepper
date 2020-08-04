import json
import coffea
import awkward
import numpy as np


def create_hist_dict(config_json):
    hist_config = json.load(open(config_json))
    return {key: HistDefinition(val) for key, val in hist_config.items()}


def not_arr(arr):
    return ~arr


def leaddiff(quantity):
    """Returns the difference in quantity of the two leading particles."""
    if isinstance(quantity, np.ndarray):
        if quantity.ndim == 1:
            quantity = quantity.reshape((-1, 1))
        quantity = awkward.JaggedArray.fromregular(quantity)
    enough = quantity.counts >= 2
    q_enough = quantity[enough]
    diff = q_enough[:, 0] - q_enough[:, 1]
    return awkward.JaggedArray.fromcounts(enough.astype(int), diff)


def concatenate(*arr, axis=0):
    arr_processed = []
    for a in arr:
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(a, np.ndarray):
            if a.ndim == 1:
                a = a.reshape((-1, 1))
            a = awkward.JaggedArray.fromregular(a)
        arr_processed.append(a)
    return awkward.concatenate(arr_processed, axis=axis)


func_dict = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,

    "not": not_arr,
    "leaddiff": leaddiff,
    "concatenate": concatenate
}


class HistDefinitionError(Exception):
    pass


class HistDefinition():
    def __init__(self, config):
        self.ylabel = "Counts"
        self.dataset_axis = coffea.hist.Cat("dataset", "Dataset name")
        self.channel_axis = coffea.hist.Cat("channel", "Channel")
        if "bins" in config:
            bins = [coffea.hist.Bin(**kwargs) for kwargs in config["bins"]]
        else:
            bins = []
        if "cats" in config:
            cats = [coffea.hist.Cat(**kwargs) for kwargs in config["cats"]]
        else:
            cats = []
        self.axes = bins + cats
        self.bin_fills = {}
        self.cat_fills = {}
        for axisname, method in config["fill"].items():
            if axisname in bins:
                self.bin_fills[axisname] = method
            elif axisname in cats:
                self.cat_fills[axisname] = method
            else:
                raise HistDefinitionError(
                    f"Fill provided for non-existing axis: {axisname}")
        missing_fills = (set(a.name for a in self.axes)
                         - set(config["fill"].keys()))
        if len(missing_fills) != 0:
            raise HistDefinitionError(
                "Missing fills for axes: " + ", ".join(missing_fills))

    @staticmethod
    def _prepare_fills(fill_vals, mask=None):
        # Check size, flatten jaggedness, pad flat arrays if needed, apply mask
        counts = None
        counts_mask = None
        size = None
        jagged = []
        flat = []
        for key, data in fill_vals.items():
            if data is None:
                continue
            if hasattr(data, "size"):
                if size is not None and data.size != size:
                    raise HistDefinitionError(f"Got inconsistent filling size "
                                              f"({size} and {data.size})")
                size = data.size
            if isinstance(data, awkward.JaggedArray):
                if counts is not None and (counts != data.counts).any():
                    if counts_mask is None:
                        counts_mask = np.full(size, True)
                    counts_mask = counts == data.counts
                    counts[~counts_mask] = 0
                else:
                    counts = data.counts
                jagged.append(key)
            else:
                flat.append(key)
        prepared = {}
        for key, data in fill_vals.items():
            if data is not None:
                if key in jagged:
                    if counts_mask is not None and mask is not None:
                        data = data[counts_mask & mask]
                    elif mask is not None:
                        data = data[mask]
                    elif counts_mask is not None:
                        data = data[counts_mask]
                    data = data.flatten()
                elif key in flat and counts is not None:
                    if isinstance(mask, awkward.JaggedArray):
                        data = data.repeat(counts)[mask.flatten()]
                    elif mask is not None:
                        data = data[mask].repeat(counts[mask])
                    else:
                        data = data.repeat(counts)
            prepared[key] = data
        return prepared

    def __call__(self, data, channels, dsname, is_mc, weight):
        fill_vals = {name: self.pick_data(method, data)
                     for name, method in self.bin_fills.items()}
        cat_masks = {name: {cat: self.pick_data(method, data)
                            for cat, method in val.items()}
                     for name, val in self.cat_fills.items()}
        if weight is not None:
            fill_vals["weight"] = weight
        cat_present = {name: (val if any(mask is not None
                                         for mask in val.values())
                              else {"All": np.full(data.size, True)})
                       for name, val in cat_masks.items()}
        axes = self.axes.copy()
        if channels is not None and len(channels) > 0:
            axes.append(self.channel_axis)
            cat_present["channel"] = {ch: data[ch] for ch in channels}
        hist = coffea.hist.Hist(self.ylabel, self.dataset_axis, *axes)
        if len(cat_present) == 0:
            prepared = self._prepare_fills(fill_vals)

            if all(val is not None for val in prepared.values()):
                hist.fill(dataset=dsname, **prepared)
        else:
            cat_combinations = None
            for name, val in cat_present.items():
                if cat_combinations is None:
                    cat_combinations = {((name, cat), ): mask
                                        for cat, mask in val.items()}
                else:
                    new_cc = {}
                    for cat, mask in val.items():
                        for key, _mask in cat_combinations.items():
                            key += ((name, cat),)
                            new_cc[key] = mask & _mask
                    cat_combinations = new_cc
            for combination, mask in cat_combinations.items():
                prepared = self._prepare_fills(fill_vals, mask)
                combination = {key: val for key, val in combination}
                if all(val is not None for val in prepared.values()):
                    hist.fill(dataset=dsname, **combination, **prepared)

        return hist

    def pick_data(self, method, data):
        orig_data = data
        if not isinstance(method, list):
            raise HistDefinitionError("Fill method must be list")
        for i, sel in enumerate(method):
            if isinstance(sel, str):
                try:
                    data = getattr(data, sel)
                except AttributeError:
                    try:
                        data = data[sel]
                    except (KeyError, ValueError):
                        break
                if callable(data):
                    data = data()
                if data is None:
                    break
            elif isinstance(sel, dict):
                if "function" in sel:
                    data = self._pick_data_from_function(
                        i, sel, data, orig_data)
                    if data is None:
                        break
                elif "key" in sel:
                    try:
                        data = data[sel["key"]]
                    except KeyError:
                        break
                elif "attribute" in sel:
                    try:
                        data = getattr(data, sel["attribute"])
                    except AttributeError:
                        break

                if "leading" in sel:
                    leading = sel["leading"]
                    if isinstance(leading, (list, tuple)):
                        start = leading[0] - 1
                        end = leading[1] - 1
                        if start >= end:
                            raise HistDefinitionError(
                                "First entry in 'leading' must be larger than "
                                "the second one")
                    else:
                        start = leading - 1
                        end = start + 1
                    if isinstance(data, np.ndarray):
                        data = awkward.JaggedArray.fromregular(data)
                    elif not isinstance(data, awkward.JaggedArray):
                        raise HistDefinitionError(
                            "Can only do 'leading' on numpy or jagged arrays")
                    # Do not change data.size as we later need to apply channel
                    # masks
                    start = np.minimum(data.stops, data.starts + start)
                    end = np.minimum(data.stops, data.starts + end)
                    data = awkward.JaggedArray(start, end, data.content)
        else:
            return data
        return None

    def _pick_data_from_function(self, i, sel, data, orig_data):
        funcname = sel["function"]
        if funcname not in func_dict.keys():
            raise HistDefinitionError(f"Unknown function {funcname}")
        func = func_dict[funcname]
        args = []
        kwargs = {}
        if "args" in sel:
            if not isinstance(sel["args"], list):
                raise HistDefinitionError("args for function must be list")
            for value in sel["args"]:
                if isinstance(value, list):
                    args.append(self.pick_data(value, orig_data))
                    if args[-1] is None:
                        return None
                else:
                    args.append(value)
        if "kwargs" in sel:
            if not isinstance(sel["kwargs"], dict):
                raise HistDefinitionError("kwargs for function must be dict")
            for key, value in sel["kwargs"].items():
                if isinstance(value, list):
                    kwargs[key] = self.pick_data(value, orig_data)
                    if kwargs[key] is None:
                        return None
                else:
                    kwargs[key] = value
        if i == 0:
            return func(*args, **kwargs)
        else:
            return func(data, *args, **kwargs)
