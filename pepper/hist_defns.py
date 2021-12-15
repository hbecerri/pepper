import coffea.hist
import awkward as ak
import numpy as np
from collections import defaultdict


def not_arr(arr):
    return ~arr


def leaddiff(quantity):
    """Returns the difference in quantity of the two leading particles."""
    if isinstance(quantity, np.ndarray):
        if quantity.ndim == 1:
            quantity = quantity.reshape((-1, 1))
        quantity = ak.Array(quantity)
    enough = quantity[ak.broadcast_arrays(ak.num(quantity) >= 2, quantity)[0]]
    return enough[:, 0:1] - enough[:, 1:2]


def concatenate(*arr, axis=0):
    arr_processed = []
    for a in arr:
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(a, np.ndarray):
            if a.ndim == 1:
                a = a.reshape((-1, 1))
            a = ak.Array(a)
        arr_processed.append(a)
    return ak.concatenate(arr_processed, axis=axis)


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
    "concatenate": concatenate,

    "sum": lambda x: ak.sum(x, axis=1),
    "num": ak.num,
    "zeros_like": ak.zeros_like,
    "ones_like": ak.ones_like,
    "full_like": ak.full_like,
}


class HistDefinitionError(Exception):
    pass


class HistFillError(ValueError):
    pass


class HistDefinition:
    def __init__(self, config):
        self._label = config.get("label", None)
        self.dataset_axis = coffea.hist.Cat("dataset", "Dataset name")
        bins = []
        if "bins" in config:
            for bin_config in config["bins"]:
                unit = bin_config.pop("unit", None)
                if unit is not None:
                    bin_config["label"] = \
                        bin_config.get("label", "") + f" ({unit})"
                bin = coffea.hist.Bin(**bin_config)
                bin.unit = unit
                bins.append(bin)
        if "cats" in config:
            cats = [coffea.hist.Cat(**kwargs) for kwargs in config["cats"]]
        else:
            cats = []
        self.axes = bins + cats
        self.bin_fills = {}
        self.cat_fills = {}
        if "weight" in config:
            self.weight = config["weight"]
        else:
            self.weight = None
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
        if "step_requirement" in config:
            self.step_requirement = config["step_requirement"]
        else:
            self.step_requirement = None

    @staticmethod
    def _prepare_fills(fill_vals, mask=None):
        """This checks for length consistency across the fill_vals,
        removes events where counts do not agree (in case of 2 dims), applies
        the given mask and flattens everything into numpy arrays"""
        if mask is not None:
            if mask.ndim == 1:
                mask = ak.fill_none(mask, False)
                # Raw numpy has better performance here
                mask = np.asarray(mask)
        size = None
        counts = None
        jagged_example = None
        prepared = {}
        for key, data in fill_vals.items():
            if data is None:
                # Early return, no need to bother with other fill_vals
                return {key: None}
            if data.ndim > 2:
                raise ValueError(f"Got more than 2 dimensions for {key}")
            if size is None:
                size = len(data)
            elif size != len(data):
                raise ValueError("Got inconsistant filling size ({size} and"
                                 f"{len(data)} for {key})")
            if mask is None:
                mask = ak.Array(np.full(size, True))
            # Make sure all fills have the same mask originating from ak.mask
            if data.ndim == 1:
                # Performance
                mask = mask & ~np.asarray(ak.is_none(data))
            else:
                mask = mask & ~ak.is_none(data)
            # Make sure all counts agree
            if data.ndim == 2:
                if counts is None:
                    counts = ak.num(data)
                    jagged_example = data
                else:
                    mask = mask & (counts == ak.num(data))
        for key, data in fill_vals.items():
            if jagged_example is not None and data.ndim == 1:
                data = ak.broadcast_arrays(data, jagged_example)[0]
            prepared[key] = np.asarray(ak.flatten(data[mask], axis=None))
        return prepared

    def __call__(self, data, categories, dsname, is_mc, weight):
        axes = self.axes.copy()
        for cat in categories.keys():
            axes.append(coffea.hist.Cat(cat, cat))
        categories = {name: {cat: [cat] for cat in cats}
                      for name, cats in categories.items()}
        categories.update(self.cat_fills)
        hist = coffea.hist.Hist(self.label, self.dataset_axis, *axes)

        fill_vals = {name: DataPicker(method)(data)
                     for name, method in self.bin_fills.items()}
        if self.weight is not None:
            fill_vals["weight"] = DataPicker(self.weight)(data)
        elif weight is not None:
            fill_vals["weight"] = weight
        if any(val is None for val in fill_vals.values()):
            none_keys = [k for k, v in fill_vals.items() if v is None]
            raise HistFillError(f"No fill for axes: {', '.join(none_keys)}")

        cat_present = defaultdict(dict)
        for name, val in categories.items():
            for cat, method in val.items():
                cat_present[name][cat] = DataPicker(method)(data)
        for name, val in cat_present.items():
            if any(mask is None for mask in val.values()):
                cat_present[name] = {"All": np.full(len(data), True)}
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

    @property
    def label(self):
        # Make this a property so that if the axes change, label is updated
        if self._label is not None:
            return self._label
        bins = [ax for ax in self.axes if isinstance(ax, coffea.hist.Bin)]
        if len(bins) == 1 and bins[0]._uniform:
            # This is a 1d, fixed-bin-width histogram.
            edges = bins[0].edges()
            width = (edges[-1] - edges[0]) / (bins[0].size - 3)
            unit = bins[0].unit
            if unit is None:
                unit = "unit" if width == 1 else "units"
            width = f"{width:.3g}"
            if "e" in width:
                width = width.replace("e", r"\times 10^{") + "}"
            return f"Events / ${width}$ {unit}"
        else:
            return "Events / bin"

    @label.setter
    def label(self, value):
        self._label = value


class DataPicker:
    def __init__(self, method):
        self._method = method

    def __call__(self, data):
        method = self._method
        orig_data = data
        if not isinstance(method, list):
            raise HistDefinitionError("Fill method must be list")
        for i, sel in enumerate(method):
            if isinstance(sel, str):
                try:
                    data = getattr(data, sel)
                except AttributeError:
                    try:
                        if (isinstance(data, ak.Array)
                                and sel not in ak.fields(data)):
                            # Workaround for awkward raising a ValueError
                            # instead of a KeyError
                            raise KeyError
                        else:
                            data = data[sel]
                    except KeyError:
                        break
                if callable(data):
                    data = data()
                if data is None:
                    break
            elif isinstance(sel, list):
                try:
                    data = data[sel]
                except (ValueError, KeyError):
                    try:
                        data_proc = {}
                        counts = None
                        can_zip = True
                        for attr in sel:
                            data_proc[attr] = getattr(data, attr)
                            if (can_zip
                                    and isinstance(data_proc[attr], ak.Array)
                                    and data_proc[attr].ndim > 1):
                                if counts is None:
                                    counts = ak.num(data_proc[attr])
                                else:
                                    can_zip = ak.all(
                                        counts == ak.num(data_proc[attr]))
                            else:
                                can_zip = False
                        if can_zip:
                            data = ak.zip(data_proc)
                        else:
                            data = ak.Array(data_proc)
                    except AttributeError:
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
                        data = ak.Array(data)
                    elif not isinstance(data, ak.Array):
                        raise HistDefinitionError(
                            "Can only do 'leading' on numpy or awkward arrays")
                    data = data[:, start:end]
            else:
                raise HistDefinitionError("Fill constains invalid type, must "
                                          f"be str, list or dict: {sel}")
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
                    args.append(self.__class__(value)(orig_data))
                    if args[-1] is None:
                        return None
                else:
                    args.append(value)
        if "kwargs" in sel:
            if not isinstance(sel["kwargs"], dict):
                raise HistDefinitionError("kwargs for function must be dict")
            for key, value in sel["kwargs"].items():
                if isinstance(value, list):
                    kwargs[key] = self.__class__(value)(orig_data)
                    if kwargs[key] is None:
                        return None
                else:
                    kwargs[key] = value
        if i == 0:
            return func(*args, **kwargs)
        else:
            return func(data, *args, **kwargs)

    @property
    def name(self):
        name = ""
        for sel in self._method:
            if isinstance(sel, str):
                name += sel.replace("/", "")
            elif isinstance(sel, list):
                name += ",".join(sel)
            elif isinstance(sel, dict):
                if "function" in sel:
                    name += sel["function"].replace("/", "")
                elif "key" in sel:
                    name += sel["key"].replace("/", "")
                elif "attribute" in sel:
                    name += sel["attribute"].replace("/", "")
                elif "leading" in sel:
                    name += "leading"
                    if isinstance(sel["leading"], tuple):
                        if (not isinstance(sel["leading"][0], int)
                                or not isinstance(sel["leading"][0], int)):
                            raise HistDefinitionError(
                                "Invalid value for leading tuple. Must be int")
                        name += (str(sel["leading"][0]) + "-"
                                 + str(sel["leading"][1]))
                    elif isinstance(sel["leading"], int):
                        name += str(sel["leading"])
                    else:
                        raise HistDefinitionError(
                            "Invalid value for leading. Must be int or tuple")
                else:
                    raise HistDefinitionError(
                        f"Dict without any recognized keys: {sel}")
            else:
                raise HistDefinitionError("Selection constains invalid type, "
                                          f"must be str, list or dict: {sel}")
            name += "_"
        name = name[:-1]  # Remove last underscore
        return name
