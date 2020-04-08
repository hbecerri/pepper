import os
import json
import coffea
import awkward
import numpy as np


def create_hist_dict(config_json):
    hist_config = json.load(open(config_json))
    return {key: HistDefinition(val) for key, val in hist_config.items()}


def leaddiff(quantity):
    """Returns the difference in quantity of the two leading particles."""
    return quantity[:, 0] - quantity[:, 1]


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

    "leaddiff": leaddiff,
}


class HistDefinitionError(Exception):
    pass


class HistDefinition():
    def __init__(self, config):
        self.ylabel = "Counts"
        self.dataset_axis = coffea.hist.Cat("dataset", "Dataset name")
        self.channel_axis = coffea.hist.Cat("channel", "Channel")
        self.axes = [coffea.hist.Bin(**kwargs) for kwargs in config["bins"]]
        if "cats" in config:
            self.axes.extend(
                [coffea.hist.Cat(**kwargs) for kwargs in config["cats"]])
        self.fill_methods = config["fill"]

    @staticmethod
    def _prepare_fills(fill_vals, mask=None):
        # Check size, flatten jaggedness, pad flat arrays if needed, apply mask
        if mask is None:
            mask = slice(None)
        counts = None
        size = None
        jagged = []
        flat = []
        for key, data in fill_vals.items():
            if data is None:
                continue
            elif isinstance(data, awkward.JaggedArray):
                if counts is not None and (counts != data.counts).any():
                    raise HistDefinitionError(
                        f"Got JaggedArrays for histogram filling with "
                        f"inconsistent counts ({counts} and {data.counts}")
                counts = data.counts
                jagged.append(key)
            else:
                flat.append(key)
            if hasattr(data, "size"):
                if size is not None and data.size != size:
                    raise HistDefinitionError(f"Got inconsistent filling size "
                                              f"({size} and {data.size})")
                size = data.size
        prepared = {}
        for key, data in fill_vals.items():
            if data is not None:
                data = data[mask]
                if key in jagged:
                    data = data.flatten()
                elif key in flat and counts is not None:
                    data = np.repeat(data, counts[mask])
            prepared[key] = data
        return prepared

    def __call__(self, data, channels, dsname, is_mc, weight):
        fill_vals = {name: self.pick_data(method, data)
                     for name, method in self.fill_methods.items()}
        if weight is not None:
            fill_vals["weight"] = weight
        if channels is not None and len(channels) > 0:
            hist = coffea.hist.Hist(
                self.ylabel, self.dataset_axis, self.channel_axis, *self.axes)

            for ch in channels:
                prepared = self._prepare_fills(fill_vals, data[ch])
                if all(val is not None for val in prepared.values()):
                    hist.fill(dataset=dsname, channel=ch, **prepared)
        else:
            hist = coffea.hist.Hist(self.ylabel, self.dataset_axis, *self.axes)
            prepared = self._prepare_fills(fill_vals)
            if all(val is not None for val in prepared.values()):
                hist.fill(dataset=dsname, **prepared)
        return hist

    def pick_data(self, method, data):
        for sel in method:
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
                    if sel["function"] not in func_dict.keys():
                        raise HistDefinitionError(f"Unknown function {sel}")
                    data = func_dict[sel["function"]](data)
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
                    if "slice" in sel:
                        raise HistDefinitionError(
                            "'leading' and 'slice' can't be specified at the "
                            "same time")
                    n = sel["leading"]
                    if n <= 0:
                        raise HistDefinitionError("'leading' must be positive")
                    data = data[:, n - 1]
                if "slice" in sel:
                    slidef = sel["slice"]
                    if isinstance(slidef, int):
                        # Simple stop slice
                        sli = slidef
                    elif isinstance(slidef, list):
                        # Multi-dimensional slice
                        sli = tuple(slice(*x) for x in slidef)
                    else:
                        sli = slice(*slidef)
                    if (isinstance(data, awkward.JaggedArray)
                            and isinstance(sli, list)
                            and len(sli) >= 2):
                        # Jagged slice, ignore events that have too few counts
                        data = data[data.counts > sli[1].stop]
                    data = data[sli]
        else:
            return data
        return None