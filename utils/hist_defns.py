import os
import json
from coffea import hist
import awkward
import numpy as np


def create_hist_dict(config_folder):
    hist_config = json.load(open(os.path.join(config_folder,
                                              "hist_config.json")))
    return {key: HistDefinition(val) for key, val in hist_config.items()}


def jet_mult(data):
    if "Jet" in data:
        return data["Jet"].counts
    else:
        return None


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

    "jet_mult": jet_mult,
}


class HistDefinitionError():
    pass


class HistDefinition():
    def __init__(self, config):
        self.dataset_axis = hist.Cat("dataset", "")
        self.axes = [hist.Bin(**kwargs) for kwargs in config["bins"]]
        if "cats" in config:
            self.axes.extend([hist.Cat(**kwargs) for kwargs in config["cats"]])
        self.fill_methods = config["fill"]

    @staticmethod
    def _prepare_fills(fill_vals, mask=None):
        # Flatten jaggedness, pad flat arrays if needed, apply mask
        if mask is None:
            mask = slice(None)
        counts = None
        jagged = []
        flat = []
        for key, data in fill_vals.items():
            if data is None:
                continue
            elif isinstance(data, awkward.JaggedArray):
                if counts is not None and counts != data.counts:
                    raise HistDefinitionError(
                        f"Got JaggedArrays for histogram filling with "
                        "disagreeing counts ({counts} and {data.counts}")
                counts = data.counts
                jagged.append(key)
            else:
                flat.append(key)
        for key in jagged:
            fill_vals[key] = fill_vals[key][mask].flatten()
        if counts is not None:
            for key in flat:
                fill_vals[key] = np.repeat(fill_vals[key][mask], counts)

    def __call__(self, data, dsname, is_mc, weight):
        channels = ["ee", "emu", "mumu", "None"]
        fill_vals = {name: self.pick_data(method, data)
                     for name, method in self.fill_methods.items()}
        if weight is not None:
            fill_vals["weight"] = weight
        if channels[0] in data:
            channel_axis = hist.Cat("channel", "")
            _hist = hist.Hist("Counts", self.dataset_axis,
                              channel_axis, *self.axes)

            for ch in channels:
                self._prepare_fills(fill_vals, data[ch])
                if all(val is not None for val in fill_vals.values()):
                    _hist.fill(dataset=dsname, channel=ch, **fill_vals)
        else:
            _hist = hist.Hist("Counts", self.dataset_axis, *self.axes)
            self._prepare_fills(fill_vals)
            if all(val is not None for val in fill_vals.values()):
                _hist.fill(dataset=dsname, **fill_vals)
        return _hist

    def pick_data(self, method, data):
        for sel in method:
            if isinstance(sel, str):
                try:
                    data = getattr(data, sel)
                except AttributeError:
                    try:
                        data = data[sel]
                    except KeyError:
                        break
                if callable(data):
                    data = data()
            elif isinstance(sel, dict):
                if "function" in sel:
                    if sel not in func_dict:
                        raise HistDefinitionError(f"Unknown function {sel}")
                    data = func_dict[sel["function"]](data)
                elif "key" in sel:
                    if sel["key"] in data:
                        data = data[sel["key"]]
                    else:
                        break
                elif "prop" in sel:
                    data = getattr(data, sel["prop"])

                if "slice" in sel and data is not None:
                    data = data[sel["slice"]]
                if "jagged_slice" in sel and data is not None:
                    safe = data[data.counts > sel["jagged_slice"]]
                    data = np.empty(len(data))
                    data = safe[:, sel["jagged_slice"]]
        else:
            return data
        return None

