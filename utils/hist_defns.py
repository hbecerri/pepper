import os
import json

from coffea import hist
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


class HistDefinition():
    def __init__(self, config):
        self.dataset_axis = hist.Cat("dataset", "")
        self.axes = [hist.Bin(**kwargs) for kwargs in config["bins"]]
        if "cats" in config:
            self.axes.extend([hist.Cat(**kwargs) for kwargs in config["cats"]])
        self.fill_methods = config["fill"]

    def __call__(self, data, dsname, is_mc, weight):
        channels = ["ee", "emu", "mumu", "None"]
        if channels[0] in data:
            channel_axis = hist.Cat("channel", "")
            _hist = hist.Hist("Counts", self.dataset_axis,
                              channel_axis, *self.axes)

            for ch in channels:
                fill_vals = {name: self.pick_data(method, data, data[ch])
                             for name, method in self.fill_methods.items()}
                if None not in fill_vals.values():
                    if is_mc:
                        _hist.fill(dataset=dsname,
                                   channel=ch,
                                   **fill_vals,
                                   weight=weight[data[ch]])
                    else:
                        _hist.fill(dataset=dsname,
                                   channel=ch,
                                   **fill_vals)
        else:
            _hist = hist.Hist("Counts", self.dataset_axis, *self.axes)
            fill_vals = {name: self.pick_data(method, data)
                         for name, method in self.fill_methods.items()}
            if None not in fill_vals.values():
                if is_mc:
                    _hist.fill(dataset=dsname,
                               **fill_vals,
                               weight=weight)
                else:
                    _hist.fill(dataset=dsname,
                               **fill_vals)
        return _hist

    def pick_data(self, method, data, mask=None):
        for sel in method:
            if isinstance(sel, str):
                if sel in data:
                    data = data[sel]
                else:
                    break
            elif isinstance(sel, dict):
                if "function" in sel:
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
            data = None
        if data is not None and mask is not None:
            return data[mask]
        return data


func_dict = {"jet_mult": jet_mult}
