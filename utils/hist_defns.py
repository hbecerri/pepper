import os
import json

from coffea import hist
import numpy as np


def create_hist_dict(config_folder):
    hist_config = json.load(open(os.path.join(config_folder, "hist_config.json")))
    return {key:Hist_def(val) for key, val in hist_config.items()}


class Hist_def():
    def __init__(self, config):
        self.dataset_axis = hist.Cat("dataset", "")
        self.axes = [hist.Bin(**kwargs) for kwargs in config["bins"]]
        if "cats" in config:
            self.axes.extend([hist.Cat(**kwargs) for kwargs in config["cats"]])
        self.fill_methods = config["fill"]

    def __call__(self, data, dsname, is_mc, weight):
        channels=["ee", "emu", "mumu", "None"]
        if "Channel" in data:
            channel_axis = hist.Cat("channel", "")
            _hist = hist.Hist("Counts", self.dataset_axis, channel_axis, *self.axes)
            
            for ch in channels:
                mask = np.where(data["Channel"] == ch)
                fill_vals = {name: self.pick_data(method, data, mask) for name, method in self.fill_methods}
                if None not in fill_vals.values():
                    if is_mc:
                        _hist.fill(dataset=dsname,
                                      channel=ch,
                                      **fill_vals,
                                      weight=weight[mask])
                    else:
                        _hist.fill(dataset=dsname,
                                      channel=ch,
                                      **fill_vals)
        else:
            _hist = hist.Hist("Counts", self.dataset_axis, *self.axes)
            fill_vals = {name: self.pick_data(method, data) for name, method in self.fill_methods}
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
            if type(sel) is str:
                if sel in data:
                    data = data[sel]
                else:
                    data=None
            elif isinstance(sel, dict):
                if "function" in sel:
                    data = sel["function"](data)
                elif "key" in sel:
                    if sel["key"] in data:
                        data = data[sel["key"]]
                    else:
                        data=None
                
                if "slice" in sel:
                    data = data[sel["slice"]]
            else:
                data = None
        if data is not None:
            return data[mask]
        else:
            return None
