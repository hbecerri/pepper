from functools import partial
import numpy as np
import coffea
import logging

import pepper


logger = logging.getLogger(__name__)


class DummyOutputFiller():
    def __init__(self, output):
        self.output = output

    def get_callbacks(self):
        return []


class OutputFiller():
    def __init__(self, output, hist_dict, is_mc, dsname, dsname_in_hist,
                 sys_enabled, sys_overwrite=None, channels=None,
                 copy_nominal=None):
        self.output = output
        if hist_dict is None:
            self.hist_dict = {}
        else:
            self.hist_dict = hist_dict
        self.is_mc = is_mc
        self.dsname = dsname
        self.dsname_in_hist = dsname_in_hist
        self.sys_enabled = sys_enabled
        self.sys_overwrite = sys_overwrite
        if channels is None:
            self.channels = tuple()
        else:
            self.channels = channels
        if copy_nominal is None:
            self.copy_nominal = {}
        else:
            self.copy_nominal = copy_nominal

    def update_ds(self, dsname, dsname_in_hist, channels):
        self.dsname = dsname
        self.dsname_in_hist = dsname_in_hist
        if channels is None:
            self.channels = tuple()
        else:
            self.channels = channels

    def fill_cutflows(self, data, systematics, cut):
        accumulator = self.output["cutflows"]
        if systematics is not None:
            weight = systematics["weight"].flatten()
        else:
            weight = np.ones(data.size)
        if "all" not in accumulator:
            accumulator["all"] = coffea.processor.defaultdict_accumulator(
                partial(coffea.processor.defaultdict_accumulator, int))
        if cut not in accumulator["all"][self.dsname]:
            accumulator["all"][self.dsname][cut] = weight.sum()
            logger.info("Filling cutflow. Current event count: " +
                        str(accumulator["all"][self.dsname][cut]))
        for ch in self.channels:
            if ch not in accumulator:
                accumulator[ch] = coffea.processor.defaultdict_accumulator(
                    partial(coffea.processor.defaultdict_accumulator, int))
            if cut not in accumulator[ch][self.dsname]:
                accumulator[ch][self.dsname][cut] = weight[data[ch]].sum()

    def fill_hists(self, data, systematics, cut):
        accumulator = self.output["hists"]
        channels = self.channels
        do_systematics = self.sys_enabled and systematics is not None
        if systematics is not None:
            weight = systematics["weight"].flatten()
        else:
            weight = None
        for histname, fill_func in self.hist_dict.items():
            if self.sys_overwrite is not None:
                sysname = self.sys_overwrite
                # But only if we want to compute systematics
                if do_systematics:
                    if (cut, histname, sysname) in accumulator:
                        hist = accumulator[(cut, histname, sysname)]
                        if pepper.misc.hist_counts(hist) > 0:
                            continue
                    sys_hist = fill_func(
                        data=data, channels=channels,
                        dsname=self.dsname_in_hist, is_mc=self.is_mc,
                        weight=weight)
                    accumulator[(cut, histname, sysname)] = sys_hist
            else:
                if (cut, histname) in accumulator:
                    hist = accumulator[(cut, histname)]
                    if pepper.misc.hist_counts(hist) > 0:
                        continue
                accumulator[(cut, histname)] = fill_func(
                    data=data, channels=channels, dsname=self.dsname_in_hist,
                    is_mc=self.is_mc, weight=weight)

                if do_systematics:
                    for syscol in systematics.columns:
                        if syscol == "weight":
                            continue
                        sysweight = weight * systematics[syscol].flatten()
                        hist = fill_func(
                            data=data, channels=channels,
                            dsname=self.dsname_in_hist, is_mc=self.is_mc,
                            weight=sysweight)
                        accumulator[(cut, histname, syscol)] = hist
                    for sys, affected_datasets in self.copy_nominal.items():
                        if self.dsname not in affected_datasets:
                            continue
                        accumulator[(cut, histname, sys)] =\
                            accumulator[(cut, histname)].copy()

    def get_callbacks(self):
        return [self.fill_cutflows, self.fill_hists]
