from functools import partial
import numpy as np
import awkward as ak
import coffea
import logging

import pepper


logger = logging.getLogger(__name__)


class DummyOutputFiller:
    def __init__(self, output):
        self.output = output

    def get_callbacks(self):
        return []


class OutputFiller:
    def __init__(self, output, hist_dict, is_mc, dsname, dsname_in_hist,
                 sys_enabled, sys_overwrite=None, channels=None,
                 copy_nominal=None, cuts_to_histogram=None):
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
        self.cuts_to_histogram = cuts_to_histogram
        if channels is None:
            self.channels = tuple()
        else:
            self.channels = channels
        if copy_nominal is None:
            self.copy_nominal = {}
        else:
            self.copy_nominal = copy_nominal

    def fill_cutflows(self, data, systematics, cut, done_steps):
        accumulator = self.output["cutflows"]
        if systematics is not None:
            weight = systematics["weight"]
        else:
            weight = ak.Array(np.ones(len(data)))
        if "all" not in accumulator:
            accumulator["all"] = coffea.processor.defaultdict_accumulator(
                partial(coffea.processor.defaultdict_accumulator, int))
        if cut not in accumulator["all"][self.dsname]:
            count = ak.sum(weight)
            accumulator["all"][self.dsname][cut] = count
            rows = len(weight)
            masked = ak.sum(ak.is_none(weight))
            logger.info(
                "Filling cutflow. Current event count: {} ({} rows, {} "
                "masked)".format(count, rows, masked))
        for ch in self.channels:
            if ch not in accumulator:
                accumulator[ch] = coffea.processor.defaultdict_accumulator(
                    partial(coffea.processor.defaultdict_accumulator, int))
            if cut not in accumulator[ch][self.dsname]:
                accumulator[ch][self.dsname][cut] = ak.sum(weight[data[ch]])

    def fill_hists(self, data, systematics, cut, done_steps):
        if self.cuts_to_histogram is not None:
            if cut not in self.cuts_to_histogram:
                return
        accumulator = self.output["hists"]
        channels = self.channels
        do_systematics = self.sys_enabled and systematics is not None
        if systematics is not None:
            weight = systematics["weight"]
        else:
            weight = None
        for histname, fill_func in self.hist_dict.items():
            if (fill_func.step_requirement is not None
                    and fill_func.step_requirement not in done_steps):
                continue
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
                    for syscol in ak.fields(systematics):
                        if syscol == "weight":
                            continue
                        sysweight = weight * systematics[syscol]
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
