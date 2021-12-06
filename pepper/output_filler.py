import logging
from copy import copy

import numpy as np
import awkward as ak

import pepper


logger = logging.getLogger(__name__)


class DummyOutputFiller:
    def __init__(self, output):
        self.output = output

    def get_callbacks(self):
        return []


class OutputFiller:
    def __init__(self, output, hist_dict, is_mc, dsname, dsname_in_hist,
                 sys_enabled, sys_overwrite=None, categories=None,
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
        if isinstance(categories, list):
            self.categories = {cat: {"all"} for cat in categories}
        elif isinstance(categories, dict):
            self.categories = copy(categories)
        elif categories is None:
            self.categories = {}
        else:
            raise ValueError(
                "'categories' must be one of a list, a dict, or None")
        if copy_nominal is None:
            self.copy_nominal = {}
        else:
            self.copy_nominal = copy_nominal

    def fill_cutflows(self, data, systematics, cut, done_steps):
        def fill_recursive(cats, weight, mask, accumulator):
            if len(cats) > 0:
                for reg in cats[0]:
                    if reg == "all":
                        new_mask = mask
                    else:
                        new_mask = mask & data[reg]
                    fill_recursive(
                        cats[1:], weight, new_mask, accumulator[reg])
            else:
                if cut not in accumulator[self.dsname]:
                    accumulator[self.dsname][cut] = ak.sum(weight[mask])

        accumulator = self.output["cutflows"]
        if systematics is not None:
            weight = systematics["weight"]
        else:
            weight = ak.Array(np.ones(len(data)))
            if hasattr(data.layout, "bytemask"):
                weight = weight.mask[~ak.is_none(data)]
        cats = list(self.categories.values())
        fill_recursive(cats, weight, np.full(len(data), True), accumulator)

    def fill_hists(self, data, systematics, cut, done_steps):
        if self.cuts_to_histogram is not None:
            if cut not in self.cuts_to_histogram:
                return
        accumulator = self.output["hists"]
        categories = self.categories
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
                        continue
                    try:
                        sys_hist = fill_func(
                            data=data, categories=categories,
                            dsname=self.dsname_in_hist, is_mc=self.is_mc,
                            weight=weight)
                    except pepper.hist_defns.HistFillError:
                        continue
                    accumulator[(cut, histname, sysname)] = sys_hist
            else:
                if (cut, histname) in accumulator:
                    continue
                try:
                    accumulator[(cut, histname)] = fill_func(
                        data=data, categories=categories,
                        dsname=self.dsname_in_hist, is_mc=self.is_mc,
                        weight=weight)
                except pepper.hist_defns.HistFillError:
                    continue

                if do_systematics:
                    for syscol in ak.fields(systematics):
                        if syscol == "weight":
                            continue
                        sysweight = weight * systematics[syscol]
                        hist = fill_func(
                            data=data, categories=categories,
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
