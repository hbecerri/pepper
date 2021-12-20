import logging
import numpy as np
import awkward as ak
import coffea.hist
from coffea.processor import dict_accumulator
import itertools
from collections import defaultdict

import pepper


logger = logging.getLogger(__name__)


class DummyOutputFiller:
    def __init__(self, output):
        self.output = output

    def get_callbacks(self):
        return []


class OutputFiller:
    def __init__(self, hist_dict, is_mc, dsname, dsname_in_hist, sys_enabled,
                 sys_overwrite=None, copy_nominal=None,
                 cuts_to_histogram=None):
        self.output = {
            "hists": {},
            # Cutflows should keep order of cuts. Due to how Coffea accumulates
            # currently for this a dict_accumulator is needed.
            "cutflows": defaultdict(dict_accumulator)
        }
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
        if copy_nominal is None:
            self.copy_nominal = {}
        else:
            self.copy_nominal = copy_nominal

    def fill_cutflows(self, data, systematics, cut, done_steps, cats):
        if self.sys_overwrite is not None:
            return
        accumulator = self.output["cutflows"]
        if cut in accumulator[self.dsname]:
            return
        if systematics is not None:
            weight = systematics["weight"]
        else:
            weight = ak.Array(np.ones(len(data)))
            if hasattr(data.layout, "bytemask"):
                weight = weight.mask[~ak.is_none(data)]
        axes = [coffea.hist.Cat(cat, cat) for cat in cats.keys()]
        hist = coffea.hist.Hist("Counts", *axes)
        if len(cats) > 0:
            for cat_position in itertools.product(*list(cats.values())):
                masks = []
                for pos in cat_position:
                    masks.append(np.asarray(ak.fill_none(data[pos], False)))
                mask = np.bitwise_and.reduce(masks, axis=0)
                count = ak.sum(weight[mask])
                args = {name: pos
                        for name, pos in zip(cats.keys(), cat_position)}
                hist.fill(**args, weight=count)
        else:
            hist.fill(weight=ak.sum(weight))
        count = hist.project().values()[()]
        num_rows = len(data)
        num_masked = ak.sum(ak.is_none(data))
        logger.info(f"Filling cutflow. Current event count: {count} "
                    f"({num_rows} rows, {num_masked} masked)")
        accumulator[self.dsname][cut] = hist

    def fill_hists(self, data, systematics, cut, done_steps, cats):
        if self.cuts_to_histogram is not None:
            if cut not in self.cuts_to_histogram:
                return
        accumulator = self.output["hists"]
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
                            data=data, categorizations=cats,
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
                        data=data, categorizations=cats,
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
                            data=data, categorizations=cats,
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

    @property
    def channels(self):
        raise AttributeError("'channels' is not used anymore. Use "
                             "Selector.set_cat('channels', [...])")
