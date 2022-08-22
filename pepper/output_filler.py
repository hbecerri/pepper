import logging
import numpy as np
import awkward as ak
import hist as hi
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
                 sys_overwrite=None, cuts_to_histogram=None):
        self.output = {
            "hists": {},
            # Cutflows should keep order of cuts. Due to how Coffea accumulates
            # currently for this a dict_accumulator is needed.
            "cutflows": defaultdict(dict)
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
        self.done_hists = set()

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
        data_fields = ak.fields(data)
        # Skip cats that are missing in data
        cats = {k: v for k, v in cats.items()
                if all(field in data_fields for field in v)}
        axes = [hi.axis.StrCategory([], name=cat, label=cat, growth=True)
                for cat in cats.keys()]
        hist = hi.Hist(hi.axis.Integer(0, 1), *axes, storage="Weight")
        if len(cats) > 0:
            for cat_position in itertools.product(*list(cats.values())):
                masks = []
                for pos in cat_position:
                    masks.append(np.asarray(ak.fill_none(data[pos], False)))
                mask = np.bitwise_and.reduce(masks, axis=0)
                count = ak.sum(weight[mask])
                args = {name: pos
                        for name, pos in zip(cats.keys(), cat_position)}
                hist.fill(0, **args, weight=count)
        else:
            hist.fill(0, weight=ak.sum(weight))
        count = hist.values().sum()
        if logger.getEffectiveLevel() <= logging.INFO:
            num_rows = len(data)
            num_masked = ak.sum(ak.is_none(data))
            logger.info(f"Filling cutflow. Current event count: {count} "
                        f"({num_rows} rows, {num_masked} masked)")
        accumulator[self.dsname][cut] = hist

    def _add_hist(self, cut, histname, sysname, dsname, hist):
        acc = self.output["hists"]
        # Split histograms by data set name. Summing histograms of the same
        # data set is generally much faster than summing across data sets
        # because normally the category axes for one data set are always
        # the same. Thus not summing across data sets will increase speed
        # significantly.
        if dsname not in acc:
            acc[dsname] = {}
        if (cut, histname) in acc[dsname]:
            try:
                acc[dsname][(cut, histname)] += hist
            except ValueError as err:
                raise ValueError(
                    f"Error adding sys {sysname} to hist {histname} for cut"
                    f" {cut} due to incompatible axes. This most likely"
                    f" caused by setting a new column after setting a new "
                    f" channel and a new systematic, consider setting "
                    f"'no_callback=True' on all new columns set before the"
                    f" next cut") from err
        else:
            acc[dsname][(cut, histname)] = hist
        self.done_hists.add((cut, histname, sysname))

    def fill_hists(self, data, systematics, cut, done_steps, cats):
        if self.cuts_to_histogram is not None:
            if cut not in self.cuts_to_histogram:
                return
        do_systematics = self.sys_enabled and systematics is not None
        if do_systematics and self.sys_overwrite is None:
            weight = {}
            for syscol in ak.fields(systematics):
                if syscol == "weight":
                    sysname = "nominal"
                    sysweight = systematics["weight"]
                else:
                    sysname = syscol
                    sysweight = systematics["weight"] * systematics[syscol]
                weight[sysname] = sysweight
        elif systematics is not None:
            weight = systematics["weight"]
        elif self.sys_enabled:
            weight = {"nominal": None}
        else:
            weight = None
        for histname, fill_func in self.hist_dict.items():
            if (fill_func.step_requirement is not None
                    and fill_func.step_requirement not in done_steps):
                continue
            try:
                if self.sys_overwrite is not None:
                    sysname = self.sys_overwrite
                    # But only if we want to compute systematics
                    if do_systematics:
                        if (cut, histname, sysname) in self.done_hists:
                            continue

                        sys_hist = fill_func(
                            data=data, categorizations=cats,
                            dsname=self.dsname_in_hist, is_mc=self.is_mc,
                            weight={sysname: weight})
                        self._add_hist(cut, histname, sysname, self.dsname,
                                       sys_hist)
                else:
                    if (cut, histname, None) in self.done_hists:
                        continue
                    hist = fill_func(
                        data=data, categorizations=cats,
                        dsname=self.dsname_in_hist, is_mc=self.is_mc,
                        weight=weight)
                    self._add_hist(cut, histname, None, self.dsname, hist)
            except pepper.hist_defns.HistFillError:
                # Ignore if fill is missing in data
                continue

    def get_callbacks(self):
        return [self.fill_cutflows, self.fill_hists]

    @property
    def channels(self):
        raise AttributeError("'channels' is not used anymore. Use "
                             "Selector.set_cat('channel', [...])")
