import json
import logging
import os
from functools import partial

import numpy as np
import awkward as ak
import coffea

import pepper


logger = logging.getLogger(__name__)


class DYOutputFiller(pepper.OutputFiller):
    def fill_cutflows(self, data, systematics, cut, done_steps):
        if systematics is not None:
            weight = systematics["weight"]
        else:
            weight = ak.Array(np.ones(len(data)))
        logger.info("Filling cutflow. Current event count: "
                    + str(ak.sum(weight)))
        self.fill_accumulator(self.output["cutflows"], cut, data, weight)
        if systematics is not None:
            weight = systematics["weight"] ** 2
        else:
            weight = ak.Array(np.ones(len(data)))
        self.fill_accumulator(self.output["cutflow_errs"], cut, data, weight)

    def fill_accumulator(self, accumulator, cut, data, weight):
        if "all" not in accumulator:
            accumulator["all"] = coffea.processor.defaultdict_accumulator(
                partial(coffea.processor.defaultdict_accumulator, int))
        if cut not in accumulator["all"][self.dsname]:
            accumulator["all"][self.dsname][cut] = ak.sum(weight)
        for ch in self.channels:
            if ch not in accumulator:
                accumulator[ch] = coffea.processor.defaultdict_accumulator(
                    partial(coffea.processor.defaultdict_accumulator, int))
            if cut not in accumulator[ch][self.dsname]:
                accumulator[ch][self.dsname][cut] = ak.sum(weight[data[ch]])


class DYprocessor(pepper.ProcessorTTbarLL):
    @property
    def accumulator(self):
        self._accumulator = coffea.processor.dict_accumulator({
            "hists": coffea.processor.dict_accumulator(),
            "cutflows": coffea.processor.dict_accumulator(),
            "cutflow_errs": coffea.processor.dict_accumulator()
        })
        return self._accumulator

    def preprocess(self, datasets):
        if "fast_dy_sfs" in self.config and self.config["fast_dy_sfs"]:
            # Only process DY MC samples and observed data
            processed = {}
            for key, value in datasets.items():
                if key in self.config["exp_datasets"] or key.startswith("DY"):
                    processed[key] = value
            return processed
        else:
            return datasets

    def setup_outputfiller(self, dsname, is_mc):
        output = self.accumulator.identity()
        sys_enabled = self.config["compute_systematics"]

        if dsname in self.config["dataset_for_systematics"]:
            dsname_in_hist = self.config["dataset_for_systematics"][dsname][0]
            sys_overwrite = self.config["dataset_for_systematics"][dsname][1]
        else:
            dsname_in_hist = dsname
            sys_overwrite = None

        if "cuts_to_histogram" in self.config:
            cuts_to_histogram = self.config["cuts_to_histogram"]
        else:
            cuts_to_histogram = None

        filler = DYOutputFiller(
            output, self.hists, is_mc, dsname, dsname_in_hist, sys_enabled,
            sys_overwrite=sys_overwrite, copy_nominal=self.copy_nominal,
            cuts_to_histogram=cuts_to_histogram)

        return filler

    def z_window(self, data):
        # Don't apply Z window cut, as we'll add columns inside and
        # outside of it later
        return np.full(len(data), True)

    def drellyan_sf_columns(self, filler, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        Z_window = (data["mll"] >= m_min) & (data["mll"] <= m_max)
        channel = {"ee": data["is_ee"],
                   "em": data["is_em"],
                   "mm": data["is_mm"]}
        Z_w = {"in": Z_window, "out": ~Z_window}
        btags = {"0b": (ak.sum(data["Jet"]["btagged"], axis=1) == 0),
                 "1b": (ak.sum(data["Jet"]["btagged"], axis=1) > 0)}
        if ("bin_dy_sfs" in self.config and
                self.config["bin_dy_sfs"] is not None):
            bin_axis = pepper.hist_defns.DataPicker(self.config["bin_dy_sfs"])
            edges = self.config["dy_sf_bin_edges"]
            bins = {str(edges[i]) + "_to_" + str(edges[i+1]):
                    (bin_axis >= edges[i]) & (bin_axis < edges[i+1])
                    for i in range(len(edges) - 1)}
        else:
            bin_axis = None
        new_chs = {}
        for ch in channel.items():
            for Zw in Z_w.items():
                for btag in btags.items():
                    if bin_axis is not None:
                        for _bin in bins.items():
                            key = (ch[0] + "_" + Zw[0] + "_" + btag[0]
                                   + "_" + _bin[0])
                            new_chs[key] = ch[1] & Zw[1] & btag[1] & _bin[1]
                    else:
                        new_chs[ch[0] + "_" + Zw[0] + "_" + btag[0]] = (
                                ch[1] & Zw[1] & btag[1])
        filler.channels = new_chs.keys()
        return new_chs

    def btag_cut(self, is_mc, data):
        return np.full(len(data), True)

    def apply_dy_sfs(self, dsname, data):
        # Don't apply old DY SFs if these are still in config
        return np.full(len(data), True)

    def save_output(self, output, dest):
        with open(os.path.join(dest, "cutflow_errs.json"), "w") as f:
            json.dump(output["cutflow_errs"], f, indent=4)
        super().save_output(output, dest)


if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        DYprocessor, "Run the DY processor to get the "
        "numbers needed for DY SF calculation")
