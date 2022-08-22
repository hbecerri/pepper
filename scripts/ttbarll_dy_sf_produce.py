import logging

import numpy as np
import awkward as ak

import pepper


logger = logging.getLogger("pepper")


class DYprocessor(pepper.ProcessorTTbarLL):
    def __init__(self, config, eventdir):
        """Create a DY Processor

        Arguments:
        config -- A Config instance, defining the configuration to use
        eventdir -- Destination directory, where the event HDF5s are saved.
                    Every chunk will be saved in its own file. If `None`,
                    nothing will be saved.
        """
        if "hists" not in config or len(config["hists"]) == 0:
            config["hists"] = {"DY_dummy_hist": pepper.HistDefinition(
                {"bins": [{"name": "pt", "label": "MET", "n_or_arr": 1,
                           "lo": 0, "hi": 1000, "unit": "GeV"}],
                 "fill": {"pt": ["MET", "pt"]}})}
        super().__init__(config, eventdir)

    def preprocess(self, datasets):
        if "fast_dy_sfs" in self.config and not self.config["fast_dy_sfs"]:
            return datasets
        else:
            # Only process DY MC samples and observed data
            processed = {}
            for key, value in datasets.items():
                if (key in self.config["exp_datasets"] or
                        self.is_dy_dataset(key)):
                    processed[key] = value
            return processed

    def z_window(self, data):
        # Don't apply Z window cut, as we'll add columns inside and
        # outside of it later
        return np.full(len(data), True)

    def drellyan_sf_columns(self, selector, data):
        m_min = self.config["z_boson_window_start"]
        m_max = self.config["z_boson_window_end"]
        Z_window = (data["mll"] >= m_min) & (data["mll"] <= m_max)
        new_chs = {"in_Z_win": Z_window, "out_Z_win": ~Z_window,
                   "0b": (ak.sum(data["Jet"]["btagged"], axis=1) == 0),
                   "1b": (ak.sum(data["Jet"]["btagged"], axis=1) > 0)}
        selector.set_cat("in_Z_window", {"in_Z_win", "out_Z_win"})
        selector.set_cat("btags", {"0b", "1b"})
        return new_chs

    def btag_cut(self, is_mc, data):
        if is_mc and (
                "btag_sf" in self.config and len(self.config["btag_sf"]) != 0):
            num_btagged = ak.sum(data["Jet"]["btagged"], axis=1)
            accept = np.asarray(
                num_btagged >= self.config["num_atleast_btagged"])
            ret = np.full(len(data), 1, dtype=float)
            weight, systematics = self.compute_weight_btag(data[accept])
            ret[accept] *= np.asarray(weight)
            return ret
        else:
            return np.full(len(data), True)

    def apply_dy_sfs(self, dsname, data):
        # Don't apply old DY SFs if these are still in config
        return np.full(len(data), True)


if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(
        DYprocessor, "Run the DY processor to fill the histogram "
        "needed for DY SF calculation")
