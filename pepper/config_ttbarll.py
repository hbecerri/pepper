#!/usr/bin/env python3

import numpy as np
import uproot
import hjson

import pepper
from pepper.scale_factors import ScaleFactors


class ConfigTTbarLL(pepper.ConfigBasicPhysics):
    def __init__(self, path_or_file, textparser=hjson.load, cwd="."):
        """Initialize the configuration.

        Arguments:
        path_or_file -- Either a path to the file containing the configuration
                        or a file-like object of it
        textparser -- Callable to be used to parse the text contained in
                      path_or_file
        cwd -- A path to use as the working directory for relative paths in the
               config. The actual working directory of the process might change
        """
        super().__init__(path_or_file, textparser, cwd)

        self.behaviors.update(
            {
                "drellyan_sf": self._get_drellyan_sf,
                "trigger_sfs": self._get_trigger_sfs,
            }
        )

    def _get_drellyan_sf(self, value):
        if isinstance(value, list):
            path, histname = value
            with uproot.open(self._get_path(path)) as f:
                hist = f[histname]
            dy_sf = ScaleFactors.from_hist(hist)
        else:
            data = self._get_maybe_external(value)
            dy_sf = ScaleFactors(
                bins=data["bins"],
                factors=np.array(data["factors"]),
                factors_up=np.array(data["factors_up"]),
                factors_down=np.array(data["factors_down"]))
        return dy_sf

    def _get_trigger_sfs(self, value):
        path, histnames = value
        ret = {}
        if len(histnames) != 3:
            raise pepper.config.ConfigError(
                "Need 3 histograms for trigger scale factors. Got "
                f"{len(histnames)}")
        with uproot.open(self._get_path(path)) as f:
            for chn, histname in zip(("is_ee", "is_em", "is_mm"), histnames):
                ret[chn] = ScaleFactors.from_hist(
                    f[histname], dimlabels=["lep1_pt", "lep2_pt"])
        return ret
