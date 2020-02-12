#!/usr/bin/env python3

import os
import numpy as np
import coffea
import awkward
from collections import namedtuple


def get_evaluator(filename):
    extractor = coffea.lookup_tools.extractor()
    extractor.import_file(filename)
    # No plausible API available yet, use non-public variable
    for key, value in extractor._filecache[filename].items():
        extractor.add_weight_set(key[0], key[1], value)
    extractor.finalize()
    return extractor.make_evaluator()


WpTuple = namedtuple("WpTuple", ("loose", "medium", "tight"))
BTAG_WP_CUTS = {
    "deepcsv": {
        "2016": WpTuple(0.2217,  0.6321,  0.8953),
        "2017": WpTuple(0.1522,  0.4941,  0.8001),
        "2018": WpTuple(0.1241,  0.4184,  0.7527),
    },
    "deepjet": {
        "2016": WpTuple(0.0614,  0.3093,  0.7221),
        "2017": WpTuple(0.0521,  0.3033,  0.7489),
        "2018": WpTuple(0.0494,  0.2770,  0.7264),
    }
}


class BTagWeighter(object):
    def __init__(self, sf_filename, eff_filename, tagger, year):
        self.eff_evaluator = get_evaluator(eff_filename)
        self.sf_evaluator = get_evaluator(sf_filename)

        # Tagger name of CSV is unknown, have not API for it. Workaround
        somekey = next(iter(self.sf_evaluator.keys()))
        self.csvtaggername = somekey.split("_")[0]

        self.wps = BTAG_WP_CUTS[tagger][year]

    def _sf_func(self, wp, sys, jf_num):
        for measurement in ("mujets", "comb", "incl"):
            key = "{}_{}_{}_{}_{}".format(
                self.csvtaggername, wp, measurement, sys, jf_num)
            if key in self.sf_evaluator:
                return self.sf_evaluator[key]
        return None

    def __call__(
            self, wp, jf, eta, pt, discr, variation="central"):
        if isinstance(wp, str):
            wp = wp.lower()
            if wp == "loose":
                wp = 0
            elif wp == "medium":
                wp = 1
            elif wp == "tight":
                wp = 2
            else:
                raise ValueError("Invalid value for wp. Expected 'loose', "
                                 "'medium' or 'tight'")
        elif not isinstance(wp, int):
            raise TypeError("Expected int or str for wp, got {}".format(wp))

        counts = pt.counts
        jf = jf.flatten()
        eta = eta.flatten()
        pt = pt.flatten()
        discr = discr.flatten()

        sf = np.ones_like(eta)
        # Workaround: Call sf function three times, see
        # https://github.com/CoffeaTeam/coffea/issues/205
        sf[jf == 0] = self._sf_func(wp, variation, 2)(eta, pt, discr)[jf == 0]
        sf[jf == 4] = self._sf_func(wp, variation, 1)(eta, pt, discr)[jf == 4]
        sf[jf == 5] = self._sf_func(wp, variation, 0)(eta, pt, discr)[jf == 5]
        sf = awkward.JaggedArray.fromcounts(counts, sf)

        eff = self.eff_evaluator["efficiency"](jf, pt, abs(eta))
        eff = awkward.JaggedArray.fromcounts(counts, eff)
        sfeff = sf * eff

        is_tagged = discr > self.wps[wp]
        is_tagged = awkward.JaggedArray.fromcounts(counts, is_tagged)

        p_mc = eff[is_tagged].prod() * (1 - eff)[~is_tagged].prod()
        p_data = sfeff[is_tagged].prod() * (1 - sfeff)[~is_tagged].prod()

        # TODO: What if one runs into numerical problems here?
        return p_mc / p_data
