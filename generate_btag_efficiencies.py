#!/usr/bin/env python3

import os
import sys
from collections import namedtuple
import json
import coffea
import pepper
import uproot3
from argparse import ArgumentParser
from collections.abc import Mapping


class HistCollection(dict):
    class Key(namedtuple(
            "HistCollectionKeyBase", ["cut", "hist", "variation"])):
        def __new__(cls, cut=None, hist=None, variation=None):
            return cls.__bases__[0].__new__(cls, cut, hist, variation)

        def fitsto(self, **kwargs):
            for key, value in kwargs.items():
                if getattr(self, key) != value:
                    return False
            else:
                return True

    def __init__(self, *args, **kwargs):
        self._path = kwargs["path"]
        del kwargs["path"]
        super().__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, fileobj):
        data = json.load(fileobj)
        path = os.path.dirname(os.path.realpath(fileobj.name))
        return cls({cls.Key(*k): v for k, v in zip(*data)}, path=path)

    def __getitem__(self, key):
        if isinstance(key, self.Key):
            return super().__getitem__(key)
        elif isinstance(key, Mapping):
            ret = self.__class__({k: v for k, v in self.items()
                                  if k.fitsto(**key)}, path=self._path)
            if len(ret) == 0:
                raise KeyError(key)
            elif len(key) == len(self.Key._fields):
                ret = next(iter(ret.values()))
            return ret
        elif isinstance(key, tuple):
            return self[dict(zip(self.Key._fields, key))]
        else:
            return self[{self.Key._fields[0]: key}]

    def load(self, key):
        return coffea.util.load(os.path.join(self._path, self[key]))


parser = ArgumentParser(
    description="Generate a ROOT file containing efficiency histograms needed "
    "for computing b-tagging scale factors. This requires a specific "
    "histogram to have been computed, see the btageff histogram in "
    "example/hist_config.json")
parser.add_argument("histsfile", help="A JSON file specifying the histograms, "
                                      "e.g. 'hists.json'")
parser.add_argument("output", help="Output ROOT file")
parser.add_argument(
    "--cut", default="Has jet(s)", help="Name of the cut before the b-tag "
                                        "requirement. (Default 'Has jet(s)')")
parser.add_argument(
    "--histname", default="btageff", help="Name of the b-tagging efficiency "
                                          "histogram. (Default 'btageff')")
parser.add_argument(
    "--central", action="store_true",
    help="Only output the central efficiency and no additional ones for "
         "systematic variations")
args = parser.parse_args()

if os.path.exists(args.output):
    if input(f"{args.output} exists. Overwrite [y/n] ") != "y":
        sys.exit(1)

with open(args.histsfile) as f:
    hists = HistCollection.from_json(f)

with uproot3.recreate(args.output) as f:
    for key, histpath in hists[dict(cut=args.cut, hist=args.histname)].items():
        hist = hists.load(key).sum("dataset", "channel")
        eff = pepper.misc.hist_divide(
            hist.integrate("btagged", int_range="yes"), hist.sum("btagged"))
        if key.variation is None:
            f["central"] = pepper.misc.export(eff)
        elif args.central:
            continue
        elif any(key.variation.endswith(x) for x in (
                "XS_down", "XS_up", "lumi_down", "lumi_up")):
            # These aren't shape uncertainties and also do not have much effect
            # on the efficiency
            continue
        else:
            f[key[2]] = pepper.misc.export(eff)
