import os
import numpy as np
import awkward as ak
import uproot
import coffea
from coffea.nanoevents import NanoAODSchema
import h5py
import json
import logging
from time import time
import abc
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm

import pepper
from pepper import Selector, OutputFiller, HDF5File
import pepper.config


logger = logging.getLogger(__name__)


class Processor(coffea.processor.ProcessorABC):
    """Class implementing input/output, setup of histograms, and utility
    classes"""
    config_class = pepper.Config
    schema_class = NanoAODSchema

    def __init__(self, config, eventdir):
        """Create a new Processor

        Arguments:
        config -- A Config instance, defining the configuration to use
        eventdir -- Destination directory, where the event HDF5s are saved.
                    Every chunk will be saved in its own file. If `None`,
                    nothing will be saved.
        """
        self._check_config_integrity(config)
        self.config = config
        if eventdir is not None:
            self.eventdir = os.path.realpath(eventdir)
        else:
            self.eventdir = None

        self.rng_seed = self._load_rng_seed()

    @staticmethod
    def _check_config_integrity(config):
        # Nothing to do here currently. Implemented in subclasses
        pass

    @staticmethod
    def _get_hists_from_config(config, key, todokey):
        if key in config:
            hists = config[key]
        else:
            hists = {}
        if todokey in config and len(config[todokey]) > 0:
            new_hists = {}
            for name in config[todokey]:
                if name in hists:
                    new_hists[name] = hists[name]
            hists = new_hists
            logger.info("Doing only the histograms: " +
                        ", ".join(hists.keys()))
        return hists

    def _load_rng_seed(self):
        if "rng_seed_file" not in self.config:
            return np.random.SeedSequence().entropy
        seed_file = self.config["rng_seed_file"]
        if os.path.exists(seed_file):
            with open(seed_file) as f:
                try:
                    seed = int(f.read())
                except ValueError as e:
                    raise pepper.config.ConfigError(
                        f"Not an int in rng_seed_file '{seed_file}'")\
                        from e
                return seed
        else:
            rng_seed = np.random.SeedSequence().entropy
            with open(seed_file, "w") as f:
                f.write(str(rng_seed))
            return rng_seed

    def preprocess(self, datasets):
        return datasets

    @staticmethod
    def postprocess(accumulator):
        return accumulator

    def _open_output(self, dsname, filetype):
        if filetype == "root":
            ext = ".root"
        elif filetype == "hdf5":
            ext = ".h5"
        else:
            raise ValueError(f"Invalid filetype: {filetype}")
        dsname = dsname.replace("/", "_")
        dsdir = os.path.join(self.eventdir, dsname)
        os.makedirs(dsdir, exist_ok=True)
        i = 0
        while True:
            filepath = os.path.join(dsdir, str(i).zfill(4) + ext)
            try:
                f = open(filepath, "x")
            except (FileExistsError, OSError):
                i += 1
                continue
            else:
                break
        f.close()
        if filetype == "root":
            f = uproot.recreate(filepath)
        elif filetype == "hdf5":
            f = h5py.File(filepath, "w")
        logger.debug(f"Opened output {filepath}")
        return f

    def _prepare_saved_columns(self, selector):
        columns = {}
        if "columns_to_save" in self.config:
            to_save = self.config["columns_to_save"]
        else:
            to_save = []
        if isinstance(to_save, dict):
            spec = to_save.items()
        else:
            spec = zip([None] * len(to_save), to_save)
        for key, specifier in spec:
            datapicker = pepper.hist_defns.DataPicker(specifier)
            item = datapicker(selector.data)
            if item is None:
                logger.info("Skipping to save column because it is not "
                            f"present: {specifier}")
                continue
            if key is None:
                key = datapicker.name
            if key in columns:
                raise pepper.config.ConfigError(
                    f"Ambiguous column to save '{key}', (from {specifier})")
            columns[key] = item
        return ak.Array(columns)

    def _save_per_event_info_hdf5(
            self, dsname, selector, identifier, save_full_sys=True):
        out_dict = {"dsname": dsname, "identifier": identifier}
        out_dict["events"] = self._prepare_saved_columns(selector)
        cutnames, cutflags = selector.get_cuts()
        out_dict["cutnames"] = cutnames
        out_dict["cutflags"] = cutflags
        if (self.config["compute_systematics"] and save_full_sys
                and selector.systematics is not None):
            out_dict["systematics"] = ak.flatten(selector.systematics,
                                                 axis=0)
        elif selector.systematics is not None:
            out_dict["weight"] = ak.flatten(
                selector.systematics["weight"], axis=0)
        with self._open_output(dsname, "hdf5") as f:
            outf = HDF5File(f)
            for key in out_dict.keys():
                outf[key] = out_dict[key]

    @staticmethod
    def _separate_masks_for_root(arrays):
        ret = {}
        for key, array in arrays.items():
            if (not isinstance(array, ak.Array)
                    or not pepper.misc.akismasked(array)):
                ret[key] = array
                continue
            if array.ndim > 2:
                raise ValueError(
                    f"Array '{key}' as too many dimensions for ROOT output")
            if "mask" + key in arrays:
                raise RuntimeError(f"Output named 'mask{key}' already present "
                                   "but need this key for storing the mask")
            ret["mask" + key] = ~ak.is_none(array)
            ret[key] = ak.fill_none(array, 0)
        return ret

    def _save_per_event_info_root(self, dsname, selector, identifier,
                                  save_full_sys=True):
        out_dict = {"dsname": dsname, "identifier": str(identifier)}
        events = self._prepare_saved_columns(selector)
        events = {f: events[f] for f in ak.fields(events)}
        additional = {}
        cutnames, cutflags = selector.get_cuts()
        out_dict["Cutnames"] = str(cutnames)
        additional["cutflags"] = cutflags
        if selector.systematics is not None:
            additional["weight"] = selector.systematics["weight"]
            if self.config["compute_systematics"] and save_full_sys:
                for field in ak.fields(selector.systematics):
                    additional[f"systematics_{field}"] = \
                        selector.systematics[field]

        for key in additional.keys():
            if key in events:
                raise RuntimeError(
                    f"branch named '{key}' already present in Events tree")
        events.update(additional)
        events = self._separate_masks_for_root(events)
        out_dict["Events"] = events
        with self._open_output(dsname, "root") as outf:
            for key in out_dict.keys():
                outf[key] = out_dict[key]

    def save_per_event_info(self, dsname, selector, save_full_sys=True):
        idn = self.get_identifier(selector)
        logger.debug("Saving per event info")
        if "column_output_format" in self.config:
            outformat = self.config["column_output_format"].lower()
        else:
            outformat = "root"
        if outformat == "root":
            self._save_per_event_info_root(
                dsname, selector, idn, save_full_sys)
        elif outformat == "hdf5":
            self._save_per_event_info_hdf5(
                dsname, selector, idn, save_full_sys)
        else:
            raise pepper.config.ConfigError(
                "Invalid value for column_output_format, must be 'root' "
                "or 'hdf'")

    @staticmethod
    def get_identifier(data):
        meta = data.metadata
        return meta["filename"], meta["entrystart"], meta["entrystop"]

    def process(self, data):
        starttime = time()
        dsname = data.metadata["dataset"]
        filename = data.metadata["filename"]
        entrystart = data.metadata["entrystart"]
        entrystop = data.metadata["entrystop"]
        logger.debug(f"Started processing {filename} from event "
                     f"{entrystart} to {entrystop - 1} for dataset {dsname}")
        is_mc = (dsname in self.config["mc_datasets"].keys())

        filler = self.setup_outputfiller(dsname, is_mc)
        selector = self.setup_selection(data, dsname, is_mc, filler)
        self.process_selection(selector, dsname, is_mc, filler)

        if self.eventdir is not None:
            self.save_per_event_info(dsname, selector)

        timetaken = time() - starttime
        logger.debug(f"Processing finished. Took {timetaken:.3f} s.")
        return filler.output

    def setup_outputfiller(self, dsname, is_mc):
        sys_enabled = self.config["compute_systematics"]

        if dsname in self.config["dataset_for_systematics"]:
            dsname_in_hist = self.config["dataset_for_systematics"][dsname][0]
            sys_overwrite = self.config["dataset_for_systematics"][dsname][1]
        elif ("datasets_to_group" in self.config
              and dsname in self.config["datasets_to_group"]):
            dsname_in_hist = self.config["datasets_to_group"][dsname]
            sys_overwrite = None
        else:
            dsname_in_hist = dsname
            sys_overwrite = None

        if "cuts_to_histogram" in self.config:
            cuts_to_histogram = self.config["cuts_to_histogram"]
        else:
            cuts_to_histogram = None

        hists = self._get_hists_from_config(
            self.config, "hists", "hists_to_do")
        filler = OutputFiller(
            hists, is_mc, dsname, dsname_in_hist, sys_enabled,
            sys_overwrite=sys_overwrite, cuts_to_histogram=cuts_to_histogram)

        return filler

    def setup_selection(self, data, dsname, is_mc, filler):
        if is_mc:
            genweight = data["genWeight"]
        else:
            genweight = None
        # Use a different seed for every chunk in a reproducable way
        seed = (self.rng_seed, uuid.UUID(data.metadata["fileuuid"]).int,
                data.metadata["entrystart"])
        selector = Selector(data, genweight, filler.get_callbacks(),
                            rng_seed=seed)
        return selector

    @abc.abstractmethod
    def process_selection(self, selector, dsname, is_mc, filler):
        """Do selection steps, e.g. cutting, defining objects

        Arguments:
        selector -- A pepper.Selector object with the event data
        dsname -- Name of the dataset from config
        is_mc -- Bool, whether events come from Monte Carlo
        filler -- pepper.OutputFiller object to controll how the output is
                  structured if needed
        """

    @staticmethod
    def _get_cuts(output):
        cutflow_all = output["cutflows"]
        cut_lists = [list(cutflow.keys()) for cutflow
                     in cutflow_all.values()]
        cuts_precursors = defaultdict(set)
        for cut_list in cut_lists:
            for i, cut in enumerate(cut_list):
                cuts_precursors[cut].update(set(cut_list[:i]))
        cuts = []
        while len(cuts_precursors) > 0:
            for cut, precursors in cuts_precursors.items():
                if len(precursors) == 0:
                    cuts.append(cut)
                    for p in cuts_precursors.values():
                        p.discard(cut)
                    cuts_precursors.pop(cut)
                    break
            else:
                raise ValueError("No well-defined ordering of cuts "
                                 "for all datasets found")
        return cuts

    @staticmethod
    def _prepare_cutflows(output):
        cutflows = output["cutflows"]
        output = {}
        for dataset, cf1 in cutflows.items():
            output[dataset] = {"all": defaultdict(float)}
            for cut, cf2 in cf1.items():
                cf = cf2.values()
                for cat_position, value in cf.items():
                    output_for_cat = output[dataset]
                    for cat_coordinate in cat_position:
                        if cat_coordinate not in output_for_cat:
                            output_for_cat[cat_coordinate] = {}
                        output_for_cat = output_for_cat[cat_coordinate]
                    if len(cat_position) > 0:
                        output_for_cat[cut] = float(value)
                    output[dataset]["all"][cut] += value
        return output

    @staticmethod
    def _save_hist_hists(key, histdict, dest, cuts):
        hist_sum = None
        for dataset, hist in histdict.items():
            if hist_sum is None:
                hist_sum = hist.copy()
            else:
                hist_sum += hist
        cutnum = cuts.index(key[0])
        fname = "Cut {:03} {}.coffea".format(cutnum, "_".join(key))
        fname = fname.replace("/", "")
        coffea.util.save(hist_sum, os.path.join(dest, fname))
        return {key: fname}

    @staticmethod
    def _save_coffea_hists(key, histdict, dest, cuts):
        hists = {}
        for dataset, hist in histdict.items():
            if "sys" in [ax.name for ax in hist.axes]:
                for sysname in hist.axes["sys"]:
                    cofhist = pepper.misc.hist2coffeahist(
                        hist[{"sys": sysname}])
                    new_key = key if sysname == "nominal" else key + (sysname,)
                    if new_key in hists:
                        hists[new_key].add(cofhist)
                    else:
                        hists[new_key] = cofhist
            else:
                cofhist = pepper.misc.hist2coffeahist(hist)
                if key in hists:
                    hists[key].add(cofhist)
                else:
                    hists[key] = cofhist
        cutnum = cuts.index(key[0])
        fnames = {}
        for new_key, hist in hists.items():
            fname = "Cut {:03} {}.coffea".format(cutnum, "_".join(new_key))
            fname = fname.replace("/", "")
            coffea.util.save(hist, os.path.join(dest, fname))
            fnames[new_key] = fname
        return fnames

    @staticmethod
    def _save_root_hists(key, histdict, dest):
        fnames = {}
        for dataset, hist in histdict.items():
            if "sys" in [ax.name for ax in hist.axes]:
                sysnames = hist.axes["sys"]
            else:
                sysnames = [None]
            for sysname in sysnames:
                if sysname is not None:
                    histsys = hist[{"sys": sysname}]
                else:
                    histsys = hist
                histsplits = pepper.misc.hist_split_strcat(histsys)
                if sysname is None or sysname == "nominal":
                    fullkey = key
                else:
                    fullkey = key + (sysname,)
                fname = '_'.join(fullkey).replace('/', '_') + ".root"
                with uproot.recreate(os.path.join(dest, fname)) as f:
                    for catkey, histsplit in histsplits.items():
                        catkey = "_".join(catkey).replace("/", "_")
                        f[catkey] = histsplit
                fnames[fullkey] = fname
        return fnames

    @classmethod
    def save_histograms(cls, format, output, dest, threads=10):
        cuts = cls._get_cuts(output)
        hists = defaultdict(dict)
        for dataset, hists_per_ds in output["hists"].items():
            for key, hist in hists_per_ds.items():
                hists[key][dataset] = hist
        with ProcessPoolExecutor(max_workers=threads) as executor:
            futures = []
            if format == "hist":
                for key, histdict in hists.items():
                    futures.append(executor.submit(
                        cls._save_hist_hists, key, histdict, dest, cuts))
            elif format == "coffea":
                for key, histdict in hists.items():
                    futures.append(executor.submit(
                        cls._save_coffea_hists, key, histdict, dest, cuts))
            elif format == "root":
                for key, histdict in hists.items():
                    futures.append(executor.submit(
                        cls._save_root_hists, key, histdict, dest))

            hist_names = {}
            for future in tqdm(concurrent.futures.as_completed(futures),
                               desc="Saving histograms", total=len(futures)):
                hist_names.update(future.result())
        with open(os.path.join(dest, "hists.json"), "w") as f:
            json.dump([[tuple(k) for k in hist_names.keys()],
                       list(hist_names.values())], f, indent=4)

    def save_output(self, output, dest):
        # Save cutflows
        with open(os.path.join(dest, "cutflows.json"), "w") as f:
            json.dump(self._prepare_cutflows(output), f, indent=4)

        if "histogram_format" in self.config:
            hform = self.config["histogram_format"].lower()
        else:
            hform = "hist"
        if hform not in ["coffea", "root", "hist"]:
            logger.warning(
                f"Invalid histogram format: {hform}. Saving as hist")
            hform = "hist"
        hist_dest = os.path.join(dest, "hists")
        os.makedirs(hist_dest, exist_ok=True)
        self.save_histograms(hform, output, hist_dest)
