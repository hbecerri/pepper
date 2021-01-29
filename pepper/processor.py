import os
from functools import partial
import awkward as ak
import coffea
import h5py
import logging
from copy import copy
from time import time
import abc

import pepper
from pepper import Selector, OutputFiller, HDF5File
import pepper.config


logger = logging.getLogger(__name__)


class Processor(coffea.processor.ProcessorABC):
    def __init__(self, config, destdir):
        """Create a new Processor

        Arguments:
        config -- A Config instance, defining the configuration to use
        destdir -- Destination directory, where the event HDF5s are saved.
                   Every chunk will be saved in its own file. If `None`,
                   nothing will be saved.
        """
        self._check_config_integrity(config)
        self.config = config
        if destdir is not None:
            self.destdir = os.path.realpath(destdir)
        else:
            self.destdir = None
        self.hists = self._get_hists_from_config(
            self.config, "hists", "hists_to_do")

        # Construct a dict of datasets for which we are not processing the
        # relevant dedicated systematic datasets, and so the nominal histograms
        # need to be copied
        self.copy_nominal = {}
        for sysdataset, syst in self.config["dataset_for_systematics"].items():
            replaced, sysname = syst
            if sysname not in self.copy_nominal:
                self.copy_nominal[sysname] = []
                # Copy all normal mc datasets
                for dataset in self.config["mc_datasets"].keys():
                    if dataset in self.config["dataset_for_systematics"]:
                        continue
                    self.copy_nominal[sysname].append(dataset)
            try:
                # Remove the ones that get replaced by a dedicated dataset
                self.copy_nominal[sysname].remove(replaced)
            except ValueError:
                pass

        if "randomised_parameter_scan_datasets" in config:
            self.rps_datasets = config["randomised_parameter_scan_datasets"]
        else:
            self.rps_datasets = []

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

    @property
    def accumulator(self):
        self._accumulator = coffea.processor.dict_accumulator({
            "hists": coffea.processor.dict_accumulator(),
            "cutflows": coffea.processor.dict_accumulator(),
        })
        return self._accumulator

    def postprocess(self, accumulator):
        return accumulator

    def _open_output(self, dsname):
        dsname = dsname.replace("/", "_")
        dsdir = os.path.join(self.destdir, dsname)
        os.makedirs(dsdir, exist_ok=True)
        i = 0
        while True:
            filepath = os.path.join(dsdir, str(i).zfill(4) + ".hdf5")
            try:
                f = h5py.File(filepath, "x")
            except (FileExistsError, OSError):
                i += 1
                continue
            else:
                break
        logger.debug(f"Opened output {filepath}")
        return f

    def _prepare_saved_columns(self, selector):
        columns = {}
        for specifier in self.config["columns_to_save"]:
            datapicker = pepper.hist_defns.DataPicker(specifier)
            item = datapicker(selector.data)
            # Strip rows that are not selected from memory
            item = ak.flatten(item, axis=0)
            if item is None:
                logger.info("Skipping to save column because it is not "
                            f"present: {specifier}")
                continue
            key = datapicker.name
            if key in columns:
                raise pepper.config.ConfigError(
                    f"Ambiguous column to save '{key}', (from {specifier})")
            columns[key] = item
        return ak.Array(columns)

    def _save_per_event_info(
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
        with self._open_output(dsname) as f:
            outf = HDF5File(f)
            for key in out_dict.keys():
                outf[key] = out_dict[key]

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
        if dsname in self.rps_datasets:
            return self.process_rps(
                data, dsname, filename, entrystart, entrystop)
        is_mc = (dsname in self.config["mc_datasets"].keys())

        filler = self.setup_outputfiller(dsname, is_mc)
        selector = self.setup_selection(data, dsname, is_mc, filler)
        self.process_selection(selector, dsname, is_mc, filler)

        if self.destdir is not None:
            logger.debug("Saving per event info")
            self._save_per_event_info(
                dsname, selector, self.get_identifier(selector))

        timetaken = time() - starttime
        logger.debug(f"Processing finished. Took {timetaken:.3f} s.")
        return filler.output

    def process_rps(self, data, dsname, filename, entrystart, entrystop):
        logger.debug("This is a randomised parameter signal sample- processing"
                     " each scan point separately")
        is_mc = (dsname in self.config["mc_datasets"].keys())
        scanpoints = [key for key in data.columns
                      if key.startswith("GenModel_")]
        output = self.accumulator.identity()
        for sp in scanpoints:
            logger.debug(f"Processing scan point {sp}")
            dsname = sp[9:]
            filler = self.setup_outputfiller(dsname, is_mc)
            selector = self.setup_selection(copy(data), dsname, is_mc, filler)
            selector.add_cut(partial(self.pick_scan_point, sp),
                             "Select scan point", no_callback=True)
            self.process_selection(selector, dsname, is_mc, filler)
            output += filler.output
            if self.destdir is not None:
                logger.debug("Saving per event info")
                self._save_per_event_info(
                    dsname, selector, (filename, entrystart, entrystop))

        logger.debug("Processing finished")
        return output

    def pick_scan_point(self, sp, data):
        return data[sp]

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

        filler = OutputFiller(
            output, self.hists, is_mc, dsname, dsname_in_hist, sys_enabled,
            sys_overwrite=sys_overwrite, copy_nominal=self.copy_nominal,
            cuts_to_histogram=cuts_to_histogram)

        return filler

    def setup_selection(self, data, dsname, is_mc, filler):
        if is_mc:
            genweight = data["genWeight"]
        else:
            genweight = None
        selector = Selector(data, genweight, filler.get_callbacks())
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
