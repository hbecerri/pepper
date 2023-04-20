import os
import abc
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy
from functools import partial
import time
import logging
import parsl
from parsl.app.app import python_app
import uproot
from coffea.processor.parsl.timeout import timeout
import coffea.processor
import coffea.util
from coffea.processor import set_accumulator
from coffea.processor.executor import (
    _compression_wrapper, _decompress, _futures_handler, UprootMissTreeError,
    FileMeta)
from coffea.processor.accumulator import (add as accum_add, iadd as accum_iadd)
from coffea.processor import ProcessorABC
from coffea.nanoevents import NanoEventsFactory
import cloudpickle
import uuid
import lz4.frame as lz4f
from tqdm import tqdm
import pepper


STATEFILE_VERSION = 3

logger = logging.getLogger(__name__)


class StateFileError(Exception):
    pass


def _wrap_execution(function, item):
    return item, function(item)


@dataclass
class ResumableExecutor(abc.ABC, coffea.processor.executor.ExecutorBase):
    """Abstract base class for executors that save their state and thus are
    able to resume if there was an interruption

    Parameters:
    state_file_name -- Name of the file to save and read the state to/from
    remove_state_at_end -- Bool, if true, remove the state file after
                           successful completion
    save_interval -- Seconds that have to pass before the state is saved
                     after start or the last save. The sate is saved only
                     after the completion of an item
    """

    state_file_name: Optional[str] = None
    remove_state_at_end: bool = False
    save_interval: int = 300

    def __post_init__(self):
        self.state = {"items_done": [], "accumulator": None,
                      "version": STATEFILE_VERSION, "userdata": {}}

    def copy(self, **kwargs):
        # Same as ExecutorBase.copy, just handling self.state correctly
        tmp = self.__dict__.copy()
        tmp.update(kwargs)
        tmp.pop("state")
        instance = type(self)(**tmp)
        # Need deep copy here to not modify the accumulator later
        instance.state = deepcopy(self.state)
        return instance

    def load_state(self, filename=None):
        """Load a previous state from a file

        Parameters:
        filename -- Name of the file to read the state from. If not given,
                    will load from self.state_file_name
        """

        if filename is None:
            if self.state_file_name is None:
                raise ValueError("Need filename")
            filename = self.state_file_name
        state = coffea.util.load(filename)
        if state["version"] != STATEFILE_VERSION:
            raise StateFileError("State file made by incompatible version: "
                                 f"{filename}")
        self.state = state

    def reset_state(self):
        self.state = {"items_done": [], "accumulator": None,
                      "version": STATEFILE_VERSION, "userdata": {}}

    def __call__(self, items, function, accumulator):
        items_done = set(self.state["items_done"])
        items = [item for item in items if item not in items_done]

        if accumulator is not None and self.state["accumulator"] is not None:
            accumulator.add(self.state["accumulator"])
            self.state["accumulator"] = accumulator
        elif accumulator is not None:
            self.state["accumulator"] = accumulator
        elif self.state["accumulator"] is not None:
            accumulator = self.state["accumulator"]

        res = self._execute(items, partial(_wrap_execution, function),
                            accumulator)
        if (self.state_file_name is not None
                and self.remove_state_at_end
                and os.path.exists(self.state_file_name)):
            os.remove(self.state_file_name)
        return res, 0

    def _track_done_items(self, gen):
        for item, result in gen:
            self.state["items_done"].append(item)
            yield result

    def _accumulate(self, gen, accum=None):
        # In addition to fullfilling the same ask as coffea's accumulate
        # this also keeps track of done items and saves the state
        gen = self._track_done_items(gen)
        gen = (x for x in gen if x is not None)
        nextstatebackup = time.time() + self.save_interval
        try:
            if accum is None:
                # Set the accumulator to the first result
                self.state["accumulator"] = accum = next(gen)
                # If there is more to do, add up results in another accumulator
                # instance
                self.state["accumulator"] = accum = accum_add(accum, next(gen))
            while True:
                if nextstatebackup <= time.time():
                    self.save_state()
                    nextstatebackup = time.time() + self.save_interval
                accum_iadd(accum, next(gen))
        except StopIteration:
            pass
        self.save_state()
        return accum

    @abc.abstractmethod
    def _execute(self, items, function, accumulator, **kwargs):
        return

    def save_state(self):
        # Save state to a new file and only replace the previous state file
        # when writing is finished. This avoids leaving only an invalid state
        # file if the program is terminated during writing.
        i = 0
        while True:
            output = os.path.join(os.path.dirname(self.state_file_name),
                                  f"pepper_temp{i}.coffea")
            if not os.path.exists(output):
                break
            i += 1
        coffea.util.save(self.state, output)
        os.replace(output, self.state_file_name)


class IterativeExecutor(ResumableExecutor):
    """Same as coffea.processor.iterative_executor while being resumable"""
    def _execute(self, items, function, accumulator):
        if len(items) == 0:
            return accumulator
        gen = tqdm(items, disable=not self.status, unit=self.unit,
                   total=len(items), desc=self.desc)
        gen = map(function, gen)
        return self._accumulate(gen, accumulator)


@dataclass
class ParslExecutor(ResumableExecutor):
    """Same as coffea.processor.parsl_executor while being resumable"""

    tailtimeout: int = None
    allow_scalein: bool = True

    def _execute(self, items, function, accumulator):
        if len(items) == 0:
            return accumulator

        if self.compression is not None:
            function = _compression_wrapper(self.compression, function)

        dfk = parsl.dfk()
        for exec in dfk.config.executors:
            if hasattr(exec, "allow_scalein"):
                exec.allow_scalein = self.allow_scalein

        app = timeout(python_app(function))

        gen = _futures_handler(map(app, items), self.tailtimeout)
        try:
            accumulator = self._accumulate(
                tqdm(
                    gen if self.compression is None else map(_decompress, gen),
                    disable=not self.status,
                    unit=self.unit,
                    total=len(items),
                    desc=self.desc,
                ),
                accumulator,
            )
        finally:
            gen.close()

        return accumulator


class Runner(coffea.processor.Runner):
    @staticmethod
    def resolve_lfn(lfn, store, xrootddomain, skippaths):
        if lfn.startswith("cmslfn://"):
            filepaths = pepper.datasets.resolve_lfn(lfn, store, xrootddomain)
        else:
            filepaths = [lfn]
        if skippaths is not None:
            filepaths = [p for p in filepaths if p not in skippaths]

        return filepaths

    @staticmethod
    def metadata_fetcher(xrootdtimeout, align_clusters, item):
        filepaths = Runner.resolve_lfn(
            item.filename, item.metadata["store_path"],
            item.metadata["xrootddomain"], item.metadata["skippaths"])
        for filepath in filepaths:
            try:
                with uproot.open(
                        {filepath: None}, timeout=xrootdtimeout) as file:
                    try:
                        tree = file[item.treename]
                    except uproot.exceptions.KeyInFileError as e:
                        raise UprootMissTreeError(str(e)) from e

                    metadata = {}
                    if item.metadata:
                        metadata.update(item.metadata)
                    metadata.update({
                        "numentries": tree.num_entries,
                        "uuid": file.file.fUUID})
                    if align_clusters:
                        metadata["clusters"] = tree.common_entry_offsets()
                    out = set_accumulator(
                        [FileMeta(
                            item.dataset, item.filename, item.treename,
                            metadata)]
                    )
            except OSError as e:
                logger.warning(
                    "Got error while opening, continuing with alternative "
                    f"file location: {e}")
                continue
            break
        else:
            raise OSError(
                "None of the file paths found for the following file could be "
                f"opened: {item.filename}")

        return out

    @staticmethod
    def _work_function(
        format,
        xrootdtimeout,
        mmap,
        schema,
        cache_function,
        use_dataframes,
        savemetrics,
        item,
        processor_instance,
    ):
        if not isinstance(processor_instance, ProcessorABC):
            processor_instance = cloudpickle.loads(
                lz4f.decompress(processor_instance))
        # The ResumableExecutor might have loaded and old state, thus giving
        # old metadata possibly before the user changed the config.
        # Instead obtain the metadata from the processor_instance
        metadata = processor_instance.pepperitemmetadata

        filepaths = Runner.resolve_lfn(
            item.filename, metadata["store_path"],
            metadata["xrootddomain"], metadata["skippaths"])

        for filepath in filepaths:
            try:
                filecontext = uproot.open(
                    {filepath: None},
                    timeout=xrootdtimeout,
                    file_handler=uproot.MemmapSource
                    if mmap
                    else uproot.MultithreadedFileSource,
                )
            except OSError as e:
                logger.warning(
                    "Got error while opening, continuing with alternative "
                    f"file location: {e}")
                continue
            break
        else:
            raise OSError(
                "None of the file paths found for the following file could be "
                f"opened: {item.filename}")

        metadata = {
            "dataset": item.dataset,
            "filename": filepath,
            "treename": item.treename,
            "entrystart": item.entrystart,
            "entrystop": item.entrystop,
            "fileuuid": str(uuid.UUID(bytes=item.fileuuid))
            if len(item.fileuuid) > 0
            else "",
        }
        if item.usermeta is not None:
            metadata.update(item.usermeta)

        materialized = []
        with filecontext as file:
            factory = NanoEventsFactory.from_root(
                file=file,
                treepath=item.treename,
                entry_start=item.entrystart,
                entry_stop=item.entrystop,
                persistent_cache=cache_function(),
                schemaclass=schema,
                metadata=metadata,
                access_log=materialized,
            )
            events = factory.events()

            out = processor_instance.process(events)

        return {"out": out}
