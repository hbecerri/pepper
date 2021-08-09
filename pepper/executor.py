import os
import abc
import time
from collections import namedtuple
import parsl
from parsl.app.app import python_app
from coffea.processor.parsl.timeout import timeout
import coffea.processor
import coffea.util
from coffea.processor.executor import (
    _compression_wrapper, _decompress, _futures_handler)
from coffea.processor.accumulator import accumulate
from tqdm import tqdm


STATEFILE_VERSION = 1


class StateFileError(Exception):
    pass


class WorkItem(namedtuple("WorkItemBase", [
    "dataset", "filename", "treename", "entrystart", "entrystop", "fileuuid",
    "usermeta"
])):
    def __new__(cls, dataset, filename, treename, entrystart, entrystop,
                fileuuid, usermeta=None):
        return cls.__bases__[0].__new__(
            cls, dataset, filename, treename, entrystart, entrystop, fileuuid,
            usermeta)

    def __len__(self):
        return self.entrystop - self.entrystart


class ResumableExecutor(abc.ABC):
    def __init__(self, state_file_name=None, remove_state_at_end=False,
                 save_interval=300):
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
        self.state_file_name = state_file_name
        self.remove_state_at_end = remove_state_at_end
        self.save_interval = save_interval
        self.state = {"items_done": [], "accumulator": None,
                      "version": STATEFILE_VERSION, "userdata": {}}

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
            raise StateFileError("State file made by incompatible version")
        self.state = state

    def reset_state(self):
        self.state = {"items_done": [], "accumulator": None,
                      "version": STATEFILE_VERSION, "userdata": {}}

    def __call__(self, items, function, accumulator, **kwargs):
        # Workaround for coffea WorkItems having no comparison
        if isinstance(next(iter(items)), coffea.processor.executor.WorkItem):
            items = [WorkItem(
                item.dataset, item.filename, item.treename, item.entrystart,
                item.entrystop, item.fileuuid, item.usermeta)
                for item in items]

        items_done = self.state["items_done"]
        items = [item for item in items if item not in items_done]

        if accumulator is not None and self.state["accumulator"] is not None:
            accumulator.add(self.state["accumulator"])
            self.state["accumulator"] = accumulator
        elif accumulator is not None:
            self.state["accumulator"] = accumulator
        elif self.state["accumulator"] is not None:
            accumulator = self.state["accumulator"]

        res = self._execute(items, function, accumulator, **kwargs)
        if (self.state_file_name is not None
                and self.remove_state_at_end
                and os.path.exists(self.state_file_name)):
            os.remove(self.state_file_name)
        return res

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

    def save_state_generator(self, gen, items):
        nextstatebackup = time.time() + self.save_interval
        for item, result in zip(items, gen):
            # Save state before appending item to done because result is not in
            # accumulator yet
            if nextstatebackup <= time.time():
                self.save_state()
                nextstatebackup = time.time() + self.save_interval
            self.state["items_done"].append(item)
            if self.state["accumulator"] is None:
                self.state["accumulator"] = result
            yield result
        self.save_state()


class IterativeExecutor(ResumableExecutor):
    """Same as coffea.processor.iterative_executor while being resumable"""
    def _execute(self, items, function, accumulator, **kwargs):
        if len(items) == 0:
            return accumulator
        status = kwargs.pop("status", True)
        unit = kwargs.pop("unit", "items")
        desc = kwargs.pop("desc", "Processing")
        gen = tqdm(items, disable=not status, unit=unit, total=len(items),
                   desc=desc)
        gen = map(function, gen)
        gen = self.save_state_generator(gen, items)
        return accumulate(gen, accumulator)


class ParslExecutor(ResumableExecutor):
    """Same as coffea.processor.parsl_executor while being resumable"""
    def _execute(self, items, function, accumulator, **kwargs):
        if len(items) == 0:
            return accumulator

        status = kwargs.pop("status", True)
        unit = kwargs.pop("unit", "items")
        desc = kwargs.pop("desc", "Processing")
        clevel = kwargs.pop("compression", 1)
        tailtimeout = kwargs.pop("tailtimeout", None)
        if clevel is not None:
            function = _compression_wrapper(clevel, function)

        parsl.dfk()

        app = timeout(python_app(function))

        gen = _futures_handler(map(app, items), tailtimeout)
        if clevel is not None:
            gen = map(_decompress, gen)
        gen = self.save_state_generator(gen, items)
        try:
            accumulator = accumulate(
                tqdm(
                    gen,
                    disable=not status,
                    unit=unit,
                    total=len(items),
                    desc=desc,
                ),
                accumulator,
            )
        finally:
            gen.close()

        return accumulator
