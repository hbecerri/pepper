import h5py
import hdf5plugin
from collections.abc import Mapping
from collections import MutableMapping
import numpy as np
import awkward as ak
import json


class CompressionWrapper:
    def __init__(self, group, compression, compression_opts=None):
        self.group = group
        self.compression = compression
        self.compression_opts = compression_opts

    def __setitem__(self, name, obj):
        if np.asarray(obj).nbytes < 16:
            # If the object size is very small, compression won't have much
            # effect. In fact the Blosc implementation raises warnings for
            # sizes < 16 bytes. Skip compression in this case
            self.group[name] = obj
            return
        ds = self.group.create_dataset(None, data=obj,
                                       compression=self.compression,
                                       compression_opts=self.compression_opts)
        self.group[name] = ds


class HDF5File(MutableMapping):
    def __init__(self, file, mode=None, compression=hdf5plugin.Blosc()):
        """Create or open an HDF5 file storing awkward arrays

        Arguments:
        file -- Filename as string or Python file object or h5py file object
        mode -- Determines whether to read ('r') or to write ('w') in case
                `file` is a string
        compression -- If the file is opened for writing, determines the
                       compression used for writing. If None, compression
                       is disabled. Either the value for `compression` in
                       `h5py.Group.create_dataset` or a mapping with keys
                       `compression` and `compression_opts`. See
                       `h5py.Group.create_dataset` for details.
        """
        if isinstance(file, str):
            if mode is None:
                raise ValueError("If file is a str, mode can not be None")
            file = h5py.File(file, mode)
            self._should_close = True
        else:
            self._should_close = False
        self._file = file
        self.compression = compression

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            try:
                value = ak.Array({k: [v] for k, v in value.items()})
            except RuntimeError:
                raise ValueError("Only dicts with string keys and simple "
                                 "values are supported")
            convertto = "dict"
        elif isinstance(value, list):
            value = ak.Array(value)
            convertto = "list"
        elif isinstance(value, str):
            value = ak.Array([value])
            convertto = "str"
        elif isinstance(value, tuple):
            value = ak.Array(value)
            convertto = "tuple"
        elif isinstance(value, np.ndarray):
            value = ak.Array(value)
            convertto = "numpy"
        elif not isinstance(value, ak.Array):
            raise ValueError(f"Invalid type for writing to HDF5: {value}")
        else:
            convertto = "None"
        group = self._file.create_group(key)
        if self.compression is not None:
            if isinstance(self.compression, Mapping):
                container = CompressionWrapper(group, **self.compression)
            else:
                container = CompressionWrapper(group, self.compression)
        else:
            container = group
        form, length, container = ak.to_buffers(value, container=container)
        group.attrs["form"] = form.tojson()
        group.attrs["length"] = json.dumps(length)
        group.attrs["parameters"] = json.dumps(ak.parameters(value))
        group.attrs["convertto"] = convertto
        self._file.attrs["version"] = 1

    def __getitem__(self, key):
        group = self._file[key]
        form = ak.forms.Form.fromjson(group.attrs["form"])
        length = json.loads(group.attrs["length"])
        parameters = json.loads(group.attrs["parameters"])
        convertto = group.attrs["convertto"]
        data = {k: np.asarray(v) for k, v in group.items()}
        value = ak.from_buffers(form, length, data)
        for parameter, param_value in parameters.items():
            value = ak.with_parameter(value, parameter, param_value)
        if convertto == "numpy":
            value = np.asarray(value)
        elif convertto == "str":
            value = value[0]
        elif convertto == "tuple":
            value = tuple(value.tolist())
        elif convertto == "list":
            value = value.tolist()
        elif convertto == "dict":
            value = {field: value[field][0] for field in ak.fields(value)}
        return value

    def __delitem__(self, key):
        del self._file[key]

    def __len__(self, key):
        return len(self._file)

    def __iter__(self):
        for key in self._file.keys():
            yield self[key]

    def __repr__(self):
        return f"<AkHdf5 ({self._file.filename})>"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close:
            self._file.close()
        return False

    def keys(self):
        return self._file.keys()
