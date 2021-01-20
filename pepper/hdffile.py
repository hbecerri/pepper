from collections import MutableMapping
import numpy as np
import awkward as ak
import json


class HDF5File(MutableMapping):
    def __init__(self, file, lazy=False):
        self._file = file
        self.lazy = lazy

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
        form, length, container = ak.to_buffers(value, container=group)
        group.attrs["form"] = form.tojson()
        group.attrs["length"] = json.dumps(length)
        group.attrs["parameters"] = json.dumps(ak.parameters(value))
        group.attrs["convertto"] = convertto

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

    def keys(self):
        return self._file.keys()
