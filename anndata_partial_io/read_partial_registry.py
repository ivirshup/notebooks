from os import PathLike
from collections import Mapping
from functools import singledispatch
from typing import NamedTuple, Union

import h5py
import numpy as np
import pandas as pd

import anndata as ad
from anndata._core.index import _normalize_indices
from anndata._core.merge import intersect_keys
from anndata._core.sparse_dataset import SparseDataset
from anndata._io.h5ad import read_dataset


# TODO: This probably should be replaced by a hashable Mapping due to conversion b/w "_" and "-"
class IOSchema(NamedTuple):
    encoding_type: str
    encoding_version: str


class IORegistry(object):
    def __init__(self):
        self.read = {}
        self.read_partial = {}
        self.write = {}
        self.write_schema = {}

    # TODO: Make this add schema fields to `attrs`
    # This might actually be more complicated, since I want to dispatch writing on types
    def register_write(self, type, schema):
        def _register(func):
            self.write_schema[type] = proc_schema(schema)
            self.write[type] = func
            return func

        return _register

    def register_read(self, schema):
        def _register(func):
            self.read[proc_schema(schema)] = func
            return func

        return _register

    def register_read_partial(self, schema):
        def _register(func):
            self.read_partial[proc_schema(schema)] = func
            return func

        return _register


_REGISTRY = IORegistry()


@singledispatch
def proc_schema(schema):
    raise NotImplementedError(f"proc_schema not defined for type: {type(schema)}.")


@proc_schema.register(IOSchema)
def proc_schema_schema(schema) -> IOSchema:
    return schema


@proc_schema.register(Mapping)
def proc_schema_mapping(schema) -> IOSchema:
    return IOSchema(**{k.replace("-", "_"): v for k, v in schema.items()})


def get_schema(elem: Union[h5py.Dataset, h5py.Group]):
    return proc_schema(
        {k: elem.attrs.get(k, "") for k in ["encoding-type", "encoding-version"]}
    )


def read_elem(elem):
    return _REGISTRY.read[get_schema(elem)](elem)


def read_elem_partial(elem, *, items=None, indices=(slice(None), slice(None))):
    return _REGISTRY.read_partial[get_schema(elem)](elem, items=items, indices=indices)


@_REGISTRY.register_read({"encoding-type": "array", "encoding-version": "0.2.0"})
def read_array(elem):
    return elem[()]


@_REGISTRY.register_read_partial(
    {"encoding-type": "array", "encoding-version": "0.2.0"}
)
def read_array_partial(elem, *, items=None, indices=(slice(None, None))):
    return elem[indices]


@_REGISTRY.register_read({"encoding-type": "csc_matrix", "encoding-version": "0.1.0"})
@_REGISTRY.register_read({"encoding-type": "csr_matrix", "encoding-version": "0.1.0"})
def read_sparse(elem):
    return SparseDataset(elem).to_memory()


@_REGISTRY.register_read_partial(
    {"encoding-type": "csc_matrix", "encoding-version": "0.1.0"}
)
@_REGISTRY.register_read_partial(
    {"encoding-type": "csr_matrix", "encoding-version": "0.1.0"}
)
def read_sparse_partial(elem, *, items=None, indices=(slice(None), slice(None))):
    return SparseDataset(elem)[indices]


@_REGISTRY.register_read({"encoding-type": "dataframe", "encoding-version": "0.1.0"})
def read_dataframe_0_1_0(elem):
    columns = list(elem.attrs["column-order"])
    idx_key = elem.attrs["_index"]
    df = pd.DataFrame(
        {k: read_elem(elem[k]) for k in columns},
        index=read_elem(elem[idx_key]),
        columns=list(columns),
    )
    if idx_key != "_index":
        df.index.name = idx_key
    return df


@_REGISTRY.register_read_partial(
    {"encoding-type": "dataframe", "encoding-version": "0.1.0"}
)
def read_partial_dataframe_0_1_0(
    elem, *, items=None, indices=(slice(None), slice(None))
):
    if items is None:
        items = slice(None)
    else:
        items = list(items)
    return read_elem(elem)[items].iloc[indices[0]]


@_REGISTRY.register_read({"encoding-type": "", "encoding-version": ""})
def read_basic(elem):
    from anndata._io import h5ad

    if isinstance(elem, Mapping):
        return {k: read_elem(v) for k, v in elem.items()}
    elif isinstance(elem, h5py.Dataset):
        return h5ad.read_dataset(elem)  # TODO: Handle legacy


@_REGISTRY.register_read_partial({"encoding-type": "", "encoding-version": ""})
def read_basic_partial(elem, *, items=None, indices=(slice(None), slice(None))):
    if isinstance(elem, Mapping):
        return _read_partial(elem, items=items, indices=indices)
    elif indices != (slice(None), slice(None)):
        return elem[indices]
    else:
        return elem[()]


def _read_partial(group, *, items=None, indices=(slice(None), slice(None))):
    if group is None:
        return None
    if items is None:
        keys = intersect_keys((group,))
    else:
        keys = intersect_keys((group, items))
    result = {}
    for k in keys:
        if isinstance(items, Mapping):
            next_items = items.get(k, None)
        else:
            next_items = None
        result[k] = read_elem_partial(group[k], items=next_items, indices=indices)
    return result


def read_indices(group):
    obs_group = group["obs"]
    obs_idx_elem = obs_group[obs_group.attrs["_index"]]
    obs_idx = read_elem(obs_idx_elem)
    var_group = group["var"]
    var_idx_elem = var_group[var_group.attrs["_index"]]
    var_idx = read_elem(var_idx_elem)
    return obs_idx, var_idx


def read_partial(
    pth: PathLike,
    *,
    obs_idx=slice(None),
    var_idx=slice(None),
    X=True,
    obs=None,
    var=None,
    obsm=None,
    varm=None,
    obsp=None,
    varp=None,
    layers=None,
    uns=None,
) -> ad.AnnData:
    result = {}
    with h5py.File(pth, "r") as f:
        obs_idx, var_idx = _normalize_indices((obs_idx, var_idx), *read_indices(f))
        result["obs"] = read_elem_partial(
            f["obs"], items=obs, indices=(obs_idx, slice(None))
        )
        result["var"] = read_elem_partial(
            f["var"], items=var, indices=(var_idx, slice(None))
        )
        if X:
            result["X"] = read_elem_partial(f["X"], indices=(obs_idx, var_idx))
        else:
            result["X"] = sparse.csr_matrix((len(result["obs"]), len(result["var"])))
        if "obsm" in f:
            result["obsm"] = _read_partial(
                f["obsm"], items=obsm, indices=(obs_idx, slice(None))
            )
        if "varm" in f:
            result["varm"] = _read_partial(
                f["varm"], items=varm, indices=(var_idx, slice(None))
            )
        if "obsp" in f:
            result["obsp"] = _read_partial(
                f["obsp"], items=obsp, indices=(obs_idx, obs_idx)
            )
        if "varp" in f:
            result["varp"] = _read_partial(
                f["varp"], items=varp, indices=(var_idx, var_idx)
            )
        if "layers" in f:
            result["layers"] = _read_partial(
                f["layers"], items=layers, indices=(obs_idx, var_idx)
            )
        if "uns" in f:
            result["uns"] = _read_partial(f["uns"], items=uns)

    return ad.AnnData(**result)

def read(pth):
    with h5py.File(pth, "r") as f:
        results = read_elem(f)
    return ad.AnnData(**results)
