from os import PathLike
from collections.abc import Mapping
from functools import singledispatch

import h5py
from scipy import sparse

import anndata as ad
from anndata._core.merge import (
    union_keys,
    intersect_keys,
    merge_nested,
)
from anndata._core.index import _normalize_indices
from anndata._io.h5ad import read_attribute, read_dataframe, read_dataset
from anndata._core.sparse_dataset import  SparseDataset


def read_elem_subset(packed):
    group, keys = packed
    return {key: read_attribute(group[key]) for key in keys}

# def read_partial(
#     pth: PathLike,
#     *,
#     obs_idx=None,
#     var_idx=None,
#     X = False,
#     obsm=None,
#     varm=None,
#     obsp=None,
#     varp=None,
#     layers=None,
#     uns=None,
# ) -> ad.AnnData:
#     idx = (obs_idx, var_idx)
#     elems = {
#         "obsm": obsm,
#         "varm": varm,
#         "obsp": obsp,
#         "varp": varp,
#         "layers": layers,
#         "uns": uns,
#     }

#     schema = {k: v for k, v in elems.items() if v is not None}
#     with h5py.File(pth, "r") as f:
#         result = merge_nested((f, schema), intersect_keys, read_elem_subset)
#     return result

###### V2

class H5Translator():
    pass


class NDArray(H5Translator):
    encoding_version = "0.2.0"
    encoding_type = "array"

    def read(elem):
        return elem[()]

    def read_partial(elem, *, items=None, indices=(slice(None), slice(None))):
        return elem[indices]


class CSRSparse(H5Translator):
    encoding_version = "0.1.0"
    encoding_type = "csr_matrix"

    def read(elem):
        return SparseDataset(elem).to_memory()

    # This can be very slow
    def read_partial(elem, *, items=None, indices=(slice(None), slice(None))):
        return SparseDataset(elem)[indices]

class CSCSparse(H5Translator):
    encoding_version = "0.1.0"
    encoding_type = "csc_matrix"

    def read(elem):
        return SparseDataset(elem).to_memory()

    def read_partial(elem, *, items=None, indices=(slice(None), slice(None))):
        return SparseDataset(elem)[indices]

class DataFrame(H5Translator):
    encoding_version = "0.1.0"
    encoding_type = "dataframe"

    def read(elem):
        columns = list(elem.attrs["column-order"])
        idx_key = elem.attrs["_index"]
        df = pd.DataFrame(
            {k: find_translator(group[k]).read(group[k]) for k in columns},
            index=find_translator(group[idx_key]).read(group[idx_key]),
            columns=list(columns),
        )
        if idx_key != "_index":
            df.index.name = idx_key
        return df

    # TODO
    def read_partial(elem, *, items=None, indices=(slice(None), slice(None))):
        if items is None:
            items = slice(None)
        else:
            items = list(items)
        return read_dataframe(elem)[items].iloc[indices[0]]


class Basic(H5Translator):
    encoding_type = ""
    encoding_version = ""

    def read(elem):
        if isinstance(elem, Mapping):
            return read(elem)
        elif isinstance(elem, h5py.Dataset):
            return read_dataset(elem)

    def read_partial(elem, *, items=None, indices=(slice(None), slice(None))):
        if isinstance(elem, Mapping):
            return _read_partial(elem, items=items, indices=indices)
        elif indices != (slice(None), slice(None)):
            return elem[indices]
        else:
            return elem[()]

def read(group):
    return {k: find_translator(v).read(v) for k, v in group.items()}

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
        result[k] = find_translator(group[k]).read_partial(
            group[k], items =next_items, indices=indices
        )
    return result

# def read_h5ad_partial(group, *, items, indices)

def read_partial(
    pth: PathLike,
    *,
    obs_idx=slice(None),
    var_idx=slice(None),
    X = True,
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
        result["obs"] = find_translator(f["obs"]).read_partial(f["obs"], items=obs, indices=(obs_idx, slice(None)))
        result["var"] = find_translator(f["var"]).read_partial(f["var"], items=var, indices=(var_idx, slice(None)))
        if X:
            result["X"] = find_translator(f["X"]).read_partial(f["X"], indices=(obs_idx, var_idx))
        else:
            result["X"] = sparse.csr_matrix((len(result["obs"]), len(result["var"])))
        result["obsm"] = _read_partial(f.get("obsm", None), items=obsm, indices=(obs_idx, slice(None)))
        result["varm"] = _read_partial(f.get("varm", None), items=varm, indices=(var_idx, slice(None)))
        result["obsp"] = _read_partial(f.get("obsp", None), items=obsp, indices=(obs_idx, obs_idx))
        result["varp"] = _read_partial(f.get("varp", None), items=varp, indices=(var_idx, var_idx))
        result["layers"] = _read_partial(f.get("layers", None), items=layers, indices=(obs_idx, var_idx))
        result["uns"] = _read_partial(f.get("uns", None), items=uns,)

    return ad.AnnData(**result)

def read_indices(group):
    obs_group = group["obs"]
    obs_idx_elem = obs_group[obs_group.attrs["_index"]]
    obs_idx = find_translator(obs_idx_elem).read(obs_idx_elem)
    var_group = group["var"]
    var_idx_elem = var_group[var_group.attrs["_index"]]
    var_idx = find_translator(var_idx_elem).read(var_idx_elem)
    return obs_idx, var_idx

# read_partial()
#     _read_partial()

#     schema = {k: v for k, v in elems.items() if v is not None}
#     with h5py.File(pth, "r") as f:
#         result = merge_nested((f, schema), intersect_keys, read_elem_subset)
#     return result

def find_translator(item):
    enc_version = item.attrs.get("encoding-version", "")
    enc_type = item.attrs.get("encoding-type", "")

    translators = H5Translator.__subclasses__()
    for t in translators:
        if (enc_type == t.encoding_type and enc_version == t.encoding_version):
            return t
    raise NotImplementedError()

