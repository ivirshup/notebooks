import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

import anndata as ad
from anndata.tests.helpers import gen_adata, assert_equal
import scanpy as sc
from read_partial import read_partial

# TODO: Can't currently say "read everything"
# def test_read(tmp_dir):
#     """Test that I can read the whole thing in right"""
#     pth = tmp_dir / "test.h5ad"
#     orig = gen_adata((10, 20))
#     orig.write(pth, compression="lzf")

#     truth = sc.read_h5ad(pth)
#     test = read_partial(pth)


def test_read_partial(tmpdir):
    pth = tmpdir / "test.h5ad"
    orig = gen_adata((10, 20))
    orig.write(pth, compression="lzf")

    truth = sc.read_h5ad(pth)[::2].copy()
    test = read_partial(pth, obs_idx=slice(None, None, 2))
    test._remove_unused_categories(test.obs, test.obs, test.uns)  # TODO: Unused categories currently not removed

    assert_equal(truth, test)

def test_read_partial_registry(tmpdir):
    import read_partial_registry

    pth = tmpdir / "test.h5ad"
    orig = gen_adata((10, 20))
    orig.write(pth, compression="lzf")

    truth = sc.read_h5ad(pth)[::2].copy()
    test = read_partial_registry.read_partial(pth, obs_idx=slice(None, None, 2))
    test._remove_unused_categories(test.obs, test.obs, test.uns)  # TODO: Unused categories currently not removed

    assert_equal(truth, test)


def test_read(tmpdir):
    from read_partial_registry import read
    pth = tmpdir / "test.h5ad"

    orig = gen_adata((10, 20))
    orig.write(pth, compression="lzf")

    truth = ad.read_h5ad(pth)
    test = read(pth)

    assert_equal(truth, test)
    assert_equal(orig, test)


def test_read_write(tmpdir):
    from read_partial_registry import read, write
    pth = tmpdir / "test.h5ad"

    orig = gen_adata((10, 20))
    orig.uns["scalars"] = {"str": "abced", "int": 1, "float": 1., "bool": True, "int32": np.int32(1)}

    write(orig, pth)
    from_disk = read(pth)

    assert_equal(orig, from_disk)


def test_write_view(tmpdir):
    from read_partial_registry import read, write
    pth = tmpdir / "test.h5ad"

    orig = gen_adata((10, 20))
    orig.uns["scalars"] = {"str": "abced", "int": 1, "float": 1., "bool": True, "int32": np.int32(1)}

    view = orig[:5, :]

    write(view, pth)
    test = read(pth)

    actual = view.copy()
    assert_equal(actual, test)


def test_write_partial_read(tmpdir):
    from read_partial_registry import read_partial, write
    pth = tmpdir / "test.h5ad"

    orig = gen_adata((10, 20))
    orig.uns["scalars"] = {"str": "abced", "int": 1, "float": 1., "bool": True, "int32": np.int32(1)}
    write(orig, pth)

    subset = orig[:5, :].copy()

    test = read_partial(pth, obs_idx=slice(None, 5))

    assert_equal(subset, test)