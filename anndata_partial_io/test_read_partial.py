from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

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
