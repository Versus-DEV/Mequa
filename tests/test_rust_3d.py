import numpy as np
import pytest

mequa_rs = pytest.importorskip("mequa_rs")

from mequa import numpy_backend as nb


def _module_supports_3d():
    try:
        mequa_rs.relax_strain(np.zeros((4, 3)), np.array([0, 1, 2], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64), 0.05, 0.1, 1, 0.05)
        return True
    except ValueError:
        return False


pytestmark = pytest.mark.skipif(not _module_supports_3d(), reason="installed mequa_rs is the old 2D build: rebuild it (rust\\build.ps1 on Windows)")


@pytest.mark.parametrize("dim,knn", [(2, 6), (3, 8)])
def test_rust_relaxation_matches_numpy_in_2d_and_3d(dim, knn):
    pts = np.ascontiguousarray(nb.synthetic_cells(800, seed=3, clusters=1, dim=dim))
    ei, ej = nb.build_knn_edges(pts, knn)
    ei, ej = np.ascontiguousarray(ei, dtype=np.int64), np.ascontiguousarray(ej, dtype=np.int64)
    pos_n, ns_n, es_n = nb.relax_strain(pts, ei, ej, load=0.05, iters=300)
    pos_r, ns_r, es_r = mequa_rs.relax_strain(pts, ei, ej, 0.05, 0.1, 300, 0.05)
    assert pos_r.shape == pts.shape
    assert np.abs(pos_r - pos_n).max() < 1e-9
    assert np.abs(ns_r - ns_n).max() < 1e-9


def test_rust_knn_refuses_3d_clearly():
    pts = np.ascontiguousarray(nb.synthetic_cells(50, seed=1, dim=3))
    with pytest.raises(ValueError, match="2D only"):
        mequa_rs.knn_edges(pts, 4)
