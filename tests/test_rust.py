import numpy as np
import pytest

mequa_rs = pytest.importorskip("mequa_rs")

from mequa import numpy_backend as nb
from mequa import rust_backend as rb


def test_rust_matches_numpy():
    pts = nb.synthetic_cells(1500)
    a = nb.analyze(pts, 3, 6, 0.05)
    b = rb.analyze(pts, 3, 6, 0.05)
    assert a["n_edges"] == b["n_edges"]
    assert abs(a["sphi"] - b["sphi"]) / a["sphi"] < 1e-9
    assert np.max(np.abs(a["node_strain"] - b["node_strain"])) < 1e-9


def test_rust_edges_equal_scipy_edges():
    pts = nb.synthetic_cells(800)
    ei, ej = nb.build_knn_edges(pts, 6)
    ri, rj = mequa_rs.knn_edges(pts, 6)
    assert np.array_equal(ei, ri) and np.array_equal(ej, rj)


def test_zero_copy_address():
    pts = np.random.default_rng(0).random((500, 2))
    assert mequa_rs.buffer_address(pts) == pts.ctypes.data


def test_bad_inputs_are_rejected_not_copied():
    pts = nb.synthetic_cells(100)
    with pytest.raises(TypeError):
        mequa_rs.buffer_address(pts.astype(np.float32))
    with pytest.raises(ValueError):
        mequa_rs.knn_edges(np.asfortranarray(pts), 4)
    with pytest.raises(ValueError):
        mequa_rs.knn_edges(pts[::2], 4)
    with pytest.raises(ValueError):
        mequa_rs.knn_edges(np.zeros((10, 3)), 2)
    ei, ej = mequa_rs.knn_edges(pts, 4)
    with pytest.raises(ValueError):
        mequa_rs.relax_strain(pts, ei + 10**6, ej)
