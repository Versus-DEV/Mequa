import numpy as np

from mequa import backend as backend_mod
from mequa import ingest
from mequa import numpy_backend as nb
from mequa.session import Session


def test_3d_analysis_is_consistent():
    pts = nb.synthetic_cells(500, seed=1, dim=3)
    r = nb.analyze(pts, k=3, knn=8, load=0.05)
    assert r["dim"] == 3 and len(r["node_strain"]) == 500
    assert r["labels"].min() == 1 and r["labels"].max() == 3
    assert np.all(np.isfinite(r["node_strain"])) and r["sphi"] > 0
    assert nb.analyze(pts, load=0.0)["node_strain"].max() < 1e-9
    assert nb.analyze(pts, load=0.1)["node_strain"].mean() > nb.analyze(pts, load=0.03)["node_strain"].mean()


def test_uniform_stiffness_equals_no_stiffness():
    pts = nb.synthetic_cells(400, seed=2)
    a = nb.analyze(pts, load=0.05)
    b = nb.analyze(pts, load=0.05, stiffness=np.ones(400))
    assert np.allclose(a["node_strain"], b["node_strain"], atol=1e-12)


def test_soft_cells_change_strain_when_the_load_path_is_connected():
    pts = nb.synthetic_cells(600, seed=5, clusters=1)
    base = nb.analyze(pts, knn=8, load=0.06)
    assert base["load_path"] and base["n_components"] == 1
    stiff = np.ones(600)
    soft = np.argsort(pts[:, 1])[:150]  # a soft band of cells
    stiff[soft] = 0.5
    graded = nb.analyze(pts, knn=8, load=0.06, stiffness=stiff)
    assert not np.allclose(base["node_strain"], graded["node_strain"], rtol=1e-3, atol=1e-5)
    assert graded["node_strain"][soft].mean() > base["node_strain"][soft].mean()


def test_disconnected_graph_is_detected():
    pts = nb.synthetic_cells(600, seed=5)  # three clusters, knn 6: graph splits in two
    r = nb.analyze(pts, knn=6, load=0.06)
    assert r["n_components"] > 1 and r["load_path"] is False


def test_relaxation_is_near_converged_on_a_connected_network():
    pts = nb.synthetic_cells(600, seed=5, clusters=1)
    ei, ej = nb.build_knn_edges(pts, 8)
    _, short, _ = nb.relax_strain(pts, ei, ej, load=0.06, iters=1500)
    _, long, _ = nb.relax_strain(pts, ei, ej, load=0.06, iters=12000)
    assert abs(short.mean() - long.mean()) / long.mean() < 0.05


def test_stiffness_length_and_range_are_checked():
    pts = nb.synthetic_cells(100)
    try:
        nb.analyze(pts, stiffness=np.ones(99))
        assert False
    except ValueError:
        pass
    s = nb.stiffness_from_areas([1, 100, 100, 100, 10000])
    assert s.min() >= 0.5 and s.max() <= 2.0


def test_router_sends_3d_and_stiffness_jobs_to_numpy_and_says_so():
    class Fake:
        name = "julia"

        def analyze(self, *a, **k):
            raise AssertionError("2D uniform jobs only")

    pts3 = nb.synthetic_cells(200, dim=3)
    res = backend_mod.analyze_routed(Fake(), pts3, load=0.05)
    assert res["backend_used"] == "numpy" and res["routed_from"] == "julia"


def test_session_rejects_wrong_stiffness_and_returns_floats():
    pts = nb.synthetic_cells(200)
    try:
        Session(pts, stiffness=np.ones(5))
        assert False
    except ValueError:
        pass
    out = Session(pts, stiffness=np.ones(200)).run(0.05)
    assert out["cells"] == 200 and isinstance(out["sphi"], float) and out["backend_used"] == "numpy"


def test_zstack_linking_recovers_3d_cells():
    imgs, truth = ingest.make_demo_stack(n_cells=40, n_slices=10, size=200, seed=7)
    pts, areas = ingest.segment_stack(imgs, "classical", z_spacing=1.0, max_dist=6.0)
    assert pts.shape[1] == 3 and len(pts) == len(areas)
    assert 0.85 * len(truth) <= len(pts) <= 1.15 * len(truth)
    t = truth.copy()
    t[:, 1] = imgs[0].shape[0] - t[:, 1]
    from scipy.spatial import cKDTree

    d, _ = cKDTree(pts).query(t)
    assert np.median(d) < 1.0  # median centre error under one pixel/slice unit


def test_csv_roundtrip_with_z_and_area(tmp_path):
    pts = nb.synthetic_cells(50, dim=3)
    areas = np.arange(50, dtype=float) + 10
    path = tmp_path / "cells.csv"
    ingest.save_points_csv(pts, path, areas=areas)
    p2, a2 = ingest.load_cells_csv(path)
    assert p2.shape == (50, 3) and np.allclose(p2, pts, atol=1e-3) and np.allclose(a2, areas)
    pts2 = nb.synthetic_cells(30)
    ingest.save_points_csv(pts2, path)
    p3, a3 = ingest.load_cells_csv(path)
    assert p3.shape == (30, 2) and a3 is None
