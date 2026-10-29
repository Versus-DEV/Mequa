import numpy as np

from mequa import numpy_backend as nb


def test_analyze_outputs_are_consistent():
    pts = nb.synthetic_cells(600)
    r = nb.analyze(pts, k=3, knn=6, load=0.05)
    assert r["labels"].min() == 1 and r["labels"].max() == 3
    assert len(r["node_strain"]) == 600
    assert np.all((r["failure_prob"] >= 0) & (r["failure_prob"] <= 1))
    assert r["sphi"] > 0
    assert r["n_edges"] > 0


def test_more_pull_means_more_strain():
    pts = nb.synthetic_cells(600)
    low = nb.analyze(pts, load=0.02)
    high = nb.analyze(pts, load=0.10)
    assert high["node_strain"].mean() > low["node_strain"].mean()
    assert high["sphi"] > low["sphi"]


def test_no_pull_means_no_strain():
    pts = nb.synthetic_cells(600)
    r = nb.analyze(pts, load=0.0)
    assert r["node_strain"].max() < 1e-9
    assert r["sphi"] == 0.0


def test_edges_are_sorted_unique_pairs():
    pts = nb.synthetic_cells(400)
    ei, ej = nb.build_knn_edges(pts, 5)
    assert np.all(ei < ej)
    keys = ei.astype(np.int64) * len(pts) + ej
    assert len(np.unique(keys)) == len(keys)
    assert np.all(np.diff(keys) > 0)
