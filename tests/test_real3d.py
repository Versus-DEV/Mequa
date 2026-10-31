import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from mequa import real3d


def _volume(seed=0, n=24):
    rng = np.random.default_rng(seed)
    shape = (40, 96, 96)
    centers = []
    while len(centers) < n:
        c = np.array([rng.uniform(8, 32), rng.uniform(10, 86), rng.uniform(10, 86)])
        if all(np.linalg.norm(c - o) > 16 for o in centers):
            centers.append(c)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vol = np.zeros(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.uint16)
    for i, c in enumerate(centers, 1):
        d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        vol += 1000 * np.exp(-d2 / (2 * 3.0 ** 2))
        labels[d2 <= 4.5 ** 2] = i
    vol += rng.normal(0, 20, shape)
    return vol, labels, np.array(centers)


def test_detection_and_analysis_on_a_synthetic_3d_stack(tmp_path):
    vol, labels, _ = _volume()
    tifffile.imwrite(tmp_path / "v.tif", vol)
    tifffile.imwrite(tmp_path / "gt.tif", labels)
    res = real3d.run(str(tmp_path / "v.tif"), str(tmp_path / "gt.tif"), radius=4.0, z_scale=1.0, knn=8, sigma=2.0, min_distance=4)
    det = res["detection_vs_expert_labels"]
    assert det["ground_truth"] == 24 and det["f1"] >= 0.9
    assert res["analysis_on_detected_cells"]["dim"] == 3


def test_centroids_from_labels_are_in_xyz_order():
    lab = np.zeros((5, 6, 7), dtype=np.uint16)
    lab[1, 2, 3] = 1
    pts = real3d.centroids_from_labels(lab, z_scale=2.0)
    assert pts.tolist() == [[3.0, 2.0, 2.0]]


def test_non_3d_input_is_refused(tmp_path):
    tifffile.imwrite(tmp_path / "flat.tif", np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="3D"):
        real3d.load_volume(str(tmp_path / "flat.tif"))
