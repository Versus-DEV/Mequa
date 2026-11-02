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


# ---- v6: robustness of ground-truth loading and scoring -------------------------------------

def _big_nuclei(seed=1):
    """Few, large nuclei (radius ~14 voxels) like a real stack, where a fixed 4-voxel peak spacing over-splits."""
    rng = np.random.default_rng(seed)
    shape = (36, 150, 150)
    centers = [(18, 35, 35), (18, 40, 110), (18, 110, 40), (18, 112, 112), (12, 75, 75)]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vol = np.zeros(shape, np.float32)
    labels = np.zeros(shape, np.uint16)
    for i, c in enumerate(centers, 1):
        d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        vol += 900 * (d2 <= 14 ** 2)
        labels[d2 <= 14 ** 2] = i
    vol += rng.normal(0, 25, shape)
    return vol, labels


def test_auto_settings_do_not_oversplit_large_nuclei(tmp_path):
    vol, labels = _big_nuclei()
    tifffile.imwrite(tmp_path / "v.tif", vol)
    tifffile.imwrite(tmp_path / "gt.tif", labels)
    res = real3d.run(str(tmp_path / "v.tif"), str(tmp_path / "gt.tif"), radius=None, z_scale=1.0, knn=3, sigma=2.0, min_distance=None)
    assert res["expert_labels"]["nuclei"] == 5
    assert res["detection_vs_expert_labels"]["radius_mode"] == "auto"
    assert res["detection_inside_expert_nuclei"]["recall"] == 1.0
    assert res["detection_inside_expert_nuclei"]["detected"] <= 8


def test_rgb_label_volume_keeps_one_id_per_colour(tmp_path):
    lab = np.zeros((4, 8, 8, 3), np.uint8)
    lab[1, 1:3, 1:3] = (255, 0, 0)
    lab[1, 5:7, 5:7] = (0, 255, 0)
    lab[2, 1:3, 5:7] = (0, 0, 255)
    tifffile.imwrite(tmp_path / "rgb.tif", lab, photometric="rgb")
    out = real3d.load_volume(str(tmp_path / "rgb.tif"), labels=True)
    assert out.shape == (4, 8, 8)
    assert sorted(np.unique(out).tolist()) == [0, 1, 2, 3]


def test_binary_mask_is_split_into_instances():
    m = np.zeros((3, 10, 10), np.uint8)
    m[1, 1:3, 1:3] = 255
    m[1, 6:8, 6:8] = 255
    inst, how = real3d.instances_from_labels(m)
    assert len(np.unique(inst)) == 3 and "connected components" in how


def test_label_shape_mismatch_is_an_error_and_permuted_axes_are_fixed():
    lab = np.zeros((6, 4, 5), np.uint16)
    fixed, note = real3d.match_label_axes(lab, (4, 6, 5))
    assert fixed.shape == (4, 6, 5) and "permuted" in note
    with pytest.raises(ValueError, match="same voxels"):
        real3d.match_label_axes(lab, (7, 7, 7))


def test_too_few_detections_gives_a_note_not_a_crash():
    out = real3d.analyze_points(np.random.default_rng(0).random((5, 3)), knn=10)
    assert "note" in out


def test_nucleus_cut_by_the_stack_edge_is_detected_inside_the_stack_not_on_the_edge():
    """BBBC033 run: detections sat on z=0 and z=31 because the distance transform saw the stack edge as open space."""
    shape = (12, 80, 80)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vol = np.zeros(shape, np.float32)
    # a flat nucleus whose centre is slice 6, big in x/y, thick enough to touch both z faces
    vol[(((yy - 40) ** 2 + (xx - 40) ** 2) <= 25 ** 2)] = 900
    vol += np.random.default_rng(0).normal(0, 10, shape).astype(np.float32)
    pts = real3d.detect_nuclei_3d(vol, sigma=1.0, min_distance=None)
    assert len(pts) >= 1
    assert np.all((pts[:, 2] >= 3) & (pts[:, 2] <= 8)), pts[:, 2]
    assert np.all(pts[:, 2] <= shape[0] - 1) and np.all(pts[:, 2] >= 0)


# ---- v6.2: textured, flat, touching nuclei (the BBBC033 over-segmentation) -------------------

def _textured_flat_nuclei(seed=3):
    import scipy.ndimage as ndi
    rng = np.random.default_rng(seed)
    shape = (20, 220, 220)
    cs = [(10, 50, 50), (10, 55, 120), (9, 120, 60), (10, 125, 135), (9, 180, 180), (10, 60, 190), (11, 185, 70), (10, 190, 130)]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    vol = np.zeros(shape, np.float32)
    lab = np.zeros(shape, np.uint16)
    for i, c in enumerate(cs, 1):
        m = (((zz - c[0]) / 4.5) ** 2 + ((yy - c[1]) / 24) ** 2 + ((xx - c[2]) / 21) ** 2) <= 1
        lab[m] = i
        vol += (700 + ndi.gaussian_filter(rng.normal(0, 1, shape).astype(np.float32), 3) * 400) * m
    vol += rng.normal(0, 40, shape)
    return ndi.gaussian_filter(vol, 0.7), lab


def test_textured_flat_nuclei_are_not_split():
    vol, lab = _textured_flat_nuclei()
    pts = real3d.detect_nuclei_3d(vol, sigma=2.0, min_distance="auto")
    sc = real3d.match_inside(pts, lab)
    assert sc["recall"] == 1.0 and sc["splits"] == 0 and sc["detected"] == 8


def test_touching_nuclei_are_still_separated():
    shape = (20, 160, 200)
    zz, yy, xx = np.mgrid[0:20, 0:160, 0:200]

    def ell(c, r):
        return (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2) <= 1
    lab = np.zeros(shape, np.uint16)
    lab[ell((10, 80, 60), (5, 26, 22))] = 1
    lab[ell((10, 80, 102), (5, 26, 22)) & (lab == 0)] = 2
    vol = (lab > 0) * 800.0 + np.random.default_rng(0).normal(0, 40, shape)
    pts = real3d.detect_nuclei_3d(vol.astype(np.float32), sigma=2.0, min_distance="auto")
    assert len(pts) == 2


def _blobs(shape=(30, 60, 60), centres=((10, 15, 15), (20, 40, 40), (12, 15, 45)), amp=(200, 200, 90)):
    import numpy as np
    zz, yy, xx = np.indices(shape)
    v = np.zeros(shape, np.float32)
    for (z, y, x), a in zip(centres, amp):
        v += a * np.exp(-(((zz - z) ** 2) + (yy - y) ** 2 + (xx - x) ** 2) / (2 * 4.0 ** 2))
    return v


def test_threshold_scale_catches_dim_nucleus():
    from mequa.real3d import detect_nuclei_3d
    v = _blobs()
    a = detect_nuclei_3d(v, sigma=1.0, threshold_scale=1.0)
    b = detect_nuclei_3d(v, sigma=1.0, threshold_scale=0.4)
    assert len(b) >= len(a)


def test_refine_keeps_count_and_stays_close():
    import numpy as np
    from mequa.real3d import detect_nuclei_3d
    v = _blobs()
    a = detect_nuclei_3d(v, sigma=1.0, refine=False)
    b = detect_nuclei_3d(v, sigma=1.0, refine=True)
    assert len(a) == len(b)
    assert np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).max() < 4


def test_diagnostics_in_run(tmp_path):
    import numpy as np, tifffile
    from mequa.real3d import run
    v = _blobs(amp=(200, 200, 200))
    lab = np.zeros(v.shape, np.uint16)
    for i, (z, y, x) in enumerate(((10, 15, 15), (20, 40, 40), (12, 15, 45)), 1):
        lab[max(z-3,0):z+4, y-3:y+4, x-3:x+4] = i
    tifffile.imwrite(tmp_path / "v.tif", v)
    tifffile.imwrite(tmp_path / "l.tif", lab)
    r = run(str(tmp_path / "v.tif"), str(tmp_path / "l.tif"), None, 1.0, 10, 1.0, "auto")
    d = r["diagnostics"]
    assert d["matched_offsets_vox"]["pairs"] >= 1
    assert len(d["z_profile"]) == 30
    assert isinstance(d["missed_nuclei"], list)
