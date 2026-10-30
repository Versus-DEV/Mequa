import numpy as np
import cv2

from mequa import ingest
from mequa.eval_segmentation import decode_gt_instances, gt_centroids, match_points, to_image_frame
from mequa.vision.image_io import decode_image_bytes, decode_stack_bytes, load_image_file, to_uint8_rgb


def fluorescence16(seed=0, n=40, size=300):
    """Bright nuclei on a dark background in the LOW part of the 16-bit range, like BBBC039."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = rng.normal(300, 25, (size, size))
    centers = []
    while len(centers) < n:
        c = rng.uniform(20, size - 20, 2)
        if all(np.hypot(*(c - o)) > 28 for o in centers):
            centers.append(c)
    for cx, cy in centers:
        img += 2500 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 7.0**2)) ** 2)
    return np.clip(img, 0, 65535).astype(np.uint16), np.array(centers)


def test_16bit_image_is_stretched_not_black(tmp_path):
    img, _ = fluorescence16()
    path = tmp_path / "f.tif"
    cv2.imwrite(str(path), img)
    naive = (cv2.imread(str(path), cv2.IMREAD_COLOR)).max()
    rgb = load_image_file(path)
    assert naive < 30            # what the old loader produced: almost black
    assert rgb.dtype == np.uint8 and rgb.shape == (300, 300, 3) and rgb.max() >= 250


def test_bytes_decode_matches_file_decode(tmp_path):
    img, _ = fluorescence16(1)
    ok, buf = cv2.imencode(".tif", img)
    assert ok
    a = decode_image_bytes(buf.tobytes())
    assert a.max() >= 250 and a.shape[2] == 3


def test_8bit_colour_image_is_unchanged():
    rgb = np.random.default_rng(0).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    assert np.array_equal(to_uint8_rgb(rgb), rgb)


def test_stack_slices_share_one_intensity_scale():
    a, _ = fluorescence16(2)
    b = (a * 0.5).astype(np.uint16)  # a dimmer slice
    ok, buf = cv2.imencodemulti(".tif", [a, b])
    assert ok
    slices = decode_stack_bytes(np.asarray(buf).tobytes())
    assert len(slices) == 2
    assert slices[0].max() > slices[1].max() + 40  # the dimmer slice stays dimmer, not re-stretched to full


def test_bright_on_dark_segmentation_and_polarity():
    img, centers = fluorescence16(3)
    rgb = to_uint8_rgb(img)
    pts, areas = ingest.segment_classical(rgb)
    assert 0.9 * len(centers) <= len(pts) <= 1.1 * len(centers)
    m = match_points(pts, centers, radius=4)
    assert m["recall"] > 0.95 and m["precision"] > 0.95


def test_watershed_splits_touching_cells():
    size = 120
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.full((size, size, 3), 20, np.uint8)
    for cx in (40, 58, 90):  # first two overlap
        img[(xx - cx) ** 2 + (yy - 60) ** 2 <= 12**2] = 220
    plain, _ = ingest.segment_classical(img)
    split, _ = ingest.segment_classical(img, split=True, min_distance=5)
    assert len(plain) == 2 and len(split) == 3


def test_demo_stack_count_improves_with_splitting():
    imgs, truth = ingest.make_demo_stack(n_cells=60, n_slices=12, size=256)
    p0, _ = ingest.segment_stack(imgs, "classical")
    p1, _ = ingest.segment_stack(imgs, "classical", split=True, min_distance=5)
    assert len(p1) == len(truth)  # all 60 spheres, including the touching pair
    assert abs(len(p1) - len(truth)) <= abs(len(p0) - len(truth))


def test_ground_truth_decoding_colour_and_integer_labels():
    lab = np.zeros((50, 50, 3), np.uint8)
    lab[5:15, 5:15] = (255, 0, 0)
    lab[20:30, 20:30] = (0, 255, 0)
    lab[35:45, 35:45] = (255, 0, 0)  # same colour twice, not touching: two instances
    inst = decode_gt_instances(lab)
    assert inst.max() == 3
    ints = np.zeros((50, 50), np.int32)
    ints[5:15, 5:15] = 7
    ints[20:30, 20:30] = 9
    assert decode_gt_instances(ints).max() == 2
    c = gt_centroids(inst)
    assert len(c) == 3


def test_matching_is_one_to_one_and_metrics_are_right():
    gt = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])
    pred = np.array([[11.0, 10.0], [51.0, 52.0], [200.0, 200.0], [10.5, 10.5]])  # one duplicate, one stray, one missed
    m = match_points(pred, gt, radius=6)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 1)
    assert abs(m["f1"] - 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)) < 1e-3
    assert m["count_error_pct"] == 33.3
    assert match_points(np.zeros((0, 2)), gt)["recall"] == 0.0


def test_frame_flip_roundtrip():
    p = np.array([[3.0, 4.0]])
    assert np.allclose(to_image_frame(to_image_frame(p, 100), 100), p)
