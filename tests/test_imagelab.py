import io
import zipfile

import numpy as np
import pytest
import tifffile

from mequa import imagelab as il


def _tif_bytes(a):
    b = io.BytesIO()
    tifffile.imwrite(b, a)
    return b.getvalue()


@pytest.fixture(scope="module")
def stack():
    return il.make_synthetic_stack(8, (22, 260, 260), seed=1)


def test_multipage_tiff_loads_as_3d_and_detects_each_nucleus(stack):
    arr = il.load_array(_tif_bytes(stack.astype(np.uint16)), "s.tif")
    assert arr.shape == stack.shape
    det = il.detect(arr)
    assert det.kind == "3D stack" and det.dim == 3 and det.n == 8


def test_downsample_scales_coordinates_back(stack):
    full = il.detect(stack)
    half = il.detect(stack, downsample_by=2)
    assert half.n == full.n
    from scipy.spatial.distance import cdist

    nearest = cdist(full.points[:, :2], half.points[:, :2]).min(axis=1)
    assert nearest.max() < 12      # every nucleus is found at the same place, within a few original pixels


def test_2d_image_gives_xy_points_and_areas(stack):
    det = il.detect(stack[11])
    assert det.kind == "2D image" and det.points.shape[1] == 2 and det.areas is not None and len(det.areas) == det.n >= 3
    assert il.stiffness_from_detection(det).shape == (det.n,)


def test_dark_nuclei_need_the_bright_switch(stack):
    inv = stack.max() - stack
    assert il.detect(inv, bright=False).n == il.detect(stack).n
    with pytest.raises(il.ImageLabError, match="dark"):
        il.detect(inv[11], bright=True)


def test_zip_of_png_slices_becomes_a_stack(stack):
    from PIL import Image

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(5):
            im = io.BytesIO()
            Image.fromarray((stack[i] / stack.max() * 255).astype(np.uint8)).save(im, "PNG")
            z.writestr(f"slice_{i:02d}.png", im.getvalue())
    arr = il.load_array(buf.getvalue(), "slices.zip")
    assert arr.shape == (5, 260, 260)


def test_rgb_image_is_averaged_to_grey_and_bad_files_explain_themselves():
    rgb = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    assert il.load_array(_tif_bytes(rgb), "c.tif").shape == (60, 80)
    with pytest.raises(il.ImageLabError, match="unsupported"):
        il.load_array(b"x", "notes.txt")
    with pytest.raises(il.ImageLabError, match="zip"):
        il.load_array(b"not a zip", "bad.zip")


def test_flat_image_and_too_thin_stack_are_refused_with_advice():
    with pytest.raises(il.ImageLabError, match="flat"):
        il.detect(np.full((50, 50), 7.0, np.float32))
    with pytest.raises(il.ImageLabError, match="3 slices"):
        il.detect(np.random.default_rng(0).normal(size=(2, 40, 40)).astype(np.float32))


def test_combine_slices_orders_by_name_and_rejects_mismatches():
    a, b = np.zeros((20, 20), np.float32), np.ones((20, 20), np.float32)
    out = il.combine_slices([("b.tif", _tif_bytes(b)), ("a.tif", _tif_bytes(a))])
    assert out[0].max() == 0 and out[1].min() == 1
    with pytest.raises(il.ImageLabError, match="same size"):
        il.combine_slices([("a.tif", _tif_bytes(a)), ("b.tif", _tif_bytes(np.ones((9, 9), np.float32)))])
