"""Focused tests for the SAM 3.1 upgrade.

Offline: request shape, mask adapter output compatibility, loader routing.
Needs numpy; the live API path is not exercised here.
"""
import numpy as np

from mequa.vision import sam31
from mequa.vision.sam_loader import SamLoader, SamResult


def test_build_request_shape():
    payload = sam31.build_request("data:image/png;base64,AAA", concept="cell")
    assert payload["model"] == "sam-3.1"
    assert payload["stream"] is True
    assert payload["metadata"] == {"mask_encoding": "one_bit"}
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "cell"}
    assert content[1]["type"] == "input_image"


def test_adapter_returns_full_frame_bool_masks():
    crop = np.ones((4, 5), dtype=bool)
    recs = [sam31.DecodedMask(pixels=crop, bbox=(2, 3, 6, 8), score=0.9)]
    res = sam31.records_to_samresult((10, 12), recs)
    assert isinstance(res, SamResult) and res.source == "sam31"
    assert len(res.masks) == 1 and res.scores == [0.9]
    mask = res.masks[0]
    assert mask.shape == (10, 12) and mask.dtype == bool
    assert mask.sum() == 20 and mask[2:6, 3:8].all() and (~mask[:2, :]).all()


def test_adapter_drops_empty_and_bad_boxes():
    recs = [
        sam31.DecodedMask(pixels=np.zeros((3, 3), dtype=bool), bbox=(0, 0, 3, 3)),
        sam31.DecodedMask(pixels=np.ones((2, 2), dtype=bool), bbox=(5, 5, 5, 8)),
    ]
    res = sam31.records_to_samresult((10, 10), recs)
    assert res.masks == [] and "no masks" in res.message


def test_adapter_empty_input_matches_legacy_empty():
    res = sam31.records_to_samresult((8, 8), [])
    assert res.masks == [] and res.scores == []


def test_adapter_output_feeds_cells_from_masks():
    """Step 3: SamResult from the SAM 3.1 adapter must satisfy cells_from_masks
    exactly like ViT-B output (same shapes in, CellSet out)."""
    from mequa.vision.cells import cells_from_masks

    a = np.zeros((40, 40), dtype=bool)
    a[5:15, 5:15] = True
    b = np.zeros((40, 40), dtype=bool)
    b[25:35, 25:35] = True
    res = sam31.records_to_samresult(
        (40, 40),
        [sam31.DecodedMask(pixels=a[5:15, 5:15], bbox=(5, 5, 15, 15), score=0.9),
         sam31.DecodedMask(pixels=b[25:35, 25:35], bbox=(25, 25, 35, 35), score=0.8)],
    )
    cells = cells_from_masks(res.masks)
    assert cells.n == 2
    assert cells.points.shape == (2, 2) and cells.stiffness.shape == (2,)
    assert cells.areas.shape == (2,) and len(cells.features) == 2
    assert {f["area_px"] for f in cells.features} == {100}


def test_loader_rejects_unknown_backend():
    try:
        SamLoader(backend="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_loader_backend_from_env(monkeypatch):
    monkeypatch.setenv("SAM_BACKEND", "local")
    assert SamLoader().backend == "local"
    assert SamLoader(backend="api").backend == "api"


def test_loader_api_mode_uses_api_result(monkeypatch):
    made = SamResult(masks=[np.ones((6, 6), dtype=bool)], scores=[1.0],
                     source="sam31", message="x")
    monkeypatch.setattr(sam31, "segment_image_api", lambda image, concept="cell": made)
    out = SamLoader(backend="api").generate_masks(np.zeros((6, 6, 3), dtype=np.uint8))
    assert out.source == "sam31" and len(out.masks) == 1


def test_loader_auto_falls_back_to_local(monkeypatch):
    def boom(image, concept="cell"):
        raise OSError("down")

    monkeypatch.setattr(sam31, "segment_image_api", boom)
    monkeypatch.setattr(SamLoader, "load", lambda self: None)

    class FakeGen:
        def generate(self, image):
            return []

    loader = SamLoader(backend="auto")
    monkeypatch.setattr(loader, "_generator", FakeGen())
    out = loader.generate_masks(np.zeros((6, 6, 3), dtype=np.uint8))
    assert out.source == "sam" and out.masks == []
