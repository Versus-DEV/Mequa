"""SAM 3.1 via the hosted Meta Model API.

SAM 3.1 is concept-prompted ("all cells") rather than grid-prompted, so there
is no local point grid: one request returns boxes plus encoded masks, decoded
with the ``meta-sam-parser`` package. Transport here is stdlib urllib; only
the mask decoding needs the external parser.

Request shape mirrors Meta's documented video example, with the media part
swapped for a single image. VERIFY against dev.meta.ai/docs/sam/segmenting
before relying on the part name: commas are not a documented multi-label
interface, so use one coherent concept phrase per request.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from mequa.vision.sam_loader import SamResult


SAM31_MODEL = "sam-3.1"
DEFAULT_CONCEPT = "cell"
PARSER_PACKAGE = "meta-sam-parser"


def _base_url() -> str:
    return os.getenv("META_BASE_URL", "https://api.meta.ai/v1").rstrip("/")


def is_available() -> tuple:
    """Can we call hosted SAM 3.1 right now? Returns (ok, message)."""
    if not os.getenv("META_API_KEY", "").strip():
        return False, "META_API_KEY is not set."
    try:
        import meta_sam_parser  # noqa: F401
    except Exception:
        return False, f"{PARSER_PACKAGE} is missing. Install with: pip install {PARSER_PACKAGE}"
    return True, f"SAM 3.1 available via {_base_url()} as {SAM31_MODEL}."


def encode_image_data_url(image_rgb: np.ndarray) -> str:
    """Encode an RGB array as a PNG data URL for the request."""
    image = np.ascontiguousarray(np.asarray(image_rgb), dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required to encode the image.") from exc
    ok, buf = cv2.imencode(".png", image[:, :, ::-1])
    if not ok:
        raise RuntimeError("Could not PNG-encode the image.")
    return "data:image/png;base64," + base64.b64encode(bytes(buf)).decode("ascii")


def build_request(image_data_url: str, concept: str = DEFAULT_CONCEPT,
                  model: str = SAM31_MODEL) -> Dict[str, Any]:
    """Build the Responses-API payload for one image + one concept phrase."""
    return {
        "model": model,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": concept},
            {"type": "input_image", "image_url": image_data_url},
        ]}],
        "stream": True,
        "metadata": {"mask_encoding": "one_bit"},
    }


def post_stream(payload: Dict[str, Any], timeout: float = 120.0) -> List[Dict[str, Any]]:
    """POST to /responses and return the streamed `data:` JSON events."""
    key = os.getenv("META_API_KEY", "").strip()
    if not key:
        raise OSError("META_API_KEY is not set.")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _base_url() + "/responses",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            events = []
            for line in resp.read().decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    body = line[5:].strip()
                    if body and body != "[DONE]":
                        events.append(json.loads(body))
            return events
    except urllib.error.HTTPError as exc:
        raise OSError(f"SAM 3.1 request failed: {exc}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise OSError(f"SAM 3.1 unreachable: {type(exc).__name__}: {exc}") from exc


@dataclass
class DecodedMask:
    """One decoded SAM 3.1 mask, normalized to full-frame coordinates."""

    pixels: np.ndarray       # full-frame (H, W) bool
    bbox: tuple              # (min_row, min_col, max_row, max_col), full-frame
    score: float = 1.0


def records_to_samresult(image_shape: tuple, records: List[DecodedMask]) -> SamResult:
    """Adapter: normalized SAM 3.1 masks -> the SamResult downstream expects.

    Output is identical in kind to the ViT-B path: full-frame boolean masks
    with per-mask scores, so cells_from_masks and the geometry extractor work
    unchanged. Empty input yields the same empty SamResult, not an error.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    masks: List[np.ndarray] = []
    scores: List[float] = []
    for rec in records:
        full = np.zeros((height, width), dtype=bool)
        r0, c0, r1, c1 = (max(0, int(v)) for v in rec.bbox)
        r1, c1 = min(height, r1), min(width, c1)
        if r1 <= r0 or c1 <= c0:
            continue
        pix = np.asarray(rec.pixels, dtype=bool)
        h, w = pix.shape[:2]
        full[r0:r1, c0:c1] = pix[: r1 - r0, : c1 - c0] if (h, w) != (r1 - r0, c1 - c0) else pix
        if full.sum() == 0:
            continue
        masks.append(full)
        scores.append(float(rec.score))
    if not masks:
        return SamResult(masks=[], scores=[], source="sam31", message="SAM 3.1 returned no masks.")
    return SamResult(masks=masks, scores=scores, source="sam31",
                     message=f"SAM 3.1 returned {len(masks)} masks.")


def decode_events(events: List[Dict[str, Any]], image_shape: tuple) -> SamResult:
    """Decode streamed API events into a SamResult. Requires meta-sam-parser."""
    try:
        from meta_sam_parser import decode_mask_to_raster
    except Exception as exc:
        raise RuntimeError(f"{PARSER_PACKAGE} is required to decode SAM 3.1 masks.") from exc
    latest: Dict[Any, Dict[str, Any]] = {}
    for event in events:
        record = event.get("record") if isinstance(event, dict) else None
        if not isinstance(record, dict):
            continue
        ident = record.get("identity", record.get("object_id"))
        prev = latest.get(ident)
        if prev is None or int(record.get("revision", 0)) > int(prev.get("revision", 0)):
            latest[ident] = record
    height, width = int(image_shape[0]), int(image_shape[1])
    decoded: List[DecodedMask] = []
    for record in latest.values():
        mask = record.get("mask") or {}
        mh, mw = int(mask.get("height", 0)), int(mask.get("width", 0))
        if mh <= 0 or mw <= 0:
            continue
        pixels = np.asarray(decode_mask_to_raster(mask), dtype=bool).reshape(mh, mw)
        box = mask.get("bbox") or record.get("bbox") or [0, 0, mh, mw]
        decoded.append(DecodedMask(pixels=pixels, bbox=tuple(box),
                                   score=float(record.get("score", 1.0))))
    # Crops are placed into full-frame masks once, inside the shared adapter.
    return records_to_samresult((height, width), decoded)


def segment_image_api(image_rgb: np.ndarray, concept: str = DEFAULT_CONCEPT,
                      timeout: float = 120.0, model: Optional[str] = None) -> SamResult:
    """Segment one RGB image with hosted SAM 3.1. Raises OSError on transport failure."""
    ok, msg = is_available()
    if not ok:
        raise OSError(msg)
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    payload = build_request(encode_image_data_url(image), concept=concept, model=model or SAM31_MODEL)
    events = post_stream(payload, timeout=timeout)
    return decode_events(events, image.shape[:2])
