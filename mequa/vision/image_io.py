"""Image loading and resizing helpers for Mequa."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


PathLike = Union[str, Path]


def require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required. Install with: pip install opencv-python"
        )


def to_uint8_rgb(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.8, lo=None, hi=None) -> np.ndarray:
    """Turn any microscopy array (8-bit, 16-bit, float; grayscale or colour) into 8-bit RGB.

    16-bit fluorescence images normally use a small part of the 16-bit range, so dividing by 256
    gives a nearly black picture. Instead the intensity is stretched between two percentiles
    (or the given lo/hi, so every slice of a stack can share one scale).
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8 and lo is None:
        return np.ascontiguousarray(arr)
    gray = arr if arr.ndim == 2 else arr.astype(np.float64).mean(axis=2)
    gray = gray.astype(np.float64)
    lo = float(np.percentile(gray, p_low)) if lo is None else float(lo)
    hi = float(np.percentile(gray, p_high)) if hi is None else float(hi)
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    return np.ascontiguousarray(np.repeat((scaled * 255).astype(np.uint8)[:, :, None], 3, axis=2))


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    return img


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode PNG/JPG/TIFF bytes into an 8-bit RGB array (16-bit images are intensity-stretched)."""
    require_cv2()

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError("Could not decode image. Supported types: PNG, JPG, TIFF.")

    return to_uint8_rgb(_bgr_to_rgb(image))


def decode_stack_bytes(image_bytes: bytes, p_low: float = 1.0, p_high: float = 99.8) -> list:
    """Decode a multi-page TIFF (a z-stack) into a list of 8-bit RGB slices that share one
    intensity scale, so brightness differences between slices are preserved."""
    require_cv2()

    ok, pages = cv2.imdecodemulti(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if not ok or not pages:
        raise ValueError("Could not decode a multi-page TIFF stack.")
    pages = [_bgr_to_rgb(p) for p in pages]
    gray = [p if p.ndim == 2 else p.astype(np.float64).mean(axis=2) for p in pages]
    allv = np.concatenate([g.ravel()[:: max(1, g.size // 50000)] for g in gray])
    lo, hi = float(np.percentile(allv, p_low)), float(np.percentile(allv, p_high))
    return [to_uint8_rgb(g, lo=lo, hi=hi) for g in gray]


def load_image_file(path: PathLike) -> np.ndarray:
    """Load an image file from disk as 8-bit RGB (16-bit TIFFs are intensity-stretched)."""
    require_cv2()

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    return to_uint8_rgb(_bgr_to_rgb(image))


def resize_max_side(image_rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Resize so the longest side is at most max_side.

    This is important for CPU-only SAM inference.
    """
    require_cv2()

    if image_rgb.ndim != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    height, width = image_rgb.shape[:2]
    longest = max(height, width)

    if max_side <= 0 or longest <= max_side:
        return image_rgb

    scale = float(max_side) / float(longest)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image_rgb,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def save_image_rgb(path: PathLike, image_rgb: np.ndarray) -> None:
    """Save an RGB NumPy image to disk."""
    require_cv2()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)