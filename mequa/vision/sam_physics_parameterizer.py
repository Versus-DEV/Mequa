"""Convert SAM masks into geometry features used by the Mequa physics solver.

This module is intentionally framework-independent. It takes binary masks and
returns measurable geometric features:

- area
- aspect ratio
- solidity
- boundary roughness
- circularity
- perimeter

These features are later mapped to stiffness values by
`mequa.physics.stiffness_from_geometry`.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np

try:
    from skimage import measure
except ImportError as exc:
    raise ImportError(
        "scikit-image is required. Install with: pip install scikit-image"
    ) from exc


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_axis_length(region, new_name: str, old_name: str) -> float:
    value = getattr(region, new_name, None)

    if value is None:
        value = getattr(region, old_name, 0.0)

    return float(value or 0.0)


def _safe_aspect_ratio(region) -> float:
    """Aspect ratio >= 1.0.

    Uses the new scikit-image names when available:

        axis_major_length
        axis_minor_length

    Falls back to the deprecated names for older versions:

        major_axis_length
        minor_axis_length
    """
    major = _safe_axis_length(region, "axis_major_length", "major_axis_length")
    minor = _safe_axis_length(region, "axis_minor_length", "minor_axis_length")

    if major <= 0.0:
        return 1.0

    minor = max(minor, 1e-6)
    ratio = major / minor

    # Cap extreme values caused by tiny or noisy regions.
    return _clamp(ratio, 1.0, 50.0)


def _safe_solidity(region) -> float:
    """Solidity = area / convex_area.

    1.0 means the region is convex/smooth.
    Lower values indicate concavities or irregular boundaries.
    """
    try:
        solidity = float(region.solidity)
    except Exception:
        solidity = 1.0

    if not math.isfinite(solidity):
        solidity = 1.0

    return _clamp(solidity, 0.0, 1.0)


def _safe_circularity(area: float, perimeter: float) -> float:
    """Circularity = 4*pi*area / perimeter^2.

    A perfect circle is close to 1.0.
    Irregular shapes are lower.
    """
    if perimeter <= 0.0:
        return 0.0

    circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
    return _clamp(circularity, 0.0, 1.0)


def extract_geometry_from_mask(
    mask: np.ndarray,
    mask_index: int = 0,
    min_area: int = 50,
) -> List[dict]:
    """Extract geometry features from one binary mask.

    Args:
        mask: 2D boolean or integer mask.
        mask_index: Index of the SAM mask, useful when SAM returns many masks.
        min_area: Ignore connected components smaller than this many pixels.

    Returns:
        List of feature dictionaries, one per connected component.
    """
    if mask is None:
        return []

    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    binary = mask.astype(np.uint8)
    labeled = measure.label(binary)

    if labeled.max() == 0:
        return []

    regions = measure.regionprops(labeled)
    features: List[dict] = []

    for region in regions:
        area = int(region.area)

        if area < min_area:
            continue

        aspect_ratio = _safe_aspect_ratio(region)
        solidity = _safe_solidity(region)

        perimeter = float(getattr(region, "perimeter", 0.0) or 0.0)
        circularity = _safe_circularity(area, perimeter)

        boundary_roughness = max(0.0, 1.0 - solidity)

        bbox = tuple(int(value) for value in region.bbox)
        centroid_row, centroid_col = (float(value) for value in region.centroid)

        feature = {
            "mask_index": int(mask_index),
            "cell_id": int(region.label),
            "area_px": area,
            "aspect_ratio": round(aspect_ratio, 6),
            "solidity": round(solidity, 6),
            "boundary_roughness": round(boundary_roughness, 6),
            "circularity": round(circularity, 6),
            "perimeter_px": round(perimeter, 6),
            "centroid_row": round(centroid_row, 6),
            "centroid_col": round(centroid_col, 6),
            "bbox_min_row": bbox[0],
            "bbox_min_col": bbox[1],
            "bbox_max_row": bbox[2],
            "bbox_max_col": bbox[3],
        }

        features.append(feature)

    return features


def extract_geometry_from_masks(
    masks: Sequence[np.ndarray] | np.ndarray,
    min_area: int = 50,
) -> List[dict]:
    """Extract geometry features from a collection of SAM masks.

    Args:
        masks: Either:
            - a list/tuple of 2D masks, or
            - a NumPy array with shape (N, H, W)
        min_area: Ignore components smaller than this many pixels.

    Returns:
        List of feature dictionaries.
    """
    if masks is None:
        return []

    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        iterable: Iterable[np.ndarray] = (masks[i] for i in range(masks.shape[0]))
    else:
        iterable = masks

    all_features: List[dict] = []

    for mask_index, mask in enumerate(iterable):
        all_features.extend(
            extract_geometry_from_mask(
                mask=mask,
                mask_index=mask_index,
                min_area=min_area,
            )
        )

    return all_features