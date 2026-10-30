"""Overlay masks on an RGB image for dashboard visualization."""

from __future__ import annotations

from typing import List

import numpy as np


PALETTE = [
    (255, 80, 80),
    (80, 255, 120),
    (80, 160, 255),
    (255, 200, 80),
    (220, 100, 255),
    (100, 255, 220),
    (255, 140, 60),
    (180, 255, 80),
]


def overlay_masks(
    image_rgb: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.40,
) -> np.ndarray:
    """Blend colored masks over an RGB image."""
    if image_rgb is None:
        raise ValueError("image_rgb is required.")

    base = np.asarray(image_rgb, dtype=np.float32).copy()

    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("image_rgb must have shape (H, W, 3).")

    color_layer = np.zeros_like(base, dtype=np.float32)
    mask_present = np.zeros(base.shape[:2], dtype=bool)

    for index, mask in enumerate(masks):
        if mask is None:
            continue

        binary = np.asarray(mask, dtype=bool)

        if binary.shape[:2] != base.shape[:2]:
            raise ValueError(
                f"Mask {index} shape {binary.shape} does not match "
                f"image shape {base.shape[:2]}."
            )

        color = PALETTE[index % len(PALETTE)]
        color_layer[binary] = color
        mask_present |= binary

    if not mask_present.any():
        return base.astype(np.uint8)

    base[mask_present] = (
        (1.0 - alpha) * base[mask_present]
        + alpha * color_layer[mask_present]
    )

    return base.astype(np.uint8)