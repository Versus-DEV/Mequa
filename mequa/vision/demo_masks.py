"""Demo masks for validating the Mequa pipeline before SAM is connected.

These are not real biology. They exist so the dashboard and solver pipeline
can be tested end-to-end.
"""

from __future__ import annotations

from typing import List

import numpy as np


def make_demo_masks(width: int = 256, height: int = 256) -> List[np.ndarray]:
    """Create three synthetic masks: round, elongated, irregular."""
    y, x = np.indices((height, width)).astype(np.float32)

    round_cell = ((x - 70.0) ** 2 + (y - 70.0) ** 2) <= (32.0 ** 2)

    elongated = (
        (((x - 180.0) / 18.0) ** 2)
        + (((y - 170.0) / 48.0) ** 2)
    ) <= 1.0

    bite = ((x - 166.0) ** 2 + (y - 170.0) ** 2) <= (12.0 ** 2)

    irregular = elongated & ~bite

    return [
        round_cell.astype(bool),
        elongated.astype(bool),
        irregular.astype(bool),
    ]


def make_demo_image(masks: List[np.ndarray]) -> np.ndarray:
    """Create a simple RGB image from demo masks for display only."""
    if masks:
        height, width = masks[0].shape[:2]
    else:
        height, width = 256, 256

    image = np.full((height, width, 3), 18, dtype=np.uint8)

    colors = [
        (80, 140, 200),
        (90, 190, 120),
        (220, 160, 80),
        (190, 100, 180),
        (120, 120, 220),
    ]

    for index, mask in enumerate(masks):
        color = colors[index % len(colors)]
        image[mask.astype(bool)] = color

    return image