"""Convert SAM-derived geometry features into spring-network stiffness values.

This is the core physics-parameterization step:

SAM does not merely draw masks.
SAM mask geometry becomes mechanical input for the solver.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def geometry_to_stiffness(
    features: Sequence[dict],
    base_stiffness: float = 1.0,
    roughness_weight: float = 5.0,
    elongation_weight: float = 0.5,
    min_stiffness: float = 0.1,
    max_stiffness: float = 20.0,
) -> List[dict]:
    """Map geometry features to stiffness values.

    The mapping is intentionally simple, measurable, and explainable:

    - Lower solidity means more irregular/concave boundary.
    - Higher aspect ratio means more elongated cell/region.
    - Irregular or elongated regions receive higher stiffness.

    This is a research model, not a clinical diagnostic model.

    Args:
        features: Output from sam_physics_parameterizer.extract_geometry_from_masks.
        base_stiffness: Baseline stiffness value.
        roughness_weight: How strongly boundary irregularity increases stiffness.
        elongation_weight: How strongly elongation increases stiffness.
        min_stiffness: Lower clamp for stiffness.
        max_stiffness: Upper clamp for stiffness.

    Returns:
        New feature dictionaries with `stiffness_k` added.
    """
    enriched: List[dict] = []

    for feature in features:
        solidity = float(feature.get("solidity", 1.0))
        aspect_ratio = float(feature.get("aspect_ratio", 1.0))

        roughness = max(0.0, 1.0 - solidity)
        elongation = max(0.0, aspect_ratio - 1.0)

        stiffness = (
            base_stiffness
            * (1.0 + roughness_weight * roughness)
            * (1.0 + elongation_weight * elongation)
        )

        stiffness = _clamp(stiffness, min_stiffness, max_stiffness)

        new_feature = dict(feature)
        new_feature["stiffness_k"] = round(float(stiffness), 6)

        enriched.append(new_feature)

    return enriched


def stiffness_array(features_with_stiffness: Sequence[dict]) -> np.ndarray:
    """Return a float32 NumPy array of stiffness values.

    This array can be passed to the NumPy/Rust/Julia solver.
    """
    values = [
        float(feature.get("stiffness_k", 1.0))
        for feature in features_with_stiffness
    ]

    return np.asarray(values, dtype=np.float32)