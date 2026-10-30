"""From SAM masks to a cell network: one node per cell, one stiffness per cell.

SAM's automatic generator returns overlapping masks (a cell, a cluster containing it, the
background). We keep one mask per cell by processing masks from smallest to largest and rejecting
any that mostly covers pixels already claimed. Each kept mask becomes a node at its centroid, and
its shape features set the node's stiffness through mequa.physics.stiffness_from_geometry.

Stiffness here is a modelling assumption (irregular or elongated cells are stiffer), normalised
to the median cell and clipped to [0.5, 2.0] so the relaxation stays stable. It is not a measured
tissue property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from mequa.physics.stiffness_from_geometry import geometry_to_stiffness
from mequa.vision.sam_physics_parameterizer import extract_geometry_from_masks

STIFFNESS_RANGE = (0.5, 2.0)


@dataclass
class CellSet:
    points: np.ndarray          # (n, 2) x, y with y pointing up
    stiffness: np.ndarray       # (n,) float64, normalised
    areas: np.ndarray           # (n,) pixels
    features: List[dict] = field(default_factory=list)
    masks_in: int = 0
    masks_dropped_background: int = 0
    masks_dropped_overlap: int = 0

    @property
    def n(self) -> int:
        return int(len(self.points))


def filter_masks(masks: Sequence[np.ndarray], min_area: int = 60, max_area_frac: float = 0.2, max_overlap: float = 0.3):
    """Keep one mask per cell. Returns (kept_masks, n_background, n_overlap)."""
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if not masks:
        return [], 0, 0
    total = masks[0].size
    sized = []
    background = 0
    for m in masks:
        a = int(m.sum())
        if a < min_area:
            continue
        if a > max_area_frac * total:
            background += 1
            continue
        sized.append((a, m))
    sized.sort(key=lambda t: t[0])
    claimed = np.zeros(masks[0].shape, dtype=bool)
    kept, overlap = [], 0
    for a, m in sized:
        if (m & claimed).sum() > max_overlap * a:
            overlap += 1
            continue
        kept.append(m)
        claimed |= m
    return kept, background, overlap


def normalize_stiffness(raw, low=STIFFNESS_RANGE[0], high=STIFFNESS_RANGE[1]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size == 0:
        return raw
    med = float(np.median(raw))
    return np.clip(raw / max(med, 1e-12), low, high)


def cells_from_masks(masks: Sequence[np.ndarray], min_area: int = 60, max_area_frac: float = 0.2, max_overlap: float = 0.3) -> CellSet:
    kept, background, overlap = filter_masks(masks, min_area, max_area_frac, max_overlap)
    if not kept:
        return CellSet(np.zeros((0, 2)), np.zeros(0), np.zeros(0), [], len(masks), background, overlap)
    height = kept[0].shape[0]
    feats = []
    for i, m in enumerate(kept):
        comps = extract_geometry_from_masks([m], min_area=min_area)
        if not comps:
            continue
        biggest = max(comps, key=lambda f: f["area_px"])  # a mask can split into pieces; use the main one
        biggest = dict(biggest)
        biggest["mask_index"] = i
        feats.append(biggest)
    feats = geometry_to_stiffness(feats)
    raw = [f["stiffness_k"] for f in feats]
    norm = normalize_stiffness(raw)
    pts = np.array([[f["centroid_col"], height - f["centroid_row"]] for f in feats], dtype=np.float64)
    areas = np.array([f["area_px"] for f in feats], dtype=np.float64)
    for f, s in zip(feats, norm):
        f["stiffness_norm"] = round(float(s), 6)
    return CellSet(np.ascontiguousarray(pts), np.ascontiguousarray(norm), areas, feats, len(masks), background, overlap)
