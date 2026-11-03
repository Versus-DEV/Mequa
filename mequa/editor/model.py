"""A 3D cell model you can edit: select, move, rotate, scale, mirror, add, delete, duplicate,
set stiffness, tag groups, clean up, with unlimited-ish undo/redo.

The model is a point cloud of cells with one stiffness value and one group id per cell. The strain
solver builds its mechanical links from nearest neighbours, so editing the cells is how the model is
edited: moving a cell changes its links, deleting a cell removes them, and a stiffer cell loads its
links harder. Nothing here depends on Streamlit, so it is tested directly.

2D datasets are held with z = 0 and flagged dim=2; `analysis_points()` hands the solver 2 columns
again, and exports drop z, so a 2D file round-trips as a 2D file.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

AXES = {"x": 0, "y": 1, "z": 2}
MAX_HISTORY = 100
MIN_STIFFNESS, MAX_STIFFNESS = 0.05, 20.0


def apply_selection(current: np.ndarray, picked: Iterable[int], mode: str = "replace") -> np.ndarray:
    """Combine a boolean selection with newly picked cell ids. mode: replace, add, remove, intersect."""
    n = len(current)
    new = np.zeros(n, dtype=bool)
    idx = np.asarray(list(picked), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < n)]
    new[idx] = True
    if mode == "replace":
        return new
    if mode == "add":
        return current | new
    if mode == "remove":
        return current & ~new
    if mode == "intersect":
        return current & new
    raise ValueError(f"unknown selection mode {mode!r}")


@dataclass
class _Snapshot:
    points: np.ndarray
    stiffness: np.ndarray
    group: np.ndarray
    note: str


@dataclass
class CellModel:
    points: np.ndarray                      # (n, 3)
    stiffness: np.ndarray                   # (n,)
    group: np.ndarray                       # (n,) int
    dim: int = 3
    name: str = "model"
    log: List[str] = field(default_factory=list)
    _undo: List[_Snapshot] = field(default_factory=list, repr=False)
    _redo: List[_Snapshot] = field(default_factory=list, repr=False)
    _original: Optional[_Snapshot] = field(default=None, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(cls, points, stiffness=None, group=None, name="model") -> "CellModel":
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError(f"points must be (n, 2) or (n, 3), got {pts.shape}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points contain NaN or infinite values")
        dim = int(pts.shape[1])
        if dim == 2:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        n = len(pts)
        st = np.ones(n) if stiffness is None else np.asarray(stiffness, dtype=np.float64).copy()
        gr = np.zeros(n, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64).copy()
        if st.shape != (n,) or gr.shape != (n,):
            raise ValueError("stiffness and group need one value per cell")
        if not np.all(np.isfinite(st)) or np.any(st <= 0):
            raise ValueError("stiffness values must be finite and positive")
        m = cls(np.ascontiguousarray(pts), st, gr, dim=dim, name=name)
        m._original = m._snap("original")
        return m

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(len(self.points))

    def analysis_points(self) -> np.ndarray:
        return np.ascontiguousarray(self.points[:, : self.dim])

    def analysis_stiffness(self) -> Optional[np.ndarray]:
        return None if np.allclose(self.stiffness, self.stiffness[0]) and self.stiffness[0] == 1.0 else self.stiffness.copy()

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n == 0:
            return np.zeros(3), np.zeros(3)
        return self.points.min(axis=0), self.points.max(axis=0)

    def _snap(self, note: str) -> _Snapshot:
        return _Snapshot(self.points.copy(), self.stiffness.copy(), self.group.copy(), note)

    def _restore(self, s: _Snapshot) -> None:
        self.points, self.stiffness, self.group = s.points.copy(), s.stiffness.copy(), s.group.copy()

    def _commit(self, note: str) -> None:
        """Call BEFORE mutating: records the current state so undo can return to it."""
        self._undo.append(self._snap(note))
        if len(self._undo) > MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()
        self.log.append(note)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> Optional[str]:
        if not self._undo:
            return None
        s = self._undo.pop()
        self._redo.append(self._snap(s.note))
        self._restore(s)
        self.log.append(f"undo: {s.note}")
        return s.note

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        s = self._redo.pop()
        self._undo.append(self._snap(s.note))
        # s is the state after the op (stored by undo); the snapshot pushed above is the state before it
        self._restore(s)
        self.log.append(f"redo: {s.note}")
        return s.note

    def revert_all(self) -> None:
        self._commit("revert to the loaded model")
        self._restore(self._original)

    def original_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        o = self._original
        return o.points[:, : self.dim].copy(), o.stiffness.copy()

    # ------------------------------------------------------------------ selection (return boolean masks)
    def mask(self, ids: Optional[Sequence[int]] = None) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        if ids is not None:
            a = np.asarray(list(ids), dtype=np.int64)
            m[a[(a >= 0) & (a < self.n)]] = True
        return m

    def select_box(self, lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return np.all((self.points >= np.minimum(lo, hi)) & (self.points <= np.maximum(lo, hi)), axis=1)

    def select_sphere(self, center: Sequence[float], radius: float) -> np.ndarray:
        return np.linalg.norm(self.points - np.asarray(center, float), axis=1) <= float(radius)

    def select_value(self, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        v = np.asarray(values, float)
        if v.shape != (self.n,):
            raise ValueError("values must have one entry per cell")
        return (v >= min(lo, hi)) & (v <= max(lo, hi))

    def select_group(self, g: int) -> np.ndarray:
        return self.group == int(g)

    def select_nearest(self, point: Sequence[float], k: int) -> np.ndarray:
        m = np.zeros(self.n, dtype=bool)
        if self.n:
            _, idx = cKDTree(self.points).query(np.asarray(point, float), k=min(int(k), self.n))
            m[np.atleast_1d(idx)] = True
        return m

    def grow(self, sel: np.ndarray, k: int = 4) -> np.ndarray:
        """Add the k nearest neighbours of every selected cell."""
        if not sel.any() or self.n <= 1:
            return sel.copy()
        _, idx = cKDTree(self.points).query(self.points[sel], k=min(k + 1, self.n))
        out = sel.copy()
        out[np.unique(idx)] = True
        return out

    # ------------------------------------------------------------------ edits
    def _need(self, sel: np.ndarray, what: str) -> np.ndarray:
        sel = np.asarray(sel, dtype=bool)
        if sel.shape != (self.n,):
            raise ValueError("selection does not match the model; select again")
        if not sel.any():
            raise ValueError(f"{what}: nothing is selected")
        return sel

    def _pivot(self, sel: np.ndarray, about) -> np.ndarray:
        if about is None or (isinstance(about, str) and about == "selection"):
            return self.points[sel].mean(axis=0)
        if isinstance(about, str) and about == "origin":
            return np.zeros(3)
        if isinstance(about, str) and about == "model":
            return self.points.mean(axis=0)
        return np.asarray(about, float)

    def _flatten(self) -> None:
        if self.dim == 2:
            self.points[:, 2] = 0.0

    def move(self, sel, delta: Sequence[float]) -> int:
        sel = self._need(sel, "move")
        d = np.asarray(delta, float)
        if not np.all(np.isfinite(d)):
            raise ValueError("move distance must be finite")
        self._commit(f"move {int(sel.sum())} cells by ({d[0]:.3g}, {d[1]:.3g}, {d[2]:.3g})")
        self.points[sel] += d
        self._flatten()
        return int(sel.sum())

    def rotate(self, sel, axis: str, degrees: float, about="selection") -> int:
        sel = self._need(sel, "rotate")
        if axis not in AXES:
            raise ValueError("axis must be x, y or z")
        if self.dim == 2 and axis != "z":
            raise ValueError("a 2D model can only be rotated about z")
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        R = {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
             "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
             "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
        p = self._pivot(sel, about)
        self._commit(f"rotate {int(sel.sum())} cells {degrees:g} deg about {axis}")
        self.points[sel] = (self.points[sel] - p) @ R.T + p
        self._flatten()
        return int(sel.sum())

    def scale(self, sel, factor, about="selection") -> int:
        sel = self._need(sel, "scale")
        f = np.broadcast_to(np.asarray(factor, float), (3,)).copy()
        if np.any(f <= 0) or not np.all(np.isfinite(f)):
            raise ValueError("scale factors must be positive")
        if self.dim == 2:
            f[2] = 1.0
        p = self._pivot(sel, about)
        self._commit(f"scale {int(sel.sum())} cells by ({f[0]:.3g}, {f[1]:.3g}, {f[2]:.3g})")
        self.points[sel] = (self.points[sel] - p) * f + p
        return int(sel.sum())

    def mirror(self, sel, axis: str, about="selection") -> int:
        sel = self._need(sel, "mirror")
        if axis not in AXES or (self.dim == 2 and axis == "z"):
            raise ValueError("choose an axis the model has")
        p = self._pivot(sel, about)
        a = AXES[axis]
        self._commit(f"mirror {int(sel.sum())} cells across {axis}")
        self.points[sel, a] = 2 * p[a] - self.points[sel, a]
        return int(sel.sum())

    def delete(self, sel) -> int:
        sel = self._need(sel, "delete")
        if int((~sel).sum()) < 8:
            raise ValueError("a model needs at least 8 cells for the solver; keep more cells")
        k = int(sel.sum())
        self._commit(f"delete {k} cells")
        keep = ~sel
        self.points, self.stiffness, self.group = self.points[keep], self.stiffness[keep], self.group[keep]
        return k

    def duplicate(self, sel, offset: Sequence[float] = (0.5, 0.5, 0.5)) -> np.ndarray:
        """Copy the selected cells, shifted by offset. Returns the mask of the new cells."""
        sel = self._need(sel, "duplicate")
        d = np.asarray(offset, float)
        self._commit(f"duplicate {int(sel.sum())} cells")
        new_pts = self.points[sel] + d
        self.points = np.vstack([self.points, new_pts])
        self.stiffness = np.concatenate([self.stiffness, self.stiffness[sel]])
        self.group = np.concatenate([self.group, self.group[sel]])
        self._flatten()
        out = np.zeros(self.n, dtype=bool)
        out[-len(new_pts):] = True
        return out

    def add_cell(self, point: Sequence[float], stiffness: float = 1.0, group: int = 0) -> int:
        if not (MIN_STIFFNESS <= stiffness <= MAX_STIFFNESS):
            raise ValueError(f"stiffness must be between {MIN_STIFFNESS} and {MAX_STIFFNESS}")
        p = np.asarray(point, float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("a cell needs finite x, y, z")
        self._commit(f"add a cell at ({p[0]:.3g}, {p[1]:.3g}, {p[2]:.3g})")
        self.points = np.vstack([self.points, p])
        self.stiffness = np.append(self.stiffness, float(stiffness))
        self.group = np.append(self.group, int(group))
        self._flatten()
        return self.n - 1

    def add_random(self, count: int, lo: Sequence[float], hi: Sequence[float], stiffness: float = 1.0, seed: int = 0) -> int:
        if count < 1 or count > 100000:
            raise ValueError("count must be between 1 and 100000")
        rng = np.random.default_rng(seed)
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        pts = rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi), size=(int(count), 3))
        self._commit(f"add {int(count)} random cells in a box")
        self.points = np.vstack([self.points, pts])
        self.stiffness = np.concatenate([self.stiffness, np.full(int(count), float(stiffness))])
        self.group = np.concatenate([self.group, np.zeros(int(count), dtype=np.int64)])
        self._flatten()
        return int(count)

    def set_stiffness(self, sel, value: float) -> int:
        sel = self._need(sel, "set stiffness")
        if not (MIN_STIFFNESS <= value <= MAX_STIFFNESS):
            raise ValueError(f"stiffness must be between {MIN_STIFFNESS} and {MAX_STIFFNESS}")
        self._commit(f"set stiffness of {int(sel.sum())} cells to {value:g}")
        self.stiffness[sel] = float(value)
        return int(sel.sum())

    def scale_stiffness(self, sel, factor: float) -> int:
        sel = self._need(sel, "scale stiffness")
        if factor <= 0:
            raise ValueError("factor must be positive")
        self._commit(f"multiply stiffness of {int(sel.sum())} cells by {factor:g}")
        self.stiffness[sel] = np.clip(self.stiffness[sel] * factor, MIN_STIFFNESS, MAX_STIFFNESS)
        return int(sel.sum())

    def set_group(self, sel, g: int) -> int:
        sel = self._need(sel, "set group")
        self._commit(f"put {int(sel.sum())} cells in group {int(g)}")
        self.group[sel] = int(g)
        return int(sel.sum())

    def jitter(self, sel, amount: float, seed: int = 0) -> int:
        sel = self._need(sel, "jitter")
        if amount < 0:
            raise ValueError("amount must be >= 0")
        rng = np.random.default_rng(seed)
        self._commit(f"jitter {int(sel.sum())} cells by up to {amount:g}")
        self.points[sel] += rng.normal(0, amount, size=(int(sel.sum()), 3))
        self._flatten()
        return int(sel.sum())

    # ------------------------------------------------------------------ whole-model clean-up
    def find_outliers(self, k: int = 6, z: float = 3.0) -> np.ndarray:
        """Cells whose mean distance to their k nearest neighbours is more than z standard deviations above typical."""
        if self.n <= k + 1:
            return np.zeros(self.n, dtype=bool)
        d, _ = cKDTree(self.points).query(self.points, k=k + 1)
        m = d[:, 1:].mean(axis=1)
        sd = m.std()
        return m > m.mean() + z * sd if sd > 0 else np.zeros(self.n, dtype=bool)

    def merge_close(self, radius: float) -> int:
        """Keep one cell from every group of cells closer than `radius` (the first one, stiffness averaged)."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        tree = cKDTree(self.points)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            return 0
        parent = np.arange(self.n)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(i) for i in range(self.n)])
        keep = roots == np.arange(self.n)
        if int(keep.sum()) < 8:
            raise ValueError("merging would leave fewer than 8 cells; use a smaller radius")
        removed = int((~keep).sum())
        self._commit(f"merge cells closer than {radius:g} ({removed} removed)")
        st = self.stiffness.copy()
        for r in np.unique(roots[~keep]):
            st[r] = self.stiffness[roots == r].mean()
        self.points, self.stiffness, self.group = self.points[keep], st[keep], self.group[keep]
        return removed

    def downsample(self, n: int, seed: int = 0) -> int:
        if n < 8 or n >= self.n:
            raise ValueError(f"choose a number between 8 and {self.n - 1}")
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(self.n, size=int(n), replace=False))
        removed = self.n - int(n)
        self._commit(f"downsample to {int(n)} cells")
        self.points, self.stiffness, self.group = self.points[keep], self.stiffness[keep], self.group[keep]
        return removed

    def recenter(self) -> None:
        self._commit("move the model centre to the origin")
        self.points -= self.points.mean(axis=0)
        self._flatten()

    def normalize_extent(self, width: float = 10.0) -> None:
        """Scale uniformly so the x-extent equals `width` (the solver pulls a fraction of the x-width)."""
        if width <= 0:
            raise ValueError("width must be positive")
        lo, hi = self.bounds()
        span = float(hi[0] - lo[0])
        if span <= 0:
            raise ValueError("the model has no extent along x")
        self._commit(f"scale the model so the x-width is {width:g}")
        f = width / span
        self.points = (self.points - lo) * f + lo
        self._flatten()

    # ------------------------------------------------------------------ report
    def describe(self) -> Dict[str, object]:
        lo, hi = self.bounds()
        return {"name": self.name, "cells": self.n, "dim": self.dim,
                "min_xyz": [round(float(v), 3) for v in lo], "max_xyz": [round(float(v), 3) for v in hi],
                "stiffness_min": round(float(self.stiffness.min()), 3), "stiffness_max": round(float(self.stiffness.max()), 3),
                "groups": sorted(int(g) for g in np.unique(self.group)), "edits": len(self._undo)}

    def copy(self) -> "CellModel":
        return copy.deepcopy(self)
