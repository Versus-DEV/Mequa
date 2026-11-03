"""The dataset every page shares.

Pages used to keep their own copies of "the current cells" and one of them (`result`) was even
shared by accident, which crashed the Workbench when the SAM page had written a different kind of
result under the same key. Now there is one explicit, typed slot for the working dataset and each
page keeps its own results under its own keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_KEY = "workspace_dataset"


@dataclass(frozen=True)
class Dataset:
    points: np.ndarray                      # (n, 2) or (n, 3), float64
    stiffness: Optional[np.ndarray]         # (n,) or None for uniform
    label: str                              # human readable provenance
    source: str                             # which page published it

    @property
    def n(self) -> int:
        return int(len(self.points))

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])


def make_dataset(points, stiffness=None, label="", source="") -> Dataset:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"points must be (n, 2) or (n, 3), got {pts.shape}")
    st = None
    if stiffness is not None:
        st = np.ascontiguousarray(stiffness, dtype=np.float64)
        if st.shape != (len(pts),):
            raise ValueError(f"stiffness must have one value per cell ({len(pts)}), got {st.shape}")
    return Dataset(pts, st, label or "unnamed dataset", source)


def publish(state, points, stiffness=None, label="", source="") -> Dataset:
    """Store the working dataset in a mapping (st.session_state, or a dict in tests)."""
    ds = make_dataset(points, stiffness, label, source)
    state[_KEY] = ds
    return ds


def current(state) -> Optional[Dataset]:
    ds = state.get(_KEY)
    return ds if isinstance(ds, Dataset) else None
