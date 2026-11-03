"""Plotly figures for the model editor (three orthographic CAD views plus one perspective view) and
the reading of box/lasso selection events. No Streamlit import, so it is unit-tested."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from mequa.editor.model import CellModel
from mequa.ui import theme

GROUP_COLORS = ["#8A8A8A", "#5B8DEF", "#3BB3C3", "#D9A441", "#9B7FD1", "#4CAF7A", "#D77FA8", "#8B9099"]
COLOR_MODES = ["selection", "stiffness", "group", "strain"]
VIEWS = {"top": ("x", "y"), "front": ("x", "z"), "side": ("y", "z")}
_AX = {"x": 0, "y": 1, "z": 2}
_BASE = "#6b7280"


def marker_for(m: CellModel, mode: str, strain: Optional[np.ndarray] = None, size: float = 5.0) -> Dict[str, Any]:
    mk: Dict[str, Any] = dict(size=size, opacity=0.9)
    if mode == "stiffness":
        mk.update(color=m.stiffness, colorscale="Viridis", showscale=False)
    elif mode == "group":
        mk.update(color=[GROUP_COLORS[int(g) % len(GROUP_COLORS)] for g in m.group])
    elif mode == "strain" and strain is not None and len(strain) == m.n:
        mk.update(color=strain, colorscale="Turbo", showscale=False)
    else:
        mk.update(color=_BASE)
    return mk


def _overlay(m: CellModel, sel: np.ndarray, axes, size: float):
    idx = np.flatnonzero(sel)
    cols = [_AX[a] for a in axes]
    kw = dict(x=m.points[idx, cols[0]], y=m.points[idx, cols[1]])
    return kw, idx


def figure_2d(m: CellModel, sel: np.ndarray, view: str, mode: str = "selection", strain: Optional[np.ndarray] = None, height: int = 330) -> go.Figure:
    a, b = VIEWS[view]
    ia, ib = _AX[a], _AX[b]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=m.points[:, ia], y=m.points[:, ib], mode="markers", marker=marker_for(m, mode, strain), name="cells",
        customdata=np.arange(m.n), hovertemplate="cell %{customdata}<br>" + a + "=%{x:.3f} " + b + "=%{y:.3f}<extra></extra>", showlegend=False))
    if sel.any():
        kw, idx = _overlay(m, sel, (a, b), 8)
        fig.add_trace(go.Scattergl(**kw, mode="markers", name="selected", hoverinfo="skip", showlegend=False,
                                   marker=dict(size=10, color=theme.rgba(theme.accent(), 0.35), line=dict(color=theme.ink(), width=1.4))))
    fig.update_layout(dragmode="select", height=height, margin=dict(l=36, r=10, t=8, b=30), clickmode="event+select",
                      xaxis_title=a, yaxis_title=b, uirevision=f"{view}")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def figure_3d(m: CellModel, sel: np.ndarray, mode: str = "selection", strain: Optional[np.ndarray] = None, height: int = 330) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=m.points[:, 0], y=m.points[:, 1], z=m.points[:, 2], mode="markers", name="cells",
                               marker={**marker_for(m, mode, strain, size=2.6), "opacity": 0.85}, customdata=np.arange(m.n),
                               hovertemplate="cell %{customdata}<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>"))
    if sel.any():
        p = m.points[sel]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name="selected", hoverinfo="skip",
                                   marker=dict(size=4.2, color=theme.accent(), opacity=1.0)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision="persp",
                      scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return fig


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        return getattr(obj, key, default)


def picked_from_event(event: Any) -> List[int]:
    """Cell ids picked by a box or lasso on the 'cells' trace (curve 0) of a 2D view."""
    sel = _get(event, "selection")
    pts = _get(sel, "points") or []
    out = []
    for p in pts:
        if _get(p, "curve_number", 0) == 0:
            cid = _get(p, "customdata")
            if isinstance(cid, (list, tuple)) and cid:
                cid = cid[0]
            if cid is None:
                cid = _get(p, "point_index", _get(p, "point_number"))
            if cid is not None:
                out.append(int(cid))
    if not out and not pts:
        # older payloads only carry point_indices (single trace)
        out = [int(i) for i in (_get(sel, "point_indices") or [])]
    return sorted(set(out))
