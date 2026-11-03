"""Mequa 3D viewer: a WebGL component (orbit, 3D box select, measure, clip, animation) plus the code
that turns cell models and molecular files into scenes for it."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

WEB = Path(__file__).resolve().parent / "web"
MAX_SCENE_CELLS = 60_000
_component = None


def _flat(a: np.ndarray, nd: int = 6) -> list:
    return np.round(np.asarray(a, dtype=np.float64), nd).ravel().tolist()


def knn_edges(points: np.ndarray, k: int = 6, max_edges: int = 40_000) -> np.ndarray:
    from mequa import numpy_backend as nb

    ei, ej = nb.build_knn_edges(np.asarray(points, dtype=np.float64), min(k, max(1, len(points) - 1)))
    e = np.column_stack([ei, ej]).astype(np.int64)
    if len(e) > max_edges:
        e = e[np.random.default_rng(0).choice(len(e), max_edges, replace=False)]
    return e


def build_scene(points: np.ndarray, *, fields: Optional[Dict[str, Sequence[float]]] = None, categorical: Sequence[str] = (),
                edges: Optional[np.ndarray] = None, edge_strain: Optional[Sequence[float]] = None, loaded: Optional[np.ndarray] = None,
                colors: Optional[Sequence[str]] = None, radii: Optional[Sequence[float]] = None, labels: Optional[Sequence[str]] = None,
                title: str = "model") -> Dict[str, Any]:
    """Everything the viewer needs, as JSON-safe lists. 2D points are lifted to z = 0."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError("points must be (n, 2) or (n, 3)")
    if len(pts) > MAX_SCENE_CELLS:
        raise ValueError(f"the viewer shows up to {MAX_SCENE_CELLS} cells; downsample first (Model Editor, Clean-up)")
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    n = len(pts)
    sc: Dict[str, Any] = {"n": n, "pos": _flat(pts), "title": title, "fields": {}, "field_kinds": {}}
    for name, vals in (fields or {}).items():
        v = np.asarray(vals, dtype=np.float64)
        if v.shape != (n,):
            raise ValueError(f"field {name!r} must have one value per cell")
        sc["fields"][name] = np.round(v, 6).tolist()
        sc["field_kinds"][name] = "categorical" if name in categorical else "continuous"
    if edges is not None and len(edges):
        sc["edges"] = np.asarray(edges, dtype=np.int64).ravel().tolist()
        if edge_strain is not None:
            es = np.asarray(edge_strain, dtype=np.float64)
            if es.shape != (len(edges),):
                raise ValueError("edge_strain must have one value per edge")
            sc["edge_strain"] = np.round(es, 6).tolist()
    if loaded is not None:
        ld = np.asarray(loaded, dtype=np.float64)
        if ld.shape[1] == 2:
            ld = np.column_stack([ld, np.zeros(len(ld))])
        if ld.shape != pts.shape:
            raise ValueError("loaded positions must match the points")
        sc["loaded"] = _flat(ld)
    if colors is not None:
        sc["colors"] = list(colors)
    if radii is not None:
        sc["radii"] = np.round(np.asarray(radii, float), 4).tolist()
    if labels is not None:
        sc["labels"] = list(labels)
    return sc


def scene_from_solve(points, res: Dict[str, Any], *, stiffness=None, group=None, knn: int = 6, title: str = "model") -> Dict[str, Any]:
    """A cell scene with the solver output: strain, zone, failure probability, stiffness, links and the loaded shape."""
    from mequa import numpy_backend as nb

    pts = np.asarray(points, dtype=np.float64)
    fields = {"strain": res["node_strain"], "zone": res["labels"], "failure_prob": res["failure_prob"]}
    if stiffness is not None:
        fields["stiffness"] = stiffness
    if group is not None:
        fields["group"] = group
    cats = [k for k in ("zone", "group") if k in fields]
    ei, ej = nb.build_knn_edges(pts, knn)
    e = np.column_stack([ei, ej])
    loaded = res.get("positions")
    es = None
    if loaded is not None:
        ld = np.asarray(loaded, float)
        rest_len = np.maximum(np.linalg.norm(pts[ei] - pts[ej], axis=1), 1e-9)
        es = np.abs(np.linalg.norm(ld[ei] - ld[ej], axis=1) - rest_len) / rest_len
    if len(e) > 40_000:
        keep = np.random.default_rng(0).choice(len(e), 40_000, replace=False)
        e, es = e[keep], (None if es is None else es[keep])
    return build_scene(pts, fields=fields, categorical=cats, edges=e, edge_strain=es, loaded=None if loaded is None else np.asarray(loaded, float)[:, :pts.shape[1]],
                       title=title)


def scene_from_molecule(mol: Dict[str, Any]) -> Dict[str, Any]:
    from mequa.viewer import pdb

    bonds = mol["bonds"]
    bb = pdb.backbone_edges(mol)
    edges = np.vstack([bonds, bb]) if len(bb) else bonds
    chain_ids = {c: i for i, c in enumerate(sorted(set(mol["chain"])))}
    return build_scene(mol["xyz"], fields={"chain": [chain_ids[c] for c in mol["chain"]], "residue": mol["resid"]}, categorical=["chain"],
                       edges=edges, colors=pdb.element_colors(mol), radii=pdb.radii(mol), labels=mol["labels"], title=mol.get("name", "molecule"))


def scene_id(scene: Dict[str, Any]) -> str:
    h = hashlib.md5()
    h.update(str(scene["n"]).encode())
    for k in ("pos", "loaded", "edges"):
        if k in scene:
            h.update(np.asarray(scene[k], dtype=np.float64).tobytes() if k != "edges" else np.asarray(scene[k], dtype=np.int64).tobytes())
    for k in sorted(scene["fields"]):
        h.update(k.encode())
        h.update(np.asarray(scene["fields"][k], dtype=np.float64).tobytes())
    if "colors" in scene:
        h.update("".join(scene["colors"]).encode())
    return h.hexdigest()[:16]


def viewer(scene: Dict[str, Any], *, selected: Optional[Sequence[int]] = None, sel_rev: int = 0, key: str = "mequa_viewer", height: int = 560,
           rep: str = "spheres", color: str = "uniform", refit: bool = False, theme: Optional[str] = None):
    """Show the viewer. Returns None until the user selects, then {"ids": [...], "nonce": n, "source": "viewer"} (the full selection)."""
    global _component
    import streamlit.components.v1 as components

    if _component is None:
        _component = components.declare_component("mequa_viewer", path=str(WEB))
    if theme is None:
        try:
            from mequa.ui import theme as _t
            theme = _t.current_mode()
        except Exception:
            theme = "light"
    return _component(scene=scene, theme=theme, scene_id=scene_id(scene), selected=[int(i) for i in (selected or [])], sel_rev=int(sel_rev), height=int(height),
                      rep=rep, color=color, refit=bool(refit), key=key, default=None)
