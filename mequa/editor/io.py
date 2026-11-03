"""Read and write edited models: CSV (round-trips with the rest of Mequa), JSON, OBJ and PLY point clouds."""

from __future__ import annotations

import io
import json
from typing import Tuple

import numpy as np

from mequa.editor.model import CellModel


def _csv_text(model: CellModel) -> str:
    cols = ["x", "y"] + (["z"] if model.dim == 3 else []) + ["stiffness", "group"]
    data = np.column_stack([model.points[:, : model.dim], model.stiffness, model.group])
    buf = io.StringIO()
    buf.write(",".join(cols) + "\n")
    np.savetxt(buf, data, delimiter=",", fmt=["%.6f"] * (model.dim + 1) + ["%d"])
    return buf.getvalue()


def to_csv(model: CellModel) -> bytes:
    return _csv_text(model).encode("utf-8")


def to_json(model: CellModel) -> bytes:
    return json.dumps({
        "format": "mequa-cell-model", "version": 1, "name": model.name, "dim": model.dim,
        "points": np.round(model.points[:, : model.dim], 6).tolist(),
        "stiffness": np.round(model.stiffness, 6).tolist(), "group": model.group.tolist(),
    }).encode("utf-8")


def to_obj(model: CellModel) -> bytes:
    lines = [f"# Mequa cell model '{model.name}', {model.n} cells (vertices only)", f"o {model.name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in model.points]
    return ("\n".join(lines) + "\n").encode("utf-8")


def to_ply(model: CellModel) -> bytes:
    head = ["ply", "format ascii 1.0", f"comment Mequa cell model {model.name}", f"element vertex {model.n}",
            "property float x", "property float y", "property float z", "property float stiffness", "property int group", "end_header"]
    rows = [f"{x:.6f} {y:.6f} {z:.6f} {s:.6f} {int(g)}" for (x, y, z), s, g in zip(model.points, model.stiffness, model.group)]
    return ("\n".join(head + rows) + "\n").encode("utf-8")


def from_bytes(data: bytes, filename: str, name: str = "") -> CellModel:
    """Build a model from an uploaded file. The extension picks the reader."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    name = name or filename.rsplit(".", 1)[0]
    if ext == "json":
        obj = json.loads(data.decode("utf-8"))
        if not isinstance(obj, dict) or "points" not in obj:
            raise ValueError("JSON must be a Mequa cell model with a 'points' list")
        return CellModel.from_arrays(obj["points"], obj.get("stiffness"), obj.get("group"), name=obj.get("name", name))
    if ext == "npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return CellModel.from_arrays(arr, name=name)
    if ext in ("csv", "txt", ""):
        import pandas as pd

        df = pd.read_csv(io.BytesIO(data))
        cols = {c.strip().lower(): c for c in df.columns}
        if "x" not in cols or "y" not in cols:
            raise ValueError("CSV needs columns named x and y (and optionally z, stiffness, group)")
        axes = [cols["x"], cols["y"]] + ([cols["z"]] if "z" in cols else [])
        pts = df[axes].to_numpy(dtype=np.float64)
        st = df[cols["stiffness"]].to_numpy(dtype=np.float64) if "stiffness" in cols else None
        gr = df[cols["group"]].to_numpy(dtype=np.int64) if "group" in cols else None
        return CellModel.from_arrays(pts, st, gr, name=name)
    raise ValueError(f"cannot read .{ext} files; use CSV, JSON or NPY")


EXPORTERS = {"CSV (x,y,z,stiffness,group)": ("csv", to_csv, "text/csv"),
             "JSON (full model)": ("json", to_json, "application/json"),
             "OBJ (point cloud)": ("obj", to_obj, "text/plain"),
             "PLY (point cloud + stiffness)": ("ply", to_ply, "text/plain")}


def export(model: CellModel, label: str) -> Tuple[str, bytes, str]:
    ext, fn, mime = EXPORTERS[label]
    return f"{model.name}.{ext}", fn(model), mime
