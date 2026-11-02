"""3D Viewer: orbit, box-select, clip, measure and animate cell models and molecules (PDB, XYZ)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import numpy_backend as nb  # noqa: E402
from mequa import ui, viewer, workspace  # noqa: E402
from mequa.session import Session  # noqa: E402
from mequa.viewer import pdb  # noqa: E402

ui.page_header("3D Viewer", "WebGL viewer for cell models and molecules. Orbit, drag a box to select, clip, measure, animate the load.", eyebrow="Analysis")


def demo_pdb() -> bytes:
    """A small two-chain helix of C-alpha atoms so the molecule path can be tried with no file."""
    lines, i = [], 1
    for ch, shift in (("A", 0.0), ("B", 9.0)):
        for r in range(1, 41):
            t = r * 100 * np.pi / 180
            x, y, z = 2.3 * np.cos(t) + shift, 2.3 * np.sin(t), 1.5 * r
            lines.append(f"ATOM  {i:5d}  CA  ALA {ch}{r:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
            i += 1
    return ("\n".join(lines) + "\nEND\n").encode()


with st.sidebar:
    st.header("Source")
    sources = ["Workspace dataset", "Synthetic 3D blob", "Molecule (PDB / XYZ)", "Demo molecule"]
    source = st.selectbox("Show", sources, index=0 if workspace.current(st.session_state) else 1)
    n_cells = st.slider("Cells", 200, 4000, 1200, 100) if source == "Synthetic 3D blob" else 0
    up = st.file_uploader("PDB or XYZ file", type=["pdb", "ent", "xyz"]) if source == "Molecule (PDB / XYZ)" else None
    st.header("Appearance")
    rep = st.selectbox("Representation", ["spheres", "ball+stick", "sticks", "points"], format_func=ui.nice_name)
    do_solve = source in ("Workspace dataset", "Synthetic 3D blob")
    if do_solve:
        st.header("Physics")
        load = st.slider("Load (fraction of width pulled)", 0.01, 0.12, 0.05, 0.01)
        k = st.slider("Density zones", 2, 8, 3)
        knn = st.slider("Neighbors per cell", 4, 14, 8)

scene, note = None, ""
try:
    if source == "Workspace dataset":
        ds = workspace.current(st.session_state)
        if ds is None:
            st.info("No dataset in the workspace yet. Build one in Workbench, Cell Viewer or Model Editor, or pick Synthetic 3D blob.")
        else:
            pts, stiff, note = ds.points, ds.stiffness, ds.label
    elif source == "Synthetic 3D blob":
        pts, stiff = nb.synthetic_cells(n_cells, seed=5, clusters=1, dim=3), None
        note = "synthetic 3D blob"
    else:
        data, name = (demo_pdb(), "demo.pdb") if source == "Demo molecule" else ((up.getvalue(), up.name) if up else (None, ""))
        if data is None:
            st.info("Upload a .pdb or .xyz file to view it.")
        else:
            mol = pdb.load(data, name)
            scene = viewer.scene_from_molecule(mol)
            note = f"{name}: {mol['xyz'].shape[0]} atoms, chains {', '.join(sorted(set(mol['chain'])))}"
    if scene is None and do_solve and "pts" in locals():
        with st.spinner("Solving…"):
            summary, res = Session(pts, stiffness=stiff, knn=knn).run(load, k=k, full=True)
        scene = viewer.scene_from_solve(pts, res, stiffness=stiff, knn=min(knn, 6), title=note)
        ui.big_number(f"{summary.get('p95_strain', summary.get('strain_p95', 0)):.3f}" if isinstance(summary, dict) else "", "p95 strain", f"{len(pts)} cells")
except Exception as exc:
    st.error(f"Could not build the scene: {exc}")

if scene is not None:
    color = "strain" if "strain" in scene["fields"] else ("chain" if "chain" in scene["fields"] else ("element" if "colors" in scene else "uniform"))
    sel = viewer.viewer(scene, rep=rep, color=color, height=640, key="viewer_" + source.split()[0].lower())
    st.caption(note + " · Drag to orbit, scroll to zoom. Colour, clip, measure and animation controls are inside the viewer.")
    if sel and sel.get("ids"):
        st.markdown(f"**{len(sel['ids'])} selected**")
        if "labels" in scene:
            st.write(", ".join(scene["labels"][i] for i in sel["ids"][:30]) + (" …" if len(sel["ids"]) > 30 else ""))
