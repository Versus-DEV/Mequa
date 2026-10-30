"""2D and 3D viewer: cells coloured by strain, zone, stiffness or failure probability."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go

from mequa import backend as backend_mod
from mequa import ingest
from mequa import numpy_backend as nb
from mequa.session import Session
from mequa.utils.env import load_env_file

load_env_file()
st.title("Cell viewer: 2D and 3D")
st.caption("The 3D path runs on the NumPy spring-network backend. Julia and Rust handle 2D uniform jobs (see the benchmark page).")


@st.cache_resource
def numpy_be():
    return backend_mod.load("numpy")


with st.sidebar:
    st.header("Data")
    source = st.selectbox(
        "Source",
        ["Synthetic 3D blob", "Synthetic 2D blob", "Demo z-stack (synthetic spheres)", "Upload z-stack images", "Upload CSV", "SAM cells (outputs/sam/cells.csv)"],
    )
    n_cells = st.slider("Cells", 200, 4000, 1200, 100) if source.startswith("Synthetic") else None
    z_spacing = st.number_input("Slice spacing (pixel units)", 0.1, 20.0, 1.0, 0.1) if "z-stack" in source else 1.0
    seg_method = st.selectbox("Segmentation", ["classical", "sam"]) if "z-stack" in source else "classical"
    split = st.checkbox("Split touching cells (watershed)", value=True) if "z-stack" in source else False
    uploads = st.file_uploader("Slices (PNG/JPG), in order", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True) if source == "Upload z-stack images" else None
    csv_up = st.file_uploader("CSV with header x,y[,z][,area][,stiffness]", type=["csv"]) if source == "Upload CSV" else None
    st.header("Physics")
    load = st.slider("Load (fraction of width pulled)", 0.01, 0.12, 0.05, 0.01)
    k = st.slider("Density zones", 2, 8, 3)
    knn = st.slider("Neighbors per cell", 4, 14, 8)
    st.header("View")
    color_by = st.selectbox("Colour by", ["strain", "zone", "failure probability", "stiffness"])
    show_links = st.checkbox("Show mechanical links", value=False)
    build = st.button("Load and analyze", type="primary")


def get_data():
    stiff = None
    if source == "Synthetic 3D blob":
        return nb.synthetic_cells(n_cells, seed=5, clusters=1, dim=3), None, "synthetic 3D blob"
    if source == "Synthetic 2D blob":
        return nb.synthetic_cells(n_cells, seed=5, clusters=1, dim=2), None, "synthetic 2D blob"
    if source == "Demo z-stack (synthetic spheres)":
        imgs, _ = ingest.make_demo_stack(n_cells=60, n_slices=12, size=256)
        pts, areas = ingest.segment_stack(imgs, "classical", z_spacing=z_spacing, split=split, min_distance=5)
        return pts, nb.stiffness_from_areas(areas), "synthetic z-stack, classical segmentation"
    if source == "Upload z-stack images":
        from mequa.vision.image_io import decode_image_bytes, decode_stack_bytes

        if not uploads:
            raise ValueError("Upload slice images, or one multi-page TIFF.")
        imgs = decode_stack_bytes(uploads[0].getvalue()) if len(uploads) == 1 else [decode_image_bytes(u.getvalue()) for u in uploads]
        if len(imgs) < 2:
            raise ValueError("A z-stack needs at least two slices.")
        pts, areas = ingest.segment_stack(imgs, seg_method, z_spacing=z_spacing, split=split, min_distance=5)
        return pts, nb.stiffness_from_areas(areas), f"{len(imgs)} slices, {seg_method} segmentation"
    if source == "Upload CSV":
        if csv_up is None:
            raise ValueError("Upload a CSV first.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_up.getvalue())
        pts, areas, stiff = ingest.load_cells_full(tmp.name)
        if stiff is None and areas is not None:
            stiff = nb.stiffness_from_areas(areas)
        return pts, stiff, "uploaded CSV"
    path = ROOT / "outputs" / "sam" / "cells.csv"
    if not path.exists():
        raise ValueError("No SAM cells yet. Run the SAM Mechanics page on a real image first.")
    pts, areas, stiff = ingest.load_cells_full(path)
    return pts, stiff, "SAM cells"


if build:
    try:
        pts, stiff, label = get_data()
        st.session_state["viewer_data"] = (pts, stiff, label)
    except Exception as exc:
        st.error(str(exc))

data = st.session_state.get("viewer_data")
if data is None:
    st.info("Pick a source and click Load and analyze.")
    st.stop()

pts, stiff, label = data
sess = Session(pts, stiffness=stiff, backend=numpy_be(), k=k, knn=knn)
with st.spinner("Relaxing the spring network..."):
    summary, res = sess.run(load, k=k, knn=knn, full=True)

st.caption(f"Data: {label}. {summary['cells']} cells, {summary['dim']}D, solver: {summary['backend_used']}.")
c = st.columns(6)
c[0].metric("Links", summary["mechanical_links"])
c[1].metric("SPHI", f"{summary['sphi']:.3f}")
c[2].metric("Mean strain", f"{summary['mean_strain']:.4f}")
c[3].metric("p95 strain", f"{summary['p95_strain']:.4f}")
c[4].metric("Max strain", f"{summary['max_strain']:.4f}")
c[5].metric("Residual", f"{summary['residual']:.1e}" if summary["residual"] is not None else "n/a")
if summary["load_path"] is False:
    st.warning(
        f"The mesh has {summary['n_components']} disconnected pieces and the pulled edge is not connected to the held edge, "
        "so the strain shown is a decaying transient, not a loaded equilibrium. Raise the neighbor count."
    )

values = {
    "strain": res["node_strain"],
    "zone": res["labels"],
    "failure probability": res["failure_prob"],
    "stiffness": stiff if stiff is not None else np.ones(len(pts)),
}[color_by]
scale = "Turbo" if color_by != "zone" else "Viridis"

if show_links:
    ei, ej = nb.build_knn_edges(pts, knn)
    if len(ei) > 3000:
        sel = np.random.default_rng(0).choice(len(ei), 3000, replace=False)
        ei, ej = ei[sel], ej[sel]


def link_trace(dims):
    xs, ys, zs = [], [], []
    for a, b in zip(ei, ej):
        xs += [pts[a, 0], pts[b, 0], None]
        ys += [pts[a, 1], pts[b, 1], None]
        if dims == 3:
            zs += [pts[a, 2], pts[b, 2], None]
    kw = dict(x=xs, y=ys, mode="lines", line=dict(width=1, color="rgba(150,150,150,0.35)"), hoverinfo="skip", showlegend=False)
    return go.Scatter3d(z=zs, **kw) if dims == 3 else go.Scatter(**kw)


tab3d, tab2d, tabtable = st.tabs(["3D view" if pts.shape[1] == 3 else "3D (needs z data)", "2D view / slice", "Zones"])

with tab3d:
    if pts.shape[1] != 3:
        st.info("This dataset is 2D. Choose a 3D source (synthetic 3D blob, z-stack, or a CSV with a z column).")
    else:
        fig = go.Figure()
        if show_links:
            fig.add_trace(link_trace(3))
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=3.5, color=values, colorscale=scale, colorbar=dict(title=color_by), opacity=0.9),
            hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<br>" + color_by + "=%{marker.color:.4f}<extra></extra>",
        ))
        fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0), scene=dict(aspectmode="data"))
        st.plotly_chart(fig, use_container_width=True)

with tab2d:
    if pts.shape[1] == 3:
        zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
        lo, hi = st.slider("z slab", zmin, zmax, (zmin, zmax), step=max((zmax - zmin) / 100, 1e-3))
        keep = (pts[:, 2] >= lo) & (pts[:, 2] <= hi)
        st.caption(f"{int(keep.sum())} cells in the slab, projected onto x and y.")
    else:
        keep = np.ones(len(pts), dtype=bool)
    fig2 = go.Figure()
    if show_links and pts.shape[1] == 2:
        fig2.add_trace(link_trace(2))
    fig2.add_trace(go.Scatter(
        x=pts[keep, 0], y=pts[keep, 1], mode="markers",
        marker=dict(size=6, color=np.asarray(values)[keep], colorscale=scale, colorbar=dict(title=color_by)),
    ))
    fig2.update_yaxes(scaleanchor="x")
    fig2.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

with tabtable:
    st.dataframe(summary["per_zone"], use_container_width=True)
