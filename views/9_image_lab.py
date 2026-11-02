"""Image Lab: upload a real microscopy image or z-stack, find the nuclei, and use them everywhere else in the app."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import imagelab as il  # noqa: E402
from mequa import ui, workspace  # noqa: E402
from mequa.session import Session  # noqa: E402
from mequa.ui import theme  # noqa: E402

ss = st.session_state
ui.page_header("Image Lab", "Upload your own microscopy image or z-stack. Mequa finds the nuclei, builds the spring network from them and sends the cells to the viewer, editor and chat.",
               eyebrow="Analysis")

with st.sidebar:
    st.header("Image")
    source = st.radio("Source", ["Upload my image", "Synthetic test stack"], label_visibility="collapsed",
                      help="The synthetic stack is made up. It exists only to check that the pipeline runs.")
    files = []
    if source == "Upload my image":
        files = st.file_uploader("TIFF, PNG, JPG, multi-page TIFF, or a zip of slices", type=["tif", "tiff", "png", "jpg", "jpeg", "zip"],
                                 accept_multiple_files=True)
    st.header("Detection")
    bright = st.checkbox("Nuclei are brighter than background", value=True, help="Untick for dark nuclei on a light background.")
    sigma = st.slider("Smoothing (pixels)", 0.5, 5.0, 2.0, 0.5, help="Larger values merge speckle inside a nucleus; too large blurs neighbours together.")
    z_scale = st.number_input("z spacing relative to x/y pixel size", 0.1, 10.0, 1.0, 0.1, help="Only used for 3D stacks.")
    min_area = st.number_input("Smallest nucleus (pixels, 2D)", 3, 5000, 30, 5)
    st.header("Load")
    load = st.slider("Load (fraction of width pulled)", 0.01, 0.12, 0.05, 0.01)
    knn = st.slider("Neighbors per cell", 4, 14, 8)

# ---- read the file(s) -------------------------------------------------------------------------------
sig = (source, tuple((f.name, f.size) for f in files)) if source == "Upload my image" else (source,)
if ss.get("lab_sig") != sig:
    ss["lab_sig"], ss["lab_arr"], ss["lab_det"], ss["lab_name"] = sig, None, None, ""
    try:
        if source == "Synthetic test stack":
            ss["lab_arr"], ss["lab_name"] = il.make_synthetic_stack(), "synthetic test stack (made up)"
        elif files:
            with st.spinner("Reading the file…"):
                ss["lab_arr"] = il.combine_slices([(f.name, f.getvalue()) for f in files])
            ss["lab_name"] = files[0].name if len(files) == 1 else f"{len(files)} slices"
    except Exception as exc:
        st.error(str(exc))

arr = ss.get("lab_arr")
if arr is None:
    st.info("Upload an image or stack in the sidebar. Real data is read as it is: nothing is preloaded.")
    st.stop()

info = il.describe(arr)
default_f = il.suggest_downsample(arr)
c = st.columns(4)
c[0].metric("Type", info["kind"])
c[1].metric("Shape", " × ".join(str(s) for s in info["shape"]))
c[2].metric("Intensity range", f"{info['min']:.0f} – {info['max']:.0f}")
c[3].metric("Source", ss["lab_name"] if len(ss["lab_name"]) < 26 else ss["lab_name"][:23] + "…")
if "synthetic" in ss["lab_name"]:
    st.warning("This is a made-up test stack, not real data. Upload your own image to analyse it.")

f = st.select_slider("Reduce x and y before detection", options=[1, 2, 4, 8], value=default_f,
                     help="Speeds up large stacks. Detected positions are scaled back to the original image. 1 = full resolution.")
if f != default_f and default_f > 1:
    st.caption(f"Suggested for this size: {default_f}.")

run = st.button("Find cells", type="primary")
if run:
    status = st.status("Detecting nuclei…", expanded=True)
    try:
        t0 = time.time()
        det = il.detect(arr, sigma=sigma, bright=bright, z_scale=z_scale, downsample_by=f, min_area=int(min_area),
                        progress=lambda m: status.write(f"{m}  ({time.time() - t0:.0f}s)"))
        if det.n < 3:
            status.update(label="Too few cells found", state="error")
            st.error(f"Only {det.n} nuclei were found. Try changing the brightness setting, lowering the smoothing, or a smaller reduction factor.")
        else:
            stiff = il.stiffness_from_detection(det)
            label = f"Image Lab: {ss['lab_name']}, {det.n} cells"
            workspace.publish(ss, det.points, stiff, label=label, source="Image Lab")
            ss["lab_det"] = det
            status.update(label=f"Found {det.n} nuclei in {det.seconds:.1f}s", state="complete", expanded=False)
    except Exception as exc:
        status.update(label="Detection failed", state="error")
        st.error(str(exc))

det = ss.get("lab_det")

# ---- picture with the detections drawn on it -------------------------------------------------------
st.markdown('<div class="mq-view-label">Image and detected cells</div>', unsafe_allow_html=True)
z_idx = None
if arr.ndim == 3:
    z_idx = st.slider("Slice", 0, arr.shape[0] - 1, arr.shape[0] // 2)
g = il.preview_slice(arr, z_idx)
scale = max(arr.shape[-2:]) / max(g.shape)           # preview pixel -> original pixel
fig = go.Figure(go.Heatmap(z=g, colorscale="gray", showscale=False, hoverinfo="skip",
                           x0=0, dx=scale, y0=0, dy=scale))
if det is not None and det.n:
    p = det.points
    if det.dim == 3 and z_idx is not None:
        zc = p[:, 2] / max(z_scale, 1e-6)
        near = np.abs(zc - z_idx) <= 3
        far = ~near
        fig.add_trace(go.Scattergl(x=p[far, 0], y=p[far, 1], mode="markers", name="other slices", hoverinfo="skip",
                                   marker=dict(size=7, color="rgba(0,0,0,0)", line=dict(color=theme.rgba(theme.accent(), 0.45), width=1))))
        fig.add_trace(go.Scattergl(x=p[near, 0], y=p[near, 1], mode="markers", name="this slice (±3)",
                                   marker=dict(size=11, color="rgba(0,0,0,0)", line=dict(color=theme.accent(), width=2))))
    else:
        fig.add_trace(go.Scattergl(x=p[:, 0], y=p[:, 1], mode="markers", name="cells",
                                   marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color=theme.accent(), width=2))))
fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), showlegend=det is not None, legend=dict(orientation="h", y=1.04, font=dict(color=theme.ink()), bgcolor="rgba(0,0,0,0)"),
                  xaxis=dict(visible=False, range=[0, arr.shape[-1]]), yaxis=dict(visible=False, range=[arr.shape[-2], 0], scaleanchor="x"))
st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

if det is None:
    st.info("Click **Find cells**. The result becomes the workspace dataset, so the 3D Viewer, Model Editor, Workbench and chat all use it.")
    st.stop()

# ---- results and mechanics -------------------------------------------------------------------------
for n in det.notes:
    st.caption(n)
c = st.columns(4)
c[0].metric("Cells found", det.n)
c[1].metric("Detection time", f"{det.seconds:.1f} s")
c[2].metric("Dimensions", f"{det.dim}D")
c[3].metric("Median cell area" if det.areas is not None else "Reduced by", f"{np.median(det.areas):.0f} px" if det.areas is not None else f"{det.downsample}×")

stiff = il.stiffness_from_detection(det)
with st.spinner("Relaxing the spring network…"):
    summary, res = Session(det.points, stiffness=stiff, knn=knn).run(load, k=3, knn=knn, full=True)
ui.big_number(f"{summary['p95_strain']:.4f}", "p95 strain at the chosen load", f"{det.n} cells")
m = st.columns(4)
m[0].metric("SPHI", f"{summary['sphi']:.3f}")
m[1].metric("Mean strain", f"{summary['mean_strain']:.4f}")
m[2].metric("Max strain", f"{summary['max_strain']:.4f}")
m[3].metric("Connected pieces", summary["n_components"])
if summary["load_path"] is False:
    st.warning(f"The cells form {summary['n_components']} disconnected pieces and the pulled edge is not connected to the held edge, so the strain is not a loaded equilibrium. Raise the neighbor count.")

import io  # noqa: E402

buf = io.StringIO()
header = "x,y,z" if det.dim == 3 else "x,y"
buf.write(header + "\n")
for row in det.points:
    buf.write(",".join(f"{v:.3f}" for v in row) + "\n")
st.download_button("Download cells (CSV)", buf.getvalue(), file_name="mequa_cells.csv", mime="text/csv")
st.caption("These cells are now the workspace dataset. Open **3D Viewer** to orbit and select them, **Model Editor** to edit them, or ask the chat about them.")
