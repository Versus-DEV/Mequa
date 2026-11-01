"""3D Model Editor: edit a cell model in three orthographic views plus a perspective view.

Box- or lasso-select cells in any of the Top / Front / Side views (like the plan, elevation and
section windows of a CAD tool), then move, rotate, scale, mirror, duplicate, delete, set stiffness
or group. Every edit is undoable. "Solve and compare" runs the strain solver on the edited model
and on the model as loaded, so the effect of an edit is a number, not a guess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import numpy_backend as nb
from mequa import ui, workspace
from mequa.editor import CellModel, apply_selection
from mequa.editor import io as eio
from mequa.editor import views
from mequa.session import Session
from mequa.utils.env import load_env_file

load_env_file()
ss = st.session_state

ui.page_header("Model Editor", "Select cells in the Top, Front and Side views, edit them, and re-solve. Every edit can be undone.", eyebrow="3D Model")


# ------------------------------------------------------------------------------------ state helpers
def model() -> CellModel | None:
    return ss.get("ed_model")


def set_model(m: CellModel) -> None:
    ss["ed_model"] = m
    ss["ed_sel"] = np.zeros(m.n, dtype=bool)
    ss["ed_rev"] = 0
    ss["ed_solve"] = None
    ss["ed_flash"] = f"Loaded '{m.name}': {m.n} cells, {m.dim}D."
    for v in views.VIEWS:
        ss[f"ed_sig_{v}"] = None


def bump(reset_selection: bool = False, select: np.ndarray | None = None, flash: str = "") -> None:
    """Call after an edit: invalidates chart selections and any cached solve."""
    m = model()
    ss["ed_rev"] = ss.get("ed_rev", 0) + 1
    ss["ed_solve"] = None
    if select is not None:
        ss["ed_sel"] = select
    elif reset_selection or ss["ed_sel"].shape != (m.n,):
        ss["ed_sel"] = np.zeros(m.n, dtype=bool)
    for v in views.VIEWS:
        ss[f"ed_sig_{v}"] = None
    if flash:
        ss["ed_flash"] = flash
    st.rerun()


def run(op, *args, reset_selection=False, select_result=False, flash="", **kw):
    """Run an editor operation, show a clear error instead of a traceback, and refresh."""
    try:
        res = op(*args, **kw)
    except ValueError as exc:
        st.error(str(exc))
        return
    bump(reset_selection=reset_selection, select=res if select_result else None, flash=flash)


# ------------------------------------------------------------------------------------ sidebar: model source
with st.sidebar:
    st.header("Model")
    source = st.selectbox("Source", ["Workspace dataset", "Synthetic 3D blob", "Synthetic flat (2D)", "Upload file"])
    n_cells = st.slider("Cells", 100, 6000, 800, 100) if source.startswith("Synthetic") else None
    up = st.file_uploader("CSV (x,y[,z,stiffness,group]), JSON or NPY", type=["csv", "json", "npy", "txt"]) if source == "Upload file" else None
    if st.button("Load model", type="primary", width="stretch"):
        try:
            if source == "Workspace dataset":
                ds = workspace.current(ss)
                if ds is None:
                    raise ValueError("The workspace has no dataset. Load cells in the Workbench or Cell Viewer first, or pick another source.")
                set_model(CellModel.from_arrays(ds.points, ds.stiffness, name="workspace"))
            elif source == "Synthetic 3D blob":
                set_model(CellModel.from_arrays(nb.synthetic_cells(n_cells, seed=5, clusters=1, dim=3), name="blob3d"))
            elif source == "Synthetic flat (2D)":
                set_model(CellModel.from_arrays(nb.synthetic_cells(n_cells, seed=5, clusters=1, dim=2), name="flat2d"))
            else:
                if up is None:
                    raise ValueError("Choose a file to upload first.")
                set_model(eio.from_bytes(up.getvalue(), up.name))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.caption("Loading replaces the model being edited and clears its undo history.")

m = model()
if m is None:
    st.info("Load a model from the sidebar to start. Try **Synthetic 3D blob**.")
    st.stop()

if "ed_sel" not in ss or ss["ed_sel"].shape != (m.n,):
    ss["ed_sel"] = np.zeros(m.n, dtype=bool)
sel: np.ndarray = ss["ed_sel"]
rev = ss.get("ed_rev", 0)

with st.sidebar:
    st.header("Selection")
    mode = st.radio("Mouse selection combines by", ["replace", "add", "remove", "intersect"], horizontal=True, format_func=ui.nice_name,
                    help="How a box or lasso drawn in a view combines with the current selection.")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("All", help="Select every cell"):
        bump(select=np.ones(m.n, bool))
    if b2.button("None", help="Clear the selection"):
        bump(select=np.zeros(m.n, bool))
    if b3.button("Inv", help="Invert the selection"):
        bump(select=~sel)
    if b4.button("Grow", help="Add the 4 nearest neighbours of every selected cell"):
        bump(select=m.grow(sel, 4))
    with st.expander("Select by range, sphere, value, id"):
        lo, hi = m.bounds()
        span = np.maximum(hi - lo, 1e-9)
        r = [st.slider(a, float(lo[i]), float(hi[i]) if hi[i] > lo[i] else float(lo[i] + 1), (float(lo[i]), float(hi[i]) if hi[i] > lo[i] else float(lo[i] + 1)),
                       step=float(span[i] / 200), key=f"rng_{a}_{rev}") for i, a in enumerate("xyz") if i < 2 or m.dim == 3]
        if st.button("Select box", key="sel_box"):
            blo = [x[0] for x in r] + ([float(lo[2])] if m.dim == 2 else [])
            bhi = [x[1] for x in r] + ([float(hi[2])] if m.dim == 2 else [])
            bump(select=apply_selection(sel, np.flatnonzero(m.select_box(blo, bhi)), mode))
        c = st.columns(3)
        cx, cy, cz = (c[i].number_input(a, value=float(m.points[:, i].mean()), format="%.3f", key=f"sph_{a}") for i, a in enumerate("xyz"))
        rad = st.number_input("Radius", 0.0, 1e6, float(span[:m.dim].max() / 8), format="%.3f", key="sph_r")
        if st.button("Select sphere", key="sel_sph"):
            bump(select=apply_selection(sel, np.flatnonzero(m.select_sphere((cx, cy, cz), rad)), mode))
        s_lo, s_hi = st.slider("Stiffness range", float(m.stiffness.min()), float(max(m.stiffness.max(), m.stiffness.min() + 0.01)),
                               (float(m.stiffness.min()), float(max(m.stiffness.max(), m.stiffness.min() + 0.01))), key=f"stf_{rev}")
        if st.button("Select by stiffness", key="sel_stf"):
            bump(select=apply_selection(sel, np.flatnonzero(m.select_value(m.stiffness, s_lo, s_hi)), mode))
        grp = st.selectbox("Group", sorted(int(g) for g in np.unique(m.group)), key="sel_grp_v")
        if st.button("Select group", key="sel_grp"):
            bump(select=apply_selection(sel, np.flatnonzero(m.select_group(grp)), mode))
        ids_txt = st.text_input("Cell ids (e.g. 0-10, 25, 40-44)", key="sel_ids_txt")
        if st.button("Select ids", key="sel_ids"):
            ids: list[int] = []
            try:
                for part in ids_txt.replace(" ", "").split(","):
                    if part:
                        a, _, b = part.partition("-")
                        ids += list(range(int(a), int(b or a) + 1))
                bump(select=apply_selection(sel, ids, mode))
            except ValueError:
                st.error("Write ids like 0-10, 25, 40-44.")
    st.header("Display")
    solve = ss.get("ed_solve")
    strain_ok = bool(solve) and solve.get("rev") == rev
    color_mode = st.selectbox("Colour by", views.COLOR_MODES, format_func=ui.nice_name,
                              help="Strain is available after Solve and compare, until the next edit.")
    if color_mode == "strain" and not strain_ok:
        st.caption("No current solve: run Solve and compare first.")

# ------------------------------------------------------------------------------------ toolbar + status
flash = ss.pop("ed_flash", "")
if flash:
    st.success(flash)

t1, t2, t3, t4, t5 = st.columns([1, 1, 1.3, 0.2, 5])
if t1.button("Undo", disabled=not m.can_undo, width="stretch"):
    m.undo()
    bump(reset_selection=True)
if t2.button("Redo", disabled=not m.can_redo, width="stretch"):
    m.redo()
    bump(reset_selection=True)
if t3.button("Revert", disabled=not m.can_undo, width="stretch", help="Return to the model as loaded (this is itself undoable)."):
    m.revert_all()
    bump(reset_selection=True, flash="Reverted to the loaded model.")
lo, hi = m.bounds()
t5.caption(f"**{m.n}** cells · **{int(sel.sum())}** selected · {m.dim}D · {len(m._undo)} edits · "
           f"x {lo[0]:.2f}…{hi[0]:.2f}  y {lo[1]:.2f}…{hi[1]:.2f}" + (f"  z {lo[2]:.2f}…{hi[2]:.2f}" if m.dim == 3 else ""))

# ------------------------------------------------------------------------------------ viewports
strain_vals = solve["node_strain"] if strain_ok else None
plot_cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["autoScale2d"], "scrollZoom": True}


def selectable(view: str, label: str):
    st.markdown(f'<div class="mq-view-label">{label}</div>', unsafe_allow_html=True)
    fig = views.figure_2d(m, sel, view, color_mode, strain_vals)
    ev = st.plotly_chart(fig, key=f"ed_{view}_{rev}", on_select="rerun", selection_mode=("box", "lasso"), width="stretch", config=plot_cfg)
    picked = views.picked_from_event(ev)
    sig = tuple(picked)
    if picked and sig != ss.get(f"ed_sig_{view}"):
        ss[f"ed_sig_{view}"] = sig
        ss["ed_sel"] = apply_selection(sel, picked, mode)
        st.rerun()


def perspective() -> None:
    """The WebGL viewer. Shift-drag in it to box select; the plotly 3D chart is the fallback if the component cannot load."""
    try:
        from mequa import viewer as wv

        fields = {"stiffness": m.stiffness, "group": m.group}
        if strain_vals is not None and len(strain_vals) == m.n:
            fields["strain"] = strain_vals
        scene = wv.build_scene(m.points, fields=fields, categorical=["group"], edges=wv.knn_edges(m.points, 5, 8000), title="model")
        want = {"selection": "uniform"}.get(color_mode, color_mode)
        want = want if want in fields else "uniform"
        sel_rev = int(abs(hash(sel.tobytes())) % 1_000_000_007)
        out = wv.viewer(scene, selected=np.flatnonzero(sel).tolist(), sel_rev=sel_rev, key="ed_gl", height=520, color=want)
        if out and out.get("nonce") != ss.get("ed_gl_nonce"):
            ss["ed_gl_nonce"] = out.get("nonce")
            picked = np.asarray(out.get("ids", []), dtype=int)
            new = np.zeros(m.n, bool)
            new[picked[picked < m.n]] = True
            if not np.array_equal(new, sel):
                ss["ed_sel"] = new
                st.rerun()
    except Exception as exc:  # never lose the editor because the viewer failed
        st.caption(f"3D viewer unavailable ({type(exc).__name__}); showing the basic chart.")
        st.plotly_chart(views.figure_3d(m, sel, color_mode, strain_vals), key="ed_persp", width="stretch", config=plot_cfg)


if m.dim == 3:
    row1 = st.columns(2)
    row2 = st.columns(2)
    with row1[0]:
        selectable("top", "Top · x–y")
    with row1[1]:
        st.markdown('<div class="mq-view-label">Perspective · WebGL, use Box select</div>', unsafe_allow_html=True)
        perspective()
    with row2[0]:
        selectable("front", "Front · x–z")
    with row2[1]:
        selectable("side", "Side · y–z")
    st.caption("Draw a box or lasso in any 2D view. Cells inside it are picked using the mode set in the sidebar, so a box in Top followed by a box in Front "
               "with mode *intersect* selects the cells inside both.")
else:
    selectable("top", "Top · x–y (2D model)")

# ------------------------------------------------------------------------------------ tools
tab_tf, tab_edit, tab_prop, tab_clean, tab_solve, tab_exp, tab_log = st.tabs(
    ["Transform", "Edit", "Properties", "Clean-up", "Solve and compare", "Export", "History"])
n_sel = int(sel.sum())
need = f"Select cells first ({n_sel} selected)." if n_sel == 0 else ""

with tab_tf:
    pivots = {"Selection centre": "selection", "Model centre": "model", "Origin": "origin"}
    c = st.columns(4)
    dx, dy, dz = (c[i].number_input(f"Move {a}", value=0.0, step=0.1, format="%.3f", key=f"mv_{a}") for i, a in enumerate("xyz"))
    if c[3].button("Move", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.move, sel, (dx, dy, dz if m.dim == 3 else 0.0), flash=f"Moved {n_sel} cells.")
    c = st.columns(4)
    rot_axis = c[0].selectbox("Rotate about", ["z", "x", "y"] if m.dim == 3 else ["z"], key="rot_axis")
    rot_deg = c[1].number_input("Degrees", value=15.0, step=5.0, key="rot_deg")
    rot_piv = c[2].selectbox("Pivot", list(pivots), key="rot_piv")
    if c[3].button("Rotate", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.rotate, sel, rot_axis, rot_deg, pivots[rot_piv], flash=f"Rotated {n_sel} cells.")
    c = st.columns(5)
    fx, fy, fz = (c[i].number_input(f"Scale {a}", value=1.0, min_value=0.01, step=0.05, format="%.3f", key=f"sc_{a}") for i, a in enumerate("xyz"))
    sc_piv = c[3].selectbox("Pivot", list(pivots), key="sc_piv")
    if c[4].button("Scale", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.scale, sel, (fx, fy, fz), pivots[sc_piv], flash=f"Scaled {n_sel} cells.")
    c = st.columns(4)
    mi_axis = c[0].selectbox("Mirror across", ["x", "y", "z"] if m.dim == 3 else ["x", "y"], key="mi_axis")
    mi_piv = c[1].selectbox("Pivot", list(pivots), key="mi_piv")
    if c[3].button("Mirror", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.mirror, sel, mi_axis, pivots[mi_piv], flash=f"Mirrored {n_sel} cells.")

with tab_edit:
    c = st.columns(4)
    ox, oy, oz = (c[i].number_input(f"Offset {a}", value=0.5, step=0.1, format="%.3f", key=f"du_{a}") for i, a in enumerate("xyz"))
    if c[3].button("Duplicate selection", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.duplicate, sel, (ox, oy, oz if m.dim == 3 else 0.0), select_result=True, flash=f"Duplicated {n_sel} cells; the copies are selected.")
    c = st.columns(5)
    ax_, ay_, az_ = (c[i].number_input(f"New cell {a}", value=float(m.points[:, i].mean()), format="%.3f", key=f"ad_{a}") for i, a in enumerate("xyz"))
    ak = c[3].number_input("Stiffness", 0.05, 20.0, 1.0, 0.1, key="ad_k")
    if c[4].button("Add cell", width="stretch"):
        try:
            idx = m.add_cell((ax_, ay_, az_), ak)
        except ValueError as exc:
            st.error(str(exc))
        else:
            new = np.zeros(m.n, bool)
            new[idx] = True
            bump(select=new, flash=f"Added cell {idx}.")
    c = st.columns(4)
    cnt = c[0].number_input("Random cells", 1, 5000, 50, 10, key="rc_n")
    box_note = c[1].selectbox("Fill", ["selection bounding box", "whole model box"], key="rc_box")
    if c[3].button("Add random cells", width="stretch"):
        if box_note.startswith("selection") and n_sel > 0:
            blo, bhi = m.points[sel].min(axis=0), m.points[sel].max(axis=0)
        else:
            blo, bhi = m.bounds()
        run(m.add_random, int(cnt), blo, bhi, 1.0, int(rev) + 1, flash=f"Added {int(cnt)} random cells.")
    c = st.columns(4)
    jit = c[0].number_input("Jitter amount", 0.0, 100.0, 0.05, 0.01, format="%.3f", key="jit")
    if c[3].button("Jitter selection", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.jitter, sel, jit, int(rev) + 1, flash=f"Jittered {n_sel} cells.")
    st.divider()
    if st.button("Delete selection", disabled=n_sel == 0, help=need, type="primary"):
        run(m.delete, sel, reset_selection=True, flash=f"Deleted {n_sel} cells.")

with tab_prop:
    st.caption("Stiffness is relative (1.0 = the reference cell). It is a modelling input, not a measured tissue property. "
               "The solver turns it into link stiffness with the harmonic mean of the two end cells.")
    c = st.columns(4)
    sv = c[0].number_input("Set stiffness to", 0.05, 20.0, 2.0, 0.1, key="pv")
    if c[1].button("Set", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.set_stiffness, sel, sv, flash=f"Set stiffness of {n_sel} cells to {sv:g}.")
    sf = c[2].number_input("Multiply by", 0.05, 20.0, 1.5, 0.1, key="pf")
    if c[3].button("Multiply", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.scale_stiffness, sel, sf, flash=f"Scaled stiffness of {n_sel} cells by {sf:g}.")
    c = st.columns(4)
    gv = c[0].number_input("Group id", 0, 99, 1, 1, key="gv")
    if c[1].button("Assign group", disabled=n_sel == 0, help=need, width="stretch"):
        run(m.set_group, sel, int(gv), flash=f"Put {n_sel} cells in group {int(gv)}.")
    st.dataframe(pd.DataFrame({"group": m.group, "stiffness": m.stiffness}).groupby("group").agg(cells=("stiffness", "size"), mean_stiffness=("stiffness", "mean"),
                                                                                                  min=("stiffness", "min"), max=("stiffness", "max")).round(3), width="stretch")

with tab_clean:
    c = st.columns(4)
    ok_k = c[0].number_input("Neighbours k", 2, 30, 6, key="ol_k")
    ok_z = c[1].number_input("Sigmas", 1.0, 10.0, 3.0, 0.5, key="ol_z")
    if c[3].button("Select outliers", width="stretch", help="Cells unusually far from their neighbours"):
        found = m.find_outliers(int(ok_k), float(ok_z))
        bump(select=found, flash=f"{int(found.sum())} outlier cells selected." if found.any() else "No outliers found.")
    c = st.columns(4)
    mr = c[0].number_input("Merge radius", 0.0, 1e6, float(max(hi[:m.dim] - lo[:m.dim]) / 500), format="%.4f", key="mg_r")
    if c[3].button("Merge close cells", width="stretch"):
        run(m.merge_close, float(mr), reset_selection=True, flash="Merged close cells.")
    c = st.columns(4)
    ds_n = c[0].number_input("Keep cells", 8, max(m.n, 9), min(m.n, max(8, m.n // 2)), 50, key=f"ds_n_{m.n}")
    if c[3].button("Downsample", width="stretch", disabled=m.n <= 8):
        run(m.downsample, int(ds_n), int(rev) + 1, reset_selection=True, flash=f"Downsampled to {int(ds_n)} cells.")
    c = st.columns(4)
    nw = c[0].number_input("x-width", 0.1, 1e6, 10.0, 1.0, key="nw")
    if c[2].button("Recentre", width="stretch"):
        m.recenter()
        bump(flash="Model centred on the origin.")
    if c[3].button("Normalise width", width="stretch", help="Scale the whole model uniformly so its x-extent is the given width"):
        run(m.normalize_extent, float(nw), flash=f"x-width set to {nw:g}.")

with tab_solve:
    c = st.columns(4)
    knn = c[0].slider("Neighbours per cell", 4, 14, 8, key="sv_knn")
    ld = c[1].slider("Load (fraction of width)", 0.01, 0.12, 0.05, 0.01, key="sv_ld")
    zn = c[2].slider("Zones", 2, 8, 3, key="sv_zn")
    if c[3].button("Solve and compare", type="primary", width="stretch"):
        try:
            with st.spinner("Relaxing the spring network (edited model and the model as loaded)..."):
                pts0, st0 = m.original_arrays()
                before, _ = Session(pts0, stiffness=None if np.allclose(st0, 1.0) else st0, k=zn, knn=knn).run(ld, full=True)
                after, res = Session(m.analysis_points(), stiffness=m.analysis_stiffness(), k=zn, knn=knn).run(ld, full=True)
            ss["ed_solve"] = {"rev": rev, "before": before, "after": after, "node_strain": np.asarray(res["node_strain"], float), "knn": knn, "load": ld}
            st.rerun()
        except Exception as exc:
            st.error(f"Solve failed: {exc}")
    sv_ = ss.get("ed_solve")
    if sv_ and sv_.get("rev") == rev:
        keys = [("cells", "Cells"), ("mechanical_links", "Links"), ("sphi", "SPHI"), ("mean_strain", "Mean strain"), ("p95_strain", "p95 strain"),
                ("max_strain", "Max strain"), ("residual", "Residual"), ("n_components", "Components")]
        rows = []
        for k_, label in keys:
            b, a = sv_["before"].get(k_), sv_["after"].get(k_)
            d = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            rows.append({"quantity": label, "as loaded": b, "edited": a, "change": None if d is None else round(d, 6)})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if sv_["after"].get("load_path") is False:
            st.warning("In the edited model the pulled edge is not connected to the held edge, so strain is a decaying transient. Raise the neighbour count or reconnect the model.")
        st.caption("Switch **Colour by** to Strain in the sidebar to see where the edited model is loaded. The numbers are for a simplified spring-network "
                   "model on research data, not a clinical prediction. The comparison is cleared by the next edit.")
    else:
        st.info("Run **Solve and compare** to see how the edits changed SPHI and strain.")

with tab_exp:
    choice = st.selectbox("Format", list(eio.EXPORTERS), key="ex_fmt")
    m.name = st.text_input("File name", m.name, key="ex_name").strip() or "model"
    fname, data, mime = eio.export(m, choice)
    st.download_button(f"Download {fname}", data=data, file_name=fname, mime=mime, type="primary")
    st.caption("CSV opens in the Cell Viewer and Workbench. The edited model is also shared with the Cell Viewer, Agent and the chat dock automatically "
               "(source “Workspace dataset”).")
    st.json(m.describe(), expanded=False)

with tab_log:
    if m.log:
        st.dataframe(pd.DataFrame({"#": range(1, len(m.log) + 1), "operation": m.log}).iloc[::-1], width="stretch", hide_index=True)
    else:
        st.info("No edits yet.")

# the edited model is the workspace dataset for every other page and for the chat dock
workspace.publish(ss, m.analysis_points(), m.analysis_stiffness(), label=f"edited model '{m.name}'", source="Model Editor")
