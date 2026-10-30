import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mequa import agent as agent_mod
from mequa import backend, couple, ingest, simulink, slx_demo
from mequa.firewall import check_request


st.markdown(
    """
    <style>
    .mequa-ok {background:#0f3d24;border:1px solid #22c55e;color:#bbf7d0;padding:10px 14px;border-radius:8px;font-weight:600;}
    .mequa-halt {background:#4a0d14;border:1px solid #ef4444;color:#fecaca;padding:10px 14px;border-radius:8px;font-weight:700;}
    .mequa-logo {background:#ffffff;border-radius:10px;padding:8px;display:inline-block;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Starting the compute backend (the Julia core compiles on first start and can take a minute)...")
def get_backend(kind):
    return backend.load({"julia": "julia_worker", "rust": "rust", "numpy": "numpy"}[kind])


backend_kind = st.sidebar.selectbox("Compute backend", ["julia", "rust", "numpy"])
be = get_backend(backend_kind)
if be.name != backend_kind:
    st.sidebar.warning(f"{backend_kind} is unavailable, running on {be.name}. See the terminal for the reason.")

logo = Path("assets/logo.png")
head_left, head_right = st.columns([1, 6])
with head_left:
    if logo.exists():
        st.markdown('<div class="mequa-logo">', unsafe_allow_html=True)
        st.image(str(logo), width=90)
        st.markdown("</div>", unsafe_allow_html=True)
with head_right:
    st.title("Mequa")
    st.caption("Quantitative workbench for tissue-mechanics research data. Research datasets only, no clinical guidance.")

with st.sidebar:
    st.subheader("Compute")
    st.write(f"Backend: **{be.name}**")
    st.subheader("Data")
    source = st.radio("Source", ["Synthetic cells", "Upload CSV (x,y)", "Segment an image"])
    n_cells = st.slider("Synthetic cells", 500, 10000, 2000, 500) if source == "Synthetic cells" else None
    if source == "Upload CSV (x,y)":
        up = st.file_uploader("CSV with header x,y", type=["csv"])
        if up is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(up.getvalue())
            st.session_state["points"] = ingest.load_points_csv(tmp.name)
    if source == "Segment an image":
        img_file = st.file_uploader("Public research image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        method = st.selectbox("Segmentation", ["sam", "classical"])
        if img_file is not None and st.button("Segment"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(img_file.getvalue())
            with st.spinner("Segmenting (SAM on CPU can take a minute)..."):
                try:
                    pts, _ = ingest.segment(ingest.load_image(tmp.name), method)
                    st.session_state["points"] = pts
                except Exception as exc:
                    st.error(f"Segmentation failed: {exc}")
    st.subheader("Experiment")
    zones = st.slider("Density zones", 2, 8, 3)
    knn = st.slider("Neighbors per cell", 3, 12, 6)
    load = st.slider("Pull (fraction of width)", 0.01, 0.15, 0.05, 0.01)
    run = st.button("Run analysis", type="primary")

if source == "Synthetic cells" and n_cells is not None:
    key = ("synthetic", n_cells)
    if st.session_state.get("points_key") != key:
        st.session_state["points"] = be.synthetic_cells(n_cells)
        st.session_state["points_key"] = key

points = st.session_state.get("points")

if run and points is not None and len(points) > knn + 1:
    with st.spinner("Solving..."):
        st.session_state["result"] = be.analyze(points, k=zones, knn=knn, load=load)

tab_map, tab_risk, tab_agent, tab_guard, tab_slx, tab_evidence = st.tabs(["Spatial map", "Risk monitor", "Agent", "Compliance firewall", "Simulink import", "Evidence"])

result = st.session_state.get("result")

with tab_map:
    if points is None:
        st.info("Choose a data source in the sidebar.")
    elif result is None or len(result["labels"]) != len(points):
        st.write(f"{len(points)} cells loaded. Press Run analysis.")
        st.plotly_chart(px.scatter(x=points[:, 0], y=points[:, 1], opacity=0.6), width="stretch")
    else:
        mode = st.radio("Color by", ["Zone", "Strain", "Failure probability"], horizontal=True)
        df = pd.DataFrame({
            "x": points[:, 0],
            "y": points[:, 1],
            "zone": result["labels"].astype(str),
            "strain": result["node_strain"],
            "failure_prob": result["failure_prob"],
        })
        if mode == "Zone":
            fig = px.scatter(df, x="x", y="y", color="zone", opacity=0.8)
        elif mode == "Strain":
            fig = px.scatter(df, x="x", y="y", color="strain", color_continuous_scale="Turbo", opacity=0.85)
        else:
            fig = px.scatter(df, x="x", y="y", color="failure_prob", color_continuous_scale="Inferno", opacity=0.85)
        fig.update_yaxes(scaleanchor="x")
        st.plotly_chart(fig, width="stretch")

with tab_risk:
    if result is None:
        st.info("Run an analysis first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("SPHI", f"{result['sphi']:.4f}")
        c2.metric("Critical strain (p90)", f"{result['s_crit']:.4f}")
        c3.metric("Mechanical links", f"{result['n_edges']}")
        st.dataframe(pd.DataFrame(agent_mod.zone_stats(result)), width="stretch")
        st.plotly_chart(px.histogram(x=result["node_strain"], nbins=60, labels={"x": "node strain"}), width="stretch")
        st.caption("Strain model: simplified spring network. SPHI: leading eigenvalue of the strain-weighted neighbor graph. Failure probability uses a stated threshold, not clinical data.")

with tab_agent:
    request = st.text_area(
        "Describe an experiment",
        "Cluster the cells into density zones and report strain risk under a 5 percent load",
    )
    if st.button("Run agent"):
        if points is None:
            st.warning("Load data first.")
        else:
            with st.spinner("Llama is planning, Julia is computing..."):
                try:
                    out = agent_mod.run_agent(request, points=points, backend=be)
                except OSError:
                    out = None
                    st.error("Ollama is not reachable. Start it and pull llama3.2:3b and llama-guard3:1b.")
            if out is not None:
                if out["status"] == "blocked":
                    st.markdown(
                        '<div class="mequa-halt">COMPLIANCE HALT - clinical guidance request detected. '
                        f"Blocked by: {', '.join(out['blocked_by'])}</div>",
                        unsafe_allow_html=True,
                    )
                elif out["status"] == "reply_blocked":
                    st.markdown('<div class="mequa-halt">COMPLIANCE HALT - generated reply blocked by output check.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="mequa-ok">VERIFIED - request and reply passed the firewall</div>', unsafe_allow_html=True)
                    st.subheader("Summary")
                    st.write(out["summary"])
                    st.subheader("Plan and critic loop")
                    st.dataframe(pd.DataFrame(out["history"]), width="stretch")
                    st.session_state["result"] = out["result"]
                st.subheader("Trace")
                st.dataframe(pd.DataFrame(out["trace"]), width="stretch")

with tab_guard:
    probe = st.text_input("Test a query", "Compute shear stress boundaries across grid parameters")
    if st.button("Check query"):
        try:
            verdict = check_request(probe)
            if verdict.degraded and verdict.allowed:
                st.warning("Firewall models were unreachable, so this query was NOT verified.")
            elif verdict.allowed:
                st.markdown('<div class="mequa-ok">STATUS: VERIFIED SECURE</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="mequa-halt">COMPLIANCE HALT: {", ".join(verdict.blocked_by)}</div>', unsafe_allow_html=True)
        except OSError:
            st.error("Ollama is not reachable.")
    report_path = Path("results/eval_all.json")
    if report_path.exists():
        st.subheader("Measured evaluation")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for set_name, policies in report.items():
            st.write(set_name)
            st.dataframe(pd.DataFrame(policies).T, width="stretch")


with tab_slx:
    st.write("Import a Simulink .slx model without MATLAB, simulate it, and use its output signal to drive the strain solver.")
    st.caption("Supported blocks: " + ", ".join(sorted(simulink.SUPPORTED)) + ". Subsystems and transfer functions are not supported yet.")
    left, right = st.columns(2)
    with left:
        slx_up = st.file_uploader("Upload .slx", type=["slx"])
        if slx_up is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".slx") as tmp:
                tmp.write(slx_up.getvalue())
            st.session_state["slx_path"] = tmp.name
            st.session_state["slx_vars"] = ""
    with right:
        if st.button("Use demo 2-element Windkessel model"):
            tmp_dir = tempfile.mkdtemp()
            st.session_state["slx_path"] = slx_demo.make_demo_slx(str(Path(tmp_dir) / "windkessel_demo.slx"))
            st.session_state["slx_vars"] = "R=1.0, C=0.3"
    slx_vars = st.text_input("Model variables (name=value, comma separated)", st.session_state.get("slx_vars", ""))
    t_end = st.slider("Simulated seconds", 1.0, 20.0, 6.0, 1.0)
    if st.session_state.get("slx_path"):
        try:
            model = simulink.load_slx(st.session_state["slx_path"])
            st.write(f"Model **{model.name}**: {len(model.blocks)} blocks, {len(model.wires)} connections")
            st.dataframe(pd.DataFrame(simulink.block_table(model)), width="stretch")
            if st.button("Simulate"):
                with st.spinner("Simulating..."):
                    sim = simulink.simulate(model, t_end=t_end, dt=0.002, variables=simulink.parse_assignments(slx_vars))
                st.session_state["sim"] = sim
        except simulink.ModelError as exc:
            st.error(f"Import failed: {exc}")
    sim = st.session_state.get("sim")
    if sim is not None:
        sig_name = st.selectbox("Signal", list(sim.signals.keys()))
        st.plotly_chart(px.line(x=sim.time, y=sim.signals[sig_name], labels={"x": "time (s)", "y": sig_name}), width="stretch")
        samples = st.slider("Load samples across the signal", 6, 40, 16)
        if st.button("Drive strain solver with this signal"):
            if points is None:
                st.warning("Load cell data first.")
            else:
                with st.spinner("Solving at each sampled load..."):
                    rows = couple.drive_strain(sim.time, sim.signals[sig_name], points, be, samples=samples)
                cdf = pd.DataFrame(rows)
                st.plotly_chart(px.line(cdf, x="time", y=["sphi", "peak_strain"], markers=True), width="stretch")
                st.dataframe(cdf, width="stretch")
                st.caption("Quasi-static coupling: the signal is rescaled into the pull fraction and the strain network is solved at each sample. Illustrative, not a validated fluid-structure model.")


with tab_evidence:
    st.write("Measured results written by the benchmark and evaluation scripts. Nothing on this tab is typed in by hand.")
    bench_path = Path("results/benchmark.json")
    if bench_path.exists():
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        rows = []
        for r in bench["rows"]:
            row = {"cells": r["cells"], "numpy_s": round(r["numpy_total_s"], 3)}
            if "rust_total_s" in r:
                row["rust_s"] = round(r["rust_total_s"], 3)
                row["rust_vs_numpy"] = round(r["rust_speedup_vs_numpy"], 2)
            if "julia_total_s" in r:
                row["julia_s"] = round(r["julia_total_s"], 3)
                row["julia_vs_numpy"] = round(r["speedup_total"], 2)
                row["julia_sphi_rel_diff"] = r["sphi_relative_difference"]
            rows.append(row)
        st.subheader("Benchmark: Julia, Rust and NumPy")
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption(
            f"Machine: {bench.get('machine')}, Python {bench.get('python')}, NumPy {bench.get('numpy')}, "
            f"Julia threads: {bench.get('julia_threads')}, Rust threads: {bench.get('rust_threads')}. "
            f"{bench.get('baseline_note', '')}"
        )
    else:
        st.info("Run python -m mequa.benchmark to populate this table.")
    eval_path = Path("results/eval_all.json")
    if eval_path.exists():
        st.subheader("Firewall evaluation")
        report = json.loads(eval_path.read_text(encoding="utf-8"))
        for set_name, policies in report.items():
            st.write(set_name)
            st.dataframe(pd.DataFrame(policies).T, width="stretch")
    else:
        st.info("Run python -m mequa.eval_all to populate this table.")
