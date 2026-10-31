"""Llama as a tool-using scientist. Every number comes from the verified solver; the answer is grounded."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import backend as backend_mod
from mequa import ingest
from mequa import numpy_backend as nb
from mequa.agents.adaptive_investigator import AdaptiveInvestigator
from mequa.session import Session
from mequa.utils.env import load_env_file

from mequa import ui  # noqa: E402

load_env_file()
ui.page_header('Adaptive Agent', 'Llama calls the solver, reads the numbers and decides what to run next. The firewall checks the request and the reply.')


@st.cache_resource
def numpy_be():
    return backend_mod.load("numpy")


with st.sidebar:
    st.header("Dataset")
    source = st.selectbox("Source", ["Synthetic 3D blob", "Synthetic 2D blob", "SAM cells (outputs/sam/cells.csv)"])
    n = st.slider("Synthetic cells", 300, 3000, 1000, 100)
    knn = st.slider("Neighbors per cell", 4, 14, 8)
    st.header("Agent")
    model = st.selectbox("Llama model", ["llama3.2:3b", "llama3.1:latest"])
    limit = st.number_input("Strain limit (p95 strain that counts as failure)", 0.005, 0.5, 0.05, 0.005, help="A modelling choice you set, not a measured tissue property.")
    max_iter = st.slider("Max tool rounds", 2, 8, 6)
    question = st.text_area("Research question", "Find the load at which this tissue sample reaches the strain limit.", height=100)
    run = st.button("Run Investigation", type="primary")

if run:
    if source.startswith("SAM"):
        path = ROOT / "outputs" / "sam" / "cells.csv"
        if not path.exists():
            st.error("No SAM cells yet. Run the SAM Mechanics page on a real image first.")
            st.stop()
        pts, _, stiff = ingest.load_cells_full(path)
    else:
        pts, stiff = nb.synthetic_cells(n, seed=5, clusters=1, dim=3 if "3D" in source else 2), None
    sess = Session(pts, stiffness=stiff, backend=numpy_be(), k=3, knn=knn)
    with st.spinner("Llama is working. On CPU a full run takes a minute or more."):
        out = AdaptiveInvestigator(sess, model=model, max_iterations=max_iter, strain_limit=limit).run(question)
    st.session_state["agent_out"] = out

out = st.session_state.get("agent_out")
if out is None:
    st.info("Choose a dataset, write a question and click Run investigation.")
    st.stop()

if out["status"] == "blocked":
    st.error("BLOCKED by the firewall (" + ", ".join(out["blocked_by"]) + "). Mequa does not give guidance for individuals.")
    st.stop()
if out["status"] == "model_unavailable":
    st.error("Ollama is not reachable or the model is not installed: " + out["error"])
    st.stop()
if out["status"] == "reply_blocked":
    st.error("The generated reply was blocked by Llama Guard.")
    st.stop()

tab_sum, tab_trace, tab_ground = st.tabs(["Summary", "Tool-call trace", "Number grounding"])
with tab_sum:
    st.markdown(out["summary"])
    st.caption(f"Model {out['model']}, {out['iterations']} rounds.")
with tab_trace:
    for s in out["trace"]:
        with st.expander(f"Round {s['iteration']}: {s['type']}" + (f" {s.get('tool', '')}" if s["type"] == "tool_call" else "")):
            st.json(s)
with tab_ground:
    g = out["grounding"]
    if g["fallback"]:
        st.warning("The model kept using numbers that are not in the solver output, so a template summary built only from solver results was used.")
    else:
        st.success("Every number in the summary traces to a solver result.")
    st.json(g)
