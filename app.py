"""Mequa Streamlit entry point: themed shell, grouped navigation, home page."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import ui
from mequa.utils.env import load_env_file
from mequa.vision.sam_loader import SamLoader

load_env_file()

st.set_page_config(page_title="Mequa", page_icon="assets/logo.png" if (ROOT / "assets" / "logo.png").exists() else "🧬", layout="wide")
ui.apply()


def home() -> None:
    ui.page_header(
        "Mequa",
        "Quantitative workbench for tissue-mechanics research data. Research datasets only, no clinical guidance for any individual.",
        eyebrow="Tissue Mechanics Workspace",
    )

    loader = SamLoader()
    sam_ok, sam_message = loader.is_available()
    guard_url = os.getenv("OLLAMA_GUARD_URL", "")
    st.markdown(
        ui.pill("SAM Ready" if sam_ok else "SAM Unavailable", "ok" if sam_ok else "bad")
        + ui.pill("Guard Node Set" if guard_url else "Guard Node Not Set", "ok" if guard_url else "idle")
        + ui.pill("Python " + sys.version.split()[0])
        + ui.pill(platform.system()),
        unsafe_allow_html=True,
    )

    st.markdown("### Core Loop")
    ui.cards([
        ("01", "Workbench", "The verified solver on Julia, Rust or NumPy, plus firewall, Simulink import and evidence."),
        ("02", "SAM Mechanics", "Research image to masks to cells to per-cell stiffness."),
        ("03", "Adaptive Agent", "Llama calls the solver as tools, sweeps loads and writes a grounded summary."),
        ("04", "Compliance Firewall", "Rules, Llama Guard and an intent classifier decide on every request."),
        ("05", "Simulink Editor", "Plain English to a validated block spec to a simulation."),
        ("06", "Cell Viewer", "Interactive 2D and 3D views of cells coloured by strain, zone or stiffness."),
        ("07", "Benchmarks", "Measured solver and MPI scaling results with the machine that produced them."),
        ("08", "Two-Machine Status", "Check the Llama Guard node on the second laptop."),
    ])

    st.markdown("### Environment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ollama Main", os.getenv("OLLAMA_MAIN_URL", "127.0.0.1:11434").replace("http://", ""))
    c2.metric("Ollama Guard", (guard_url or "not set").replace("http://", ""))
    c3.metric("Guard Model", os.getenv("GUARD_MODEL", "llama-guard3:1b"))
    c4.metric("Planner Model", os.getenv("PLANNER_MODEL", "llama3.2:3b"))
    with st.expander("SAM details"):
        st.write(sam_message)
        st.write(f"Checkpoint: `{loader.checkpoint_path}`")
        st.write(f"Model type: `{loader.model_type}`")
        st.write(f"Repo root: `{ROOT}`")


nav = st.navigation(
    {
        "Overview": [st.Page(home, title="Home", icon=":material/home:", default=True)],
        "Analysis": [
            st.Page("pages/0_workbench.py", title="Workbench", icon=":material/dashboard:"),
            st.Page("pages/1_sam_mechanics.py", title="SAM Mechanics", icon=":material/biotech:"),
            st.Page("pages/5_mesh_viewer.py", title="Cell Viewer", icon=":material/view_in_ar:"),
        ],
        "Intelligence": [
            st.Page("pages/3_adaptive_agent.py", title="Adaptive Agent", icon=":material/psychology:"),
            st.Page("pages/3b_research_pivot.py", title="Compliance Firewall", icon=":material/shield:"),
            st.Page("pages/4_simulink_editor.py", title="Simulink Editor", icon=":material/schema:"),
        ],
        "System": [
            st.Page("pages/6_solver_benchmark.py", title="Benchmarks", icon=":material/speed:"),
            st.Page("pages/2_two_machine_status.py", title="Two-Machine Status", icon=":material/lan:"),
        ],
    }
)
nav.run()
