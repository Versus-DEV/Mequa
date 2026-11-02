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
from mequa.ui import shell
from mequa.utils.env import load_env_file
from mequa.vision.sam_loader import SamLoader

load_env_file()

st.set_page_config(page_title="Mequa Workspace", page_icon="assets/logo.png" if (ROOT / "assets" / "logo.png").exists() else "🧬",
                   layout="wide", initial_sidebar_state="expanded")
shell.render()
shell.render_status("pre")


def home() -> None:
    from mequa import chat as chat_mod, settings, workspace

    ui.page_header(
        "Mequa",
        "A research workbench for tissue mechanics: bring in an image or a set of cells, solve the strain, and ask questions about the result. "
        "Research data only; no clinical guidance for any individual.",
        eyebrow="Tissue mechanics workspace",
    )
    ds = workspace.current(st.session_state)
    main = chat_mod.ollama_status(os.getenv("OLLAMA_MAIN_URL") or "http://127.0.0.1:11434")
    mode_txt = {"main": "Main machine", "dedicated": "Guard machine only", "auto": "Automatic"}[settings.guard_mode()]
    ui.kpis([
        ("Dataset", f"{ds.n} cells" if ds else "None loaded", f"{ds.dim}D · from {ds.source}" if ds else "Start in Image Lab or Workbench"),
        ("Language model", "Online" if main["up"] else "Offline", "Ollama on this machine" if main["up"] else "Start Ollama to use the assistant"),
        ("Safety firewall", mode_txt, "Checks every request and every reply"),
    ])

    c1, c2, c3, _ = st.columns([1.3, 1.3, 1.3, 3])
    c1.page_link(PAGES["lab"], label="Analyse an image", icon=":material/image_search:", width="stretch")
    c2.page_link(PAGES["bench"], label="Open workbench", icon=":material/dashboard:", width="stretch")
    c3.page_link(PAGES["agent"], label="Ask the assistant", icon=":material/chat:", width="stretch")

    groups = [
        ("Analyse", [
            ("image_search", "Image Lab", "Upload an image or z-stack. Nuclei are detected and sent to every other tool."),
            ("dashboard", "Workbench", "Verified strain solver on Julia, Rust or NumPy, with evidence and Simulink import."),
            ("biotech", "SAM Mechanics", "Research image to masks, cells and per-cell stiffness."),
        ]),
        ("Explore and edit", [
            ("view_in_ar", "3D Viewer", "Orbit, select, clip and measure cell models and PDB / XYZ molecules."),
            ("edit", "Model Editor", "Edit a cell model in three views, set stiffness, then re-solve."),
            ("scatter_plot", "Cell Viewer", "Quick 3D view of the current cells."),
        ]),
        ("Assistant and safety", [
            ("psychology", "Adaptive Agent", "The model calls the solver as tools, sweeps loads and writes a checked summary."),
            ("shield", "Compliance Firewall", "Rules, Llama Guard and an intent check on every request."),
            ("schema", "Simulink Editor", "Plain English to a validated block spec to a simulation."),
        ]),
        ("System", [
            ("speed", "Benchmarks", "Measured solver and MPI scaling, with the machine that produced them."),
            ("lan", "Machines", "Choose where the firewall runs and check both nodes."),
        ]),
    ]
    for title, items in groups:
        st.markdown(f'<div class="mq-group">{title}</div>', unsafe_allow_html=True)
        ui.cards(items)

    with st.expander("System details"):
        loader = SamLoader()
        sam_ok, sam_message = loader.is_available()
        guard_url = settings.configured_guard_url()
        st.write(f"Python {sys.version.split()[0]} on {platform.system()} · SAM: {'ready' if sam_ok else 'unavailable'}")
        st.write(f"Ollama main: `{os.getenv('OLLAMA_MAIN_URL', '127.0.0.1:11434')}` · guard: `{guard_url or 'not set'}`")
        st.write(f"Guard model: `{os.getenv('GUARD_MODEL', 'llama-guard3:1b')}` · planner model: `{os.getenv('PLANNER_MODEL', 'llama3.2:3b')}`")
        st.write(sam_message)


PAGES = {
    "home": st.Page(home, title="Home", icon=":material/home:", default=True),
    "bench": st.Page("views/0_workbench.py", title="Workbench", icon=":material/dashboard:"),
    "sam": st.Page("views/1_sam_mechanics.py", title="SAM Mechanics", icon=":material/biotech:"),
    "lab": st.Page("views/9_image_lab.py", title="Image Lab", icon=":material/image_search:"),
    "viewer": st.Page("views/8_viewer.py", title="3D Viewer", icon=":material/view_in_ar:"),
    "cells": st.Page("views/5_mesh_viewer.py", title="Cell Viewer", icon=":material/scatter_plot:"),
    "editor": st.Page("views/7_model_editor.py", title="Model Editor", icon=":material/edit:"),
    "agent": st.Page("views/3_adaptive_agent.py", title="Adaptive Agent", icon=":material/psychology:"),
    "fw": st.Page("views/3b_research_pivot.py", title="Compliance Firewall", icon=":material/shield:"),
    "sim": st.Page("views/4_simulink_editor.py", title="Simulink Editor", icon=":material/schema:"),
    "benchmarks": st.Page("views/6_solver_benchmark.py", title="Benchmarks", icon=":material/speed:"),
    "machines": st.Page("views/2_two_machine_status.py", title="Machines", icon=":material/lan:"),
}
nav = st.navigation(
    {
        "Overview": [PAGES["home"]],
        "Analysis": [PAGES[k] for k in ("bench", "sam", "lab", "viewer", "cells", "editor")],
        "Intelligence": [PAGES[k] for k in ("agent", "fw", "sim")],
        "System": [PAGES["benchmarks"], PAGES["machines"]],
    }
)
try:
    from streamlit.runtime.scriptrunner import StopException
except Exception:  # pragma: no cover - other Streamlit versions
    StopException = ()
try:
    nav.run()
except StopException:
    pass  # a page called st.stop(); still draw the top bar and status rail below
shell.render_status("post")
shell.refresh_if_dataset_changed()
