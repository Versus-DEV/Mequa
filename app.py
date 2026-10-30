"""Mequa Streamlit entry point."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mequa.utils.env import load_env_file
from mequa.vision.sam_loader import SamLoader


load_env_file()

st.set_page_config(
    page_title="Mequa",
    page_icon="🧬",
    layout="wide",
)

st.title("Mequa")
st.caption(
    "Research-only tissue mechanics workspace. "
    "No clinical guidance for an individual person."
)

loader = SamLoader()
sam_ok, sam_message = loader.is_available()

left, right = st.columns([2, 1])

with left:
    st.subheader("Core loop")

    st.write(
        """
        1. **Workbench** - the verified solver (Julia, Rust, NumPy), firewall, Simulink import, benchmark.
        2. **SAM Mechanics** - image to masks to cells to per-cell stiffness.
        3. **Two-Machine Status** - check the Llama Guard node.
        4. **Adaptive Agent** - Llama calls the solver, sweeps loads, grounded summary.
        5. **Compliance firewall** - decision for any request.
        6. **Simulink editor** - prompt to validated block spec to simulation.
        7. **Cell viewer** - 2D and 3D.
        8. **Benchmarks and MPI scaling** - measured results.
        """
    )

    st.info("Use the sidebar to open each page.")

with right:
    st.subheader("Environment")

    st.write(f"Python: `{sys.version.split()[0]}`")
    st.write(f"Platform: `{platform.platform()}`")
    st.write(f"Repo root: `{ROOT}`")

    if sam_ok:
        st.success("SAM: available")
    else:
        st.warning("SAM: not available")

    with st.expander("SAM details", expanded=False):
        st.write(sam_message)
        st.write(f"Checkpoint: `{loader.checkpoint_path}`")
        st.write(f"Model type: `{loader.model_type}`")

    st.divider()

    st.write("Ollama main:")
    st.code(os.getenv("OLLAMA_MAIN_URL", "http://127.0.0.1:11434"))

    st.write("Ollama guard:")
    st.code(os.getenv("OLLAMA_GUARD_URL", "not set"))

    st.write("Guard model:")
    st.code(os.getenv("GUARD_MODEL", "llama-guard3:1b"))

    st.write("Planner model:")
    st.code(os.getenv("PLANNER_MODEL", "llama3.2:3b"))