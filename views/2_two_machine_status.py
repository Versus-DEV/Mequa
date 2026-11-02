"""Machines: choose who runs the safety firewall, and check the nodes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import settings, ui  # noqa: E402
from mequa.guard.remote_guard import list_models  # noqa: E402
from mequa.ui import shell  # noqa: E402
from mequa.utils.env import load_env_file  # noqa: E402

load_env_file()
ui.page_header("Machines", "Choose where the safety firewall runs. Local network only; never expose Ollama to the internet.", eyebrow="System")

g = settings.guard_status()
state = "ok" if g["ok"] else "bad"
st.markdown(ui.pill(g["note"], state), unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("### Firewall mode")
    shell.guard_mode_selector("guard_mode_page")
    st.caption("The firewall (rules, Llama Guard, intent classifier) always runs. This only decides on which machine. "
               "If the chosen guard cannot be reached, requests are refused rather than answered unchecked.")
with right:
    st.markdown("### Addresses")
    cfg = settings.load()
    url = st.text_input("Guard machine URL", value=cfg.get("guard_url") or os.getenv("OLLAMA_GUARD_URL", ""), placeholder="http://192.168.1.72:11434")
    if url != (cfg.get("guard_url") or os.getenv("OLLAMA_GUARD_URL", "")):
        settings.save(guard_url=url)
        st.rerun()
    st.write(f"Main machine: `{settings.main_url()}`")

st.divider()
if st.button("Check nodes", type="primary"):
    for name, u in (("Main", settings.main_url()), ("Guard", settings.configured_guard_url())):
        st.markdown(f"**{name}**")
        if not u:
            st.warning("URL not set.")
            continue
        try:
            with st.spinner(f"Checking {name}…"):
                models = list_models(u, timeout=5.0)
            st.success(f"{name} reachable at `{u}`")
            st.write(", ".join(f"`{m}`" for m in models) if models else "Reachable, but no models are pulled.")
        except Exception as exc:
            st.error(f"{name} at `{u}` failed: {exc}")
