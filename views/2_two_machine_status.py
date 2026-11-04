"""Model API: where the language model runs, and check the connection."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import settings, ui  # noqa: E402
from mequa.agents import meta_client  # noqa: E402
from mequa.ui import shell  # noqa: E402
from mequa.utils.env import load_env_file  # noqa: E402

load_env_file()
ui.page_header("Model API", "The assistant, firewall guard and block-spec writer run on the Meta Model API.", eyebrow="System")

g = settings.guard_status()
state = "ok" if g["ok"] else "bad"
st.markdown(ui.pill(g["note"], state), unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("### Firewall mode")
    shell.guard_mode_selector("guard_mode_page")
    st.caption("The firewall (rules, Meta guard, intent classifier) always runs. This only decides on which machine "
               "the legacy Ollama path uses. If the chosen guard cannot be reached, requests are refused rather "
               "than answered unchecked.")
with right:
    st.markdown("### Connection")
    st.write(f"Provider: `{meta_client.provider()}`")
    st.write(f"Base URL: `{meta_client.get_base_url()}`")
    import os

    key_set = bool(os.getenv("META_API_KEY", "").strip())
    st.write(f"API key: `{'set' if key_set else 'NOT SET — export META_API_KEY'}`")
    st.write(f"Agent model: `{meta_client.get_agent_model()}`")
    st.write(f"Guard model: `{meta_client.get_guard_model()}`")
    st.write(f"Planner model: `{meta_client.get_planner_model()}`")

st.divider()
if st.button("Check connection", type="primary"):
    with st.spinner("Checking…"):
        out = meta_client.status(timeout=10.0)
    if out.get("up"):
        st.success(f"Reachable at `{out['url']}`")
        st.write(", ".join(f"`{m}`" for m in out.get("models", [])) if out.get("models") else "Reachable.")
    else:
        st.error(f"Failed: `{out.get('error', 'unreachable')}` — check META_API_KEY and META_BASE_URL.")
