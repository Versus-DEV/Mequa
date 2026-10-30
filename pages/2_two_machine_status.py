"""Two-machine Ollama status page."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mequa.guard.remote_guard import list_models
from mequa.utils.env import load_env_file


load_env_file()

st.title("Two-Machine Status")
st.caption("Local network only. Do not expose Ollama to the internet.")

main_url = os.getenv("OLLAMA_MAIN_URL", "http://127.0.0.1:11434")
guard_url = os.getenv("OLLAMA_GUARD_URL", "")

st.write(f"Main URL: `{main_url}`")
st.write(f"Guard URL: `{guard_url or 'not set'}`")

if st.button("Check nodes", type="primary"):
    nodes = [
        ("MAIN", main_url),
        ("GUARD", guard_url),
    ]

    for name, url in nodes:
        st.subheader(name)

        if not url:
            st.error("URL not set.")
            continue

        st.write(f"URL: `{url}`")

        try:
            with st.spinner(f"Checking {name}..."):
                models = list_models(url, timeout=5.0)

            st.success(f"{name} reachable.")

            if models:
                st.write("Models:")
                for model in models:
                    st.write(f"- `{model}`")
            else:
                st.warning(" reachable, but no models found.")

        except Exception as exc:
            st.error(f"{name} failed: {exc}")