"""Compliance page: shows the firewall decision for a request and, when blocked, generic research suggestions."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa.guard.research_pivot import pivot_to_research
from mequa.utils.env import load_env_file

from mequa import ui  # noqa: E402

load_env_file()

ui.page_header('Compliance Firewall', 'Three layers on every request: rules, Llama Guard (blocks on S6) and a Llama intent classifier.')

with st.sidebar:
    st.header("Request")
    prompt = st.text_area(
        "Type a request",
        value="Cluster the cells into density zones and report strain under a 5 percent load.",
        height=140,
    )
    check = st.button("Check Request", type="primary")

if check:
    with st.spinner("Running the firewall layers..."):
        try:
            st.session_state["fw"] = pivot_to_research(prompt)
        except OSError as exc:
            st.session_state["fw"] = {"decision": "error", "message": f"Ollama is not reachable: {exc}"}

result = st.session_state.get("fw")
if result is None:
    st.info("Enter a request and click Check request.")
    st.stop()

if result["decision"] == "allowed":
    st.success("Allowed: research request.")
    if result.get("degraded"):
        st.warning("A layer was unreachable, so this decision used fewer than three layers.")
    if result.get("flags"):
        st.caption(f"Flagged by Llama Guard (not blocking): {result['flags']}")
elif result["decision"] == "blocked":
    st.error("BLOCKED. " + result["message"])
    st.caption(f"Blocked by: {', '.join(result['blocked_by'])}")
    st.write("Research questions Mequa can answer on a dataset:")
    for s in result["suggestions"]:
        st.code(s, language=None)
else:
    st.error(result["message"])
