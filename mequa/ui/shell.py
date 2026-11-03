"""The workspace chrome around every page: status rail at the bottom and the AI chat dock on the right.

app.py calls render() once per run, before the active page. The dock is fixed to the viewport; the
page scrolls in the space left over. Chat history lives in st.session_state["chat"].
"""

from __future__ import annotations

import html
import os

import streamlit as st

from mequa import chat as chat_mod
from mequa import export_chat
from mequa import settings, workspace
from mequa.ui import theme


@st.cache_data(ttl=5, show_spinner=False)
def _ollama_cached(url: str):
    return chat_mod.ollama_status(url)


def _guard_label(g: dict) -> tuple[str, str]:
    """(css state, short text) for the guard indicator."""
    if g["mode"] == "main":
        return "ok", "Main machine only"
    if g["on_main"] and not g["fallback"]:
        return "ok", "Main machine (no guard set)"
    if g["fallback"]:
        return "warn", "Main (guard offline)"
    return ("ok", "Guard machine") if g["guard_up"] else ("bad", "Guard offline")


def _top_chrome(main: dict, phase: str = "post") -> None:
    ds = workspace.current(st.session_state)
    g = settings.guard_status()
    gs, gt = _guard_label(g)
    dsx = html.escape(f"{ds.n} cells · {ds.dim}D" if ds else "no dataset")
    st.markdown(
        f'<div class="mq-brand {phase}">Mequa<small>research workspace</small></div>'
        f'<div class="mq-chip {phase}"><span><span class="led {"ok" if main["up"] else "bad"}"></span>Ollama {"online" if main["up"] else "offline"}</span>'
        f'<span class="sep">|</span><span><span class="led {gs}"></span>{html.escape(gt)}</span>'
        f'<span class="sep">|</span><span>{dsx}</span></div>', unsafe_allow_html=True)


def _status_bar(main: dict, phase: str = "post") -> None:
    ds = workspace.current(st.session_state)
    g = settings.guard_status()
    gs, gt = _guard_label(g)
    parts = [
        '<span class="sp"></span>',
        '<span>Research use only. Not for clinical decisions.</span>',
        f'<span>Mequa {theme.VERSION}</span>',
    ]
    st.markdown(f'<div class="mq-status {phase}">' + "".join(parts) + "</div>", unsafe_allow_html=True)


def guard_mode_selector(key: str) -> None:
    """Radio for the three guard modes; persists to config/mequa_settings.json."""
    labels = {"auto": "Auto: guard machine if reachable, else main",
              "dedicated": "Guard machine only (strict)",
              "main": "Main machine does everything"}
    cur = settings.guard_mode()
    pick = st.radio("Who runs the safety firewall?", list(labels), index=list(labels).index(cur) if cur in labels else 0,
                    format_func=labels.get, key=key)
    if pick != cur:
        settings.save(guard_mode=pick)
        st.rerun()


_PIN_JS = """
(function(){
  const w = window.parent && window.parent.document ? window.parent : window;
  if (w.__mqPin) return; w.__mqPin = true;
  const d = w.document;
  const pin = () => { const s = d.querySelector('.st-key-chat_scroll'); if (s) s.scrollTop = s.scrollHeight; };
  new w.MutationObserver(() => { clearTimeout(w.__mqPinT); w.__mqPinT = setTimeout(pin, 60); }).observe(d.body, {childList:true, subtree:true});
  pin();
})();
"""


def _pin_chat_to_bottom() -> None:
    """Scroll the chat to its newest message whenever it changes. One observer per browser tab."""
    try:
        st.html(f"<script>{_PIN_JS}</script>", unsafe_allow_javascript=True)
    except TypeError:  # older Streamlit without script support in st.html
        import streamlit.components.v1 as components
        components.html(f"<script>{_PIN_JS}</script>", height=0)


def _chat_dock(main: dict) -> None:
    ss = st.session_state
    ss.setdefault("chat", [])
    ss.setdefault("chat_model", os.getenv("PLANNER_MODEL", "llama3.2:3b"))
    ss.setdefault("chat_limit", 0.05)
    ds = workspace.current(ss)

    with st.container(key="chat_dock"):
        _pin_chat_to_bottom()
        with st.container(key="chat_head"):
            left, mid, right = st.columns([6, 2, 2], vertical_alignment="center")
            left.markdown(f'<div class="mq-chat-title">Assistant<i></i></div><div class="mq-chat-sub">{html.escape(ss["chat_model"])} · '
                          f'{"%d cells" % ds.n if ds else "no dataset"}</div>', unsafe_allow_html=True)
            with mid.popover("⚙", help="Chat settings"):
                models = main["models"] or [ss["chat_model"]]
                if ss["chat_model"] not in models:
                    models = [ss["chat_model"]] + models
                ss["chat_model"] = st.selectbox("Model", models, index=models.index(ss["chat_model"]))
                ss["chat_limit"] = st.number_input("Strain limit (p95)", 0.005, 0.5, float(ss["chat_limit"]), 0.005,
                                                   help="A modelling choice you set, not a measured tissue property.")
                guard_mode_selector("guard_mode_chat")
                if st.button("Clear conversation", width="stretch"):
                    ss["chat"] = []
                    st.rerun()
                st.markdown("**Export for Meta AI**")
                has_chat = bool(ss["chat"])
                md_text = export_chat.build_markdown(ss["chat"], ds, ss["chat_model"], ss["chat_limit"]) if has_chat else ""
                st.download_button("Download PDF", export_chat.build_pdf(md_text) if has_chat else b"",
                                   file_name="mequa_chat.pdf", mime="application/pdf",
                                   disabled=not has_chat, width="stretch", key="export_pdf")
                st.download_button("Download Markdown", md_text, file_name="mequa_chat.md",
                                   mime="text/markdown", disabled=not has_chat, width="stretch", key="export_md")
                if has_chat:
                    with st.expander("Paste-ready text"):
                        st.code(export_chat.build_prompt(ss["chat"], ds, ss["chat_model"], ss["chat_limit"]), language=None)
            if right.button("✕", help="Hide chat", key="chat_hide"):
                ss["chat_open"] = False
                st.rerun()

        with st.container(key="chat_scroll"):
            with st.container(key="chat_msgs"):
                if not ss["chat"]:
                    st.markdown('<div class="mq-empty">Ask a question about the dataset in your workspace.<br><br>'
                                'For example: <em>At what load does this sample reach the strain limit?</em> Type <code>/help</code> for commands.<br><br>'
                                'Answers come from solver output and are checked by the firewall.</div>', unsafe_allow_html=True)
                for m in ss["chat"]:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])
                pending = ss.pop("chat_pending", None)
                if pending:
                    with st.chat_message("assistant"):
                        with st.spinner("Working on it…"):
                            res = chat_mod.answer(pending, ds, model=ss["chat_model"], strain_limit=ss["chat_limit"], main_status=main)
                        st.markdown(res["text"])
                    ss["chat"].append({"role": "assistant", "content": res["text"]})

        with st.container(key="chat_input"):
            prompt = st.chat_input("Ask about this dataset…", key="chat_prompt")
        if prompt:
            ss["chat"].append({"role": "user", "content": prompt})
            ss["chat_pending"] = prompt
            st.rerun()


def _theme_switch() -> None:
    """Light / Green-and-black switch, pinned in the top bar. The choice is saved for next time."""
    ss = st.session_state
    ss.setdefault("theme_mode", settings.load().get("theme", "light"))
    with st.container(key="theme_switch"):
        pick = st.segmented_control("Theme", list(theme.THEME_LABELS), default=ss["theme_mode"], format_func=theme.THEME_LABELS.get,
                                    label_visibility="collapsed", key="theme_pick")
    if pick and pick != ss["theme_mode"]:
        ss["theme_mode"] = pick
        settings.save(theme=pick)
        st.rerun()


def _dataset_signature(ds) -> tuple:
    if ds is None:
        return ()
    head = ds.points[:64]
    return (ds.n, ds.dim, ds.label, ds.source, round(float(head.sum()), 6))


def refresh_if_dataset_changed() -> None:
    """The chat dock is drawn before the page runs. If the page then published a different dataset, rerun once so the dock sees it."""
    ss = st.session_state
    now = _dataset_signature(workspace.current(ss))
    if now != ss.get("_dock_sig"):
        ss["_dock_sig"] = now
        st.rerun()


def render_status(phase: str = "post") -> None:
    """Top chip and bottom rail. Drawn once before the page ("pre", so they exist even if the page calls st.stop(), after which Streamlit
    accepts no more output) and again after it ("post", showing what the page just published; CSS hides the "pre" copy)."""
    main = _ollama_cached(os.getenv("OLLAMA_MAIN_URL") or "http://127.0.0.1:11434")
    _top_chrome(main, phase)
    _status_bar(main, phase)


def render() -> None:
    ss = st.session_state
    ss.setdefault("chat_open", True)
    _theme_switch()
    st.session_state["_dock_sig"] = _dataset_signature(workspace.current(st.session_state))
    theme.apply(chat_open=ss["chat_open"])
    main = _ollama_cached(os.getenv("OLLAMA_MAIN_URL") or "http://127.0.0.1:11434")
    if ss["chat_open"]:
        _chat_dock(main)
    else:
        with st.container(key="chat_open"):
            if st.button("Assistant ▸", key="chat_show", help="Show the chat dock"):
                ss["chat_open"] = True
                st.rerun()
