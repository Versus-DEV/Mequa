"""Chat export for Mequa: Markdown, PDF and a paste-ready prompt for an outside assistant such as Meta AI."""

from __future__ import annotations

import datetime as dt
import textwrap
from typing import Any, Dict, List, Optional, Tuple

HALT = "**COMPLIANCE HALT.**"

PREAMBLE = (
    "This is an export from Mequa, a research workbench for tissue-mechanics datasets. "
    "Every number in Mequa's replies was produced by its verified solver and checked against the solver output; "
    "none were estimated by a language model. The data are research cell positions, not patient data. "
    "Please help me understand or critique the research results. Do not give diagnosis, prognosis or "
    "treatment advice for any individual."
)

_ASCII = {
    "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00b7": "-", "\u2026": "...", "\u2022": "-", "\u2248": "~", "\u00d7": "x", "\u2192": "->",
}


def _turns(chat: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(chat):
        role = chat[i].get("role")
        text = str(chat[i].get("content", ""))
        nxt = chat[i + 1] if i + 1 < len(chat) else None
        if role == "user" and nxt is not None and str(nxt.get("content", "")).startswith(HALT):
            out.append(("note", "A request and its reply were withheld: blocked by the Mequa compliance firewall."))
            i += 2
            continue
        if role == "assistant" and text.startswith(HALT):
            out.append(("note", "A reply was withheld by the Mequa compliance firewall."))
        else:
            out.append((str(role), text))
        i += 1
    return out


def build_markdown(chat: List[Dict[str, Any]], dataset: Optional[Any] = None, model: str = "",
                   strain_limit: Optional[float] = None) -> str:
    lines = ["# Mequa chat export", "", f"Exported: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
             "## Context for the assistant reading this", "", PREAMBLE, "", "## Setup", ""]
    if dataset is not None:
        lines.append(f"- Dataset: {dataset.n} cells, {dataset.dim}D, {dataset.label}")
    else:
        lines.append("- Dataset: none loaded")
    if model:
        lines.append(f"- Model: {model}")
    if strain_limit is not None:
        lines.append(f"- Strain limit (p95): {strain_limit} (a modelling choice, not a measured tissue property)")
    lines += ["", "## Conversation", ""]
    for kind, text in _turns(chat):
        if kind == "note":
            lines += [f"_{text}_", ""]
        else:
            lines += [f"### {'Researcher' if kind == 'user' else 'Mequa'}", "", text.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def build_prompt(chat: List[Dict[str, Any]], dataset: Optional[Any] = None, model: str = "",
                 strain_limit: Optional[float] = None, max_chars: int = 6000) -> str:
    body = build_markdown(chat, dataset, model, strain_limit)
    if len(body) > max_chars:
        head, _, tail = body.partition("## Conversation")
        keep = max(0, max_chars - len(head) - 60)
        tail = tail[-keep:] if keep else ""
        j = tail.find("###")
        tail = tail[j:] if j >= 0 else tail
        body = head + "## Conversation (most recent part only)\n\n" + tail
    return body + "\nMy question: "


def _latin(text: str) -> str:
    for k, v in _ASCII.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(markdown_text: str) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    for raw in markdown_text.splitlines():
        line = _latin(raw).replace("**", "").replace("`", "")
        if not line.strip():
            pdf.ln(4)
            continue
        if line.startswith("### "):
            size, style, line = 11, "B", line[4:]
        elif line.startswith("## "):
            size, style, line = 13, "B", line[3:]
        elif line.startswith("# "):
            size, style, line = 16, "B", line[2:]
        else:
            size, style = 10, ""
        pdf.set_font("Helvetica", style, size)
        for part in textwrap.wrap(line, 90, break_long_words=True) or [""]:
            pdf.multi_cell(0, size * 0.55 + 1, part, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
