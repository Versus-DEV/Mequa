from types import SimpleNamespace

from mequa import export_chat

CHAT = [
    {"role": "user", "content": "Find the load at which p95 strain hits the limit."},
    {"role": "assistant", "content": "At 4.8 % load the p95 strain reaches 0.05 \u2014 verified."},
    {"role": "user", "content": "blocked question text"},
    {"role": "assistant", "content": "**COMPLIANCE HALT.** Mequa does not give clinical guidance"},
]
DS = SimpleNamespace(n=2000, dim=2, label="synthetic")


def test_blocked_turn_is_withheld():
    md = export_chat.build_markdown(CHAT, DS, "llama3.2:3b", 0.05)
    assert "blocked question text" not in md
    assert "withheld" in md
    assert "4.8 % load" in md


def test_prompt_respects_limit_and_keeps_header():
    p = export_chat.build_prompt(CHAT * 50, DS, "llama3.2:3b", 0.05, max_chars=1500)
    assert len(p) < 1700
    assert "Do not give diagnosis" in p


def test_pdf_is_valid():
    pdf = export_chat.build_pdf(export_chat.build_markdown(CHAT, DS))
    assert pdf.startswith(b"%PDF")


def test_empty_chat_still_exports():
    md = export_chat.build_markdown([], None)
    assert "Dataset: none loaded" in md
