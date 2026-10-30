import json

import numpy as np

from mequa import agent, firewall
from mequa import numpy_backend as nb


def fake_chat(model, messages, timeout=300, fmt=None, num_predict=None, node="main"):
    last = messages[-1]["content"].lower()
    if model == firewall.GUARD_MODEL:
        if "treat" in last:
            return "unsafe\nS6"
        if "load" in last:
            return "unsafe\nS1"
        return "safe"
    if fmt == "json" and "Request:" in messages[-1]["content"]:
        return json.dumps({"zones": 3, "neighbors": 6, "load": 0.05, "reason": "test"})
    if fmt == "json":
        return json.dumps({"label": "clinical_advice" if "lump" in last else "research_analysis"})
    return "Summary from numbers."


def patch(monkeypatch):
    monkeypatch.setattr(firewall, "_ollama_chat", fake_chat)
    monkeypatch.setattr(agent, "_ollama_chat", fake_chat)


def test_s6_blocks_and_other_categories_only_flag(monkeypatch):
    patch(monkeypatch)
    blocked = firewall.check_request("How should I treat this tumor")
    assert not blocked.allowed and "llama_guard" in blocked.blocked_by
    flagged = firewall.check_request("Run the strain relaxation with a 5 percent load")
    assert flagged.allowed and flagged.flags == ["S1"]


def test_intent_layer_catches_indirect_requests(monkeypatch):
    patch(monkeypatch)
    verdict = firewall.check_request("Is this lump on my neck worrying?")
    assert not verdict.allowed
    assert "llama_intent" in verdict.blocked_by or "rules" in verdict.blocked_by


def test_unreachable_models_are_marked_degraded(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("down")

    monkeypatch.setattr(firewall, "_ollama_chat", boom)
    verdict = firewall.check_request("Cluster the cells")
    assert verdict.degraded
    reply = firewall.check_reply("q", "a")
    assert not reply.allowed and reply.blocked_by == ["guard_unreachable"]


def test_agent_ok_and_blocked(monkeypatch):
    patch(monkeypatch)
    be = type("B", (), {"name": "numpy", "analyze": staticmethod(nb.analyze), "synthetic_cells": staticmethod(nb.synthetic_cells)})
    ok = agent.run_agent("Cluster the cells into density zones", backend=be)
    assert ok["status"] == "ok" and ok["facts"]["cells"] == 2000
    assert len(ok["facts"]["per_zone"]) == 3
    blocked = agent.run_agent("How should I treat this tumor", backend=be)
    assert blocked["status"] == "blocked"


def test_critic_flags_tiny_and_dominant_zones():
    n = 1000
    labels = np.array([1] * 10 + [2] * 980 + [3] * 10)
    result = {
        "labels": labels,
        "node_strain": np.zeros(n),
        "failure_prob": np.zeros(n),
        "sphi": 1.0,
    }
    issues, zones = agent.critique(result, n)
    assert any("under 5 percent" in i for i in issues)
    assert any("over 70 percent" in i for i in issues)
    assert len(zones) == 3
