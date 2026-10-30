import numpy as np

from mequa import numpy_backend as nb
from mequa.agents.adaptive_investigator import AdaptiveInvestigator, _parse_args
from mequa.agents.number_grounding import check_number_grounding, template_summary, ungrounded_numbers
from mequa.firewall import Verdict
from mequa.session import Session


def allow(*a, **k):
    return Verdict(allowed=True)


def make_session():
    pts = nb.synthetic_cells(300, seed=5, clusters=1)
    return Session(pts, knn=8)


class Scripted:
    """Stands in for Ollama: returns scripted messages and records what it was sent."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, model, tools=None, **kw):
        self.seen.append([dict(m) for m in messages])
        return {"message": self.replies.pop(0)}


def call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_arguments_may_be_object_or_json_string():
    assert _parse_args({"a": 1}) == {"a": 1}
    assert _parse_args('{"a": 1}') == {"a": 1}
    assert _parse_args("not json") == {} and _parse_args(None) == {}


def test_loop_calls_solver_keeps_history_and_grounds_the_answer():
    sess = make_session()
    expected = sess.run(0.05)
    chat = Scripted([
        call("run_solver", {"load_fraction": 0.05}),
        {"role": "assistant", "content": f"At load 0.05 the SPHI is {expected['sphi']} and max strain is {expected['max_strain']}."},
    ])
    inv = AdaptiveInvestigator(sess, chat_fn=chat, check_request=allow, check_reply=lambda q, t: Verdict(allowed=True))
    out = inv.run("What is the strain at 5 percent load?")
    assert out["status"] == "ok" and not out["grounding"]["fallback"]
    # the second model call must contain the assistant's tool call AND the tool result
    roles = [m["role"] for m in chat.seen[1]]
    assert roles[-2:] == ["assistant", "tool"] and "tool_calls" in chat.seen[1][-2]


def test_invented_number_is_rejected_then_rewritten():
    sess = make_session()
    r = sess.run(0.05)
    chat = Scripted([
        call("run_solver", {"load_fraction": 0.05}),
        {"role": "assistant", "content": "The SPHI is 99.123 under this load."},          # invented
        {"role": "assistant", "content": f"The SPHI is {r['sphi']} under this load."},     # fixed
    ])
    out = AdaptiveInvestigator(sess, chat_fn=chat, check_request=allow, check_reply=lambda q, t: Verdict(allowed=True)).run("q")
    assert out["status"] == "ok" and not out["grounding"]["fallback"]
    assert out["grounding"]["attempts"][0]["ungrounded"] == ["99.123"]
    assert out["grounding"]["attempts"][1]["ungrounded"] == []


def test_persistent_invention_falls_back_to_template():
    sess = make_session()
    chat = Scripted([call("run_solver", {"load_fraction": 0.05})] + [{"role": "assistant", "content": "SPHI is 4242.42."}] * 3)
    out = AdaptiveInvestigator(sess, chat_fn=chat, check_request=allow, check_reply=lambda q, t: Verdict(allowed=True)).run("q")
    assert out["grounding"]["fallback"] is True and "4242.42" not in out["summary"]


def test_sweep_tool_reports_threshold_or_none():
    sess = make_session()
    inv = AdaptiveInvestigator(sess, strain_limit=0.05, chat_fn=None, check_request=allow, check_reply=allow)
    res = inv.execute("sweep_load", {"start": 0.01, "stop": 0.12, "steps": 5})
    assert len(res["rows"]) == 5 and res["strain_limit"] == 0.05
    assert res["threshold_load"] is None or 0.01 <= res["threshold_load"] <= 0.12


def test_bad_tool_input_is_clamped_or_reported_not_raised():
    sess = make_session()
    inv = AdaptiveInvestigator(sess, chat_fn=None, check_request=allow, check_reply=allow)
    assert "error" in inv.execute("nope", {})
    assert "error" in inv.execute("sweep_load", {"start": 0.1, "stop": 0.05, "steps": 5})
    assert inv.execute("run_solver", {"load_fraction": 99})["load_fraction"] == 0.2  # clamped


def test_blocked_request_never_reaches_the_model():
    def blocked(text):
        return Verdict(allowed=False, blocked_by=["llama_intent"])

    chat = Scripted([])
    out = AdaptiveInvestigator(make_session(), chat_fn=chat, check_request=blocked, check_reply=allow).run("treat my patient")
    assert out["status"] == "blocked" and chat.seen == []


def test_model_unavailable_is_reported():
    def down(**kw):
        return {"error": "URLError: connection refused"}

    out = AdaptiveInvestigator(make_session(), chat_fn=down, check_request=allow, check_reply=allow).run("q")
    assert out["status"] == "model_unavailable"


def test_grounding_accepts_rounding_percent_and_zone_ids_but_not_invention():
    facts = {"sphi": 0.123456, "load": 0.05, "per_zone": [{"zone": 1, "mean_strain": 0.0123}, {"zone": 2, "mean_strain": 0.04}]}
    assert ungrounded_numbers("SPHI is 0.123 at 5 percent load; zone 2 is 0.04.", facts, [1, 2]) == []
    assert ungrounded_numbers("SPHI is 0.5.", facts, [1, 2]) == ["0.5"]
    assert check_number_grounding("SPHI is 0.13.", [facts])["pass"] is False  # 0.123456 rounds to 0.12, not 0.13
    assert ungrounded_numbers("the 90th percentile is high", facts) == [] and ungrounded_numbers("a 91th percentile", facts) == ["91"]
    assert check_number_grounding("SPHI is 0.123.", [facts])["pass"] is True


def test_template_uses_only_facts():
    facts = {"cells": 10, "zones": 2, "mechanical_links": 30, "load_fraction": 0.05, "sphi": 1.5, "mean_strain": 0.01,
             "max_strain": 0.1, "s_crit": 0.05, "per_zone": [{"zone": 1, "mean_strain": 0.02}, {"zone": 2, "mean_strain": 0.01}]}
    assert ungrounded_numbers(template_summary(facts), facts, [1, 2]) == []
