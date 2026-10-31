import json

from mequa import numpy_backend as nb
from mequa.agents.adaptive_investigator import AdaptiveInvestigator
from mequa.firewall import Verdict
from mequa.session import Session


def _chat_that_skips_sweep_until_told():
    calls = {"n": 0}

    def chat(messages, model, tools=None, **kw):
        calls["n"] += 1
        last_user = [m for m in messages if m["role"] == "user"][-1]["content"]
        tools_seen = [m for m in messages if m["role"] == "tool"]
        if "Call sweep_load" in last_user and not any(m.get("tool_name") == "sweep_load" for m in tools_seen):
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "sweep_load", "arguments": {"start": 0.01, "stop": 0.12, "steps": 6}}}]}}
        if not tools_seen:
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_solver", "arguments": {"load_fraction": 0.05}}}]}}
        res = json.loads(tools_seen[-1]["content"])
        txt = res.get("threshold_load")
        return {"message": {"role": "assistant", "content": f"Result {txt}" if txt else "Ran the solver once."}}

    return chat


def test_threshold_question_forces_a_sweep_when_the_model_skips_it():
    sess = Session(nb.synthetic_cells(200, seed=1, clusters=1), knn=8)
    ok = lambda *a, **k: Verdict(allowed=True)
    inv = AdaptiveInvestigator(sess, chat_fn=_chat_that_skips_sweep_until_told(), check_request=ok, check_reply=ok)
    res = inv.run("At what load does strain reach the limit?")
    assert any(s.get("tool") == "sweep_load" for s in res["trace"] if s["type"] == "tool_call")


def _chat_that_prints_tool_calls_as_text():
    def chat(messages, model, tools=None, **kw):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            return {"message": {"role": "assistant", "content": '{"name": "sweep_load", "parameters": {"start": 0.01, "stop": 0.12, "steps": 6}}'}}
        res = json.loads(tool_msgs[-1]["content"])
        return {"message": {"role": "assistant", "content": f"Threshold {res['threshold_load']}"}}

    return chat


def _chat_that_never_sweeps():
    def chat(messages, model, tools=None, **kw):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_solver", "arguments": {"load_fraction": 0.05}}}]}}
        sweep = [m for m in tool_msgs if m.get("tool_name") == "sweep_load"]
        if sweep:
            return {"message": {"role": "assistant", "content": f"Threshold {json.loads(sweep[-1]['content'])['threshold_load']}"}}
        return {"message": {"role": "assistant", "content": "I ran the solver once."}}

    return chat


def test_a_tool_call_printed_as_text_is_executed():
    sess = Session(nb.synthetic_cells(200, seed=1, clusters=1), knn=8)
    ok = lambda *a, **k: Verdict(allowed=True)
    res = AdaptiveInvestigator(sess, chat_fn=_chat_that_prints_tool_calls_as_text(), check_request=ok, check_reply=ok).run("Find the failure threshold")
    sweeps = [s for s in res["trace"] if s["type"] == "tool_call" and s["tool"] == "sweep_load"]
    assert sweeps and not sweeps[0].get("forced") and not res["forced_sweep"]


def test_a_model_that_never_sweeps_gets_a_labelled_forced_sweep():
    sess = Session(nb.synthetic_cells(200, seed=1, clusters=1), knn=8)
    ok = lambda *a, **k: Verdict(allowed=True)
    res = AdaptiveInvestigator(sess, chat_fn=_chat_that_never_sweeps(), check_request=ok, check_reply=ok).run("At what load does strain reach the limit?")
    forced = [s for s in res["trace"] if s.get("forced")]
    assert res["forced_sweep"] and len(forced) == 1
    assert any(d["event"].startswith("model ignored") for d in res["diagnostics"])
