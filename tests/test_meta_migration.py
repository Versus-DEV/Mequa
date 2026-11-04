"""Focused regression tests for the Meta migration (steps 1-2).

Covers: strict-JSON guard parsing with fail-closed behavior, legacy
Llama Guard text compatibility (existing monkeypatched tests rely on it),
legacy model-tag resolution, and OpenAI-style string tool-call arguments.
"""
import json

from mequa import firewall
from mequa.agents import meta_client
from mequa.agents.adaptive_investigator import _parse_args


def test_guard_json_blocked_and_allowed():
    unsafe, cats = firewall._parse_guard_output('{"blocked": true, "category": "S6", "reason": "dose"}')
    assert unsafe and cats == ["S6"]
    unsafe, cats = firewall._parse_guard_output('{"blocked": false, "category": "none", "reason": "ok"}')
    assert not unsafe and cats == []


def test_guard_legacy_text_still_works():
    unsafe, cats = firewall._parse_guard_output("unsafe\nS6")
    assert unsafe and cats == ["S6"]
    unsafe, cats = firewall._parse_guard_output("unsafe\nS1")
    assert unsafe and cats == ["S1"]
    unsafe, _ = firewall._parse_guard_output("safe")
    assert not unsafe


def test_guard_garbage_fails_closed():
    for raw in ("", "maybe", '{"blocked": "yes"}', '{"nope": 1}', "not json at all"):
        unsafe, cats = firewall._parse_guard_output(raw)
        assert unsafe and cats == ["S6"]


def test_guard_check_sends_system_prompt_and_json(monkeypatch):
    seen = {}

    def fake(model, messages, timeout=300, fmt=None, num_predict=None, node="main"):
        seen.update(model=model, messages=messages, fmt=fmt, node=node)
        assert messages[0]["role"] == "system"
        return '{"blocked": false, "category": "none", "reason": "research"}'

    monkeypatch.setattr(firewall, "_ollama_chat", fake)
    unsafe, cats, _ = firewall.guard_check("Cluster the cells")
    assert not unsafe and cats == []
    assert seen["fmt"] == "json" and seen["node"] == "guard"


def test_guard_check_fail_closed_on_garbage(monkeypatch):
    monkeypatch.setattr(
        firewall, "_ollama_chat",
        lambda *a, **k: "the model rambled without JSON",
    )
    verdict = firewall.check_request("Cluster the cells", use_intent=False)
    assert not verdict.allowed and "llama_guard" in verdict.blocked_by


def test_resolve_model_maps_legacy_and_passes_muse(monkeypatch):
    monkeypatch.setenv("META_GUARD_MODEL", "muse-spark-1.3")
    monkeypatch.setenv("META_AGENT_MODEL", "muse-spark-1.3")
    assert firewall._resolve_model("llama-guard3:1b", "guard") == "muse-spark-1.3"
    assert firewall._resolve_model("llama3.2:3b", "main") == "muse-spark-1.3"
    assert firewall._resolve_model("muse-wonder", "main") == "muse-wonder"


def test_string_tool_arguments_survive():
    # OpenAI-compatible APIs return arguments as a JSON string; the
    # investigator must still execute the call.
    calls = meta_client._normalize_tool_calls(
        [{"type": "function", "function": {"name": "run_solver", "arguments": '{"load_fraction": 0.05}'}}]
    )
    assert calls[0]["function"]["name"] == "run_solver"
    assert _parse_args(calls[0]["function"]["arguments"]) == {"load_fraction": 0.05}


def test_meta_chat_posts_completions_with_bearer(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "hi"}}]
            }).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["payload"] = json.loads(req.data.decode())
        return FakeResp()

    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("META_BASE_URL", "https://api.meta.ai/v1")
    monkeypatch.setattr(meta_client.urllib.request, "urlopen", fake_urlopen)
    out = meta_client.chat(messages=[{"role": "user", "content": "hi"}], model="muse-spark-1.3")
    assert out["message"]["content"] == "hi"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer test-key"
    assert captured["payload"]["model"] == "muse-spark-1.3"
