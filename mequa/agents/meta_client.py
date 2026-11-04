"""Meta Model API client for Mequa agents (OpenAI-compatible).

Single HTTP owner for all LLM calls. Uses only the standard library
(urllib) so the runtime dependency set does not change.

Return shapes mirror ``mequa.agents.ollama_client`` so callers and test
fakes keep working:
  chat()     -> {"message": {"role", "content", "tool_calls": [...]}} or {"error": ...}
  generate() -> {"response": str} or {"error": ...}
  status()   -> {"up", "models", "url", ...}; never raises.

Tool calls are normalized to the Ollama-style envelope
``{"function": {"name": ..., "arguments": ...}}`` where ``arguments`` stays
a JSON-encoded string when the API returns one (the investigator's
``_parse_args`` accepts both dicts and strings, so either form works).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_AGENT_MODEL = "muse-spark-1.3"
DEFAULT_GUARD_MODEL = "muse-spark-1.3"
DEFAULT_PLANNER_MODEL = "muse-wonder"
DEFAULT_TIMEOUT_SECONDS = 60.0


def provider() -> str:
    """Active LLM provider name. Anything other than 'meta' means legacy Ollama."""
    return os.getenv("LLM_PROVIDER", "meta").strip().lower() or "meta"


def get_base_url() -> str:
    return os.getenv("META_BASE_URL", DEFAULT_BASE_URL).rstrip("/") or DEFAULT_BASE_URL


def get_agent_model() -> str:
    return os.getenv("META_AGENT_MODEL", DEFAULT_AGENT_MODEL).strip() or DEFAULT_AGENT_MODEL


def get_guard_model() -> str:
    return os.getenv("META_GUARD_MODEL", DEFAULT_GUARD_MODEL).strip() or DEFAULT_GUARD_MODEL


def get_planner_model() -> str:
    return os.getenv("META_PLANNER_MODEL", DEFAULT_PLANNER_MODEL).strip() or DEFAULT_PLANNER_MODEL


def get_timeout(default: float = DEFAULT_TIMEOUT_SECONDS) -> float:
    try:
        return float(os.getenv("META_TIMEOUT", str(default)))
    except (TypeError, ValueError):
        return default


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("META_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST JSON and return the decoded body. Never raises: errors come back as {"error": ...}."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        return {"error": f"{type(exc).__name__}: {exc}", "body": detail}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        # No key, no network, timeout, or a non-JSON reply: report it, do not crash the caller.
        return {"error": f"{type(exc).__name__}: {exc}", "body": ""}


def _normalize_tool_calls(raw_calls: Any) -> List[Dict[str, Any]]:
    """Map OpenAI-style tool_calls to the Ollama-style envelope the investigator reads.

    OpenAI returns arguments as a JSON-encoded string; keep it as-is because
    ``AdaptiveInvestigator`` parses ``fn["arguments"]`` with ``_parse_args``,
    which accepts both dicts and strings.
    """
    calls: List[Dict[str, Any]] = []
    if not isinstance(raw_calls, list):
        return calls
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        calls.append({"function": {"name": fn["name"], "arguments": fn.get("arguments", {})}})
    return calls


def _first_choice(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        choices = data.get("choices", [])
        if choices and isinstance(choices[0], dict):
            return choices[0].get("message", {}) or {}
    except Exception:
        pass
    return {}


def chat(
    messages: List[Dict[str, str]],
    model: str,
    base_url: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Chat with the Meta Model API (OpenAI-compatible /chat/completions)."""
    if not os.getenv("META_API_KEY", "").strip():
        return {"error": "ValueError: META_API_KEY is not set", "body": ""}
    base = (base_url or get_base_url()).rstrip("/")
    wait = get_timeout() if timeout is None else timeout
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    data = _post_json(base + "/chat/completions", payload, timeout=wait)
    if "error" in data:
        return {"error": data["error"], "body": data.get("body", "")}
    msg = _first_choice(data)
    if not msg:
        return {"error": "ValueError: empty choices in Meta response", "body": json.dumps(data)[:500]}
    content = msg.get("content") or ""
    return {
        "message": {
            "role": msg.get("role", "assistant"),
            "content": content if isinstance(content, str) else json.dumps(content),
            "tool_calls": _normalize_tool_calls(msg.get("tool_calls")),
        }
    }


def generate(
    prompt: str,
    model: str,
    base_url: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Single-turn generation, mapped onto /chat/completions for reliability."""
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    out = chat(
        messages=msgs,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if "error" in out:
        return out
    return {"response": out["message"].get("content", "")}


def status(timeout: float = 5.0) -> Dict[str, Any]:
    """Is the Meta API reachable? Never raises (used by health cards)."""
    base = get_base_url()
    if not os.getenv("META_API_KEY", "").strip():
        return {"up": False, "url": base, "models": [], "error": "no META_API_KEY"}
    request = urllib.request.Request(base + "/models", headers=_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        models: List[str] = []
        raw = data.get("data", data.get("models", []))
        if isinstance(raw, list):
            for m in raw:
                if isinstance(m, dict) and m.get("id"):
                    models.append(str(m["id"]))
                elif isinstance(m, str):
                    models.append(m)
        return {"up": True, "url": base, "models": sorted(models)}
    except Exception as exc:  # health probe: report, never raise
        return {"up": False, "url": base, "models": [], "error": f"{type(exc).__name__}"}
