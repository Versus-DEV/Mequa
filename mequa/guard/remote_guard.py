"""Client for the guard model (Meta Model API by default, remote Ollama node legacy).

This module is intentionally dependency-light.
It uses only Python standard library networking.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List


DEFAULT_TIMEOUT_SECONDS = 30.0


def _use_meta() -> bool:
    from mequa.agents import meta_client

    return meta_client.provider() == "meta"


def get_base_url(env_name: str = "OLLAMA_GUARD_URL", default: str = "http://127.0.0.1:11434") -> str:
    if _use_meta():
        from mequa.agents import meta_client

        return meta_client.get_base_url()
    return os.getenv(env_name, default).rstrip("/")


def _get_json(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        return json.loads(body)


def _post_json(
    url: str,
    payload: Dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except Exception:
            return {
                "error": str(exc),
                "body": body,
            }


def ping_ollama(base_url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Check whether the model API is reachable. Raises on failure (callers treat it as down)."""
    if _use_meta():
        from mequa.agents import meta_client

        out = meta_client.status(timeout=timeout)
        if not out.get("up"):
            raise OSError(out.get("error", "Meta API unreachable"))
        return {"models": [{"name": m} for m in out.get("models", [])]}
    url = base_url.rstrip("/") + "/api/tags"
    return _get_json(url, timeout=timeout)


def list_models(base_url: str, timeout: float = 10.0) -> List[str]:
    """List models available on the model API."""
    response = ping_ollama(base_url, timeout=timeout)
    models = response.get("models", [])
    names = []
    for model in models:
        if isinstance(model, dict):
            names.append(str(model.get("name", "")))
        else:
            names.append(str(model))
    return [n for n in names if n]


def classify_prompt(
    prompt: str,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Send a prompt to the guard model and return a simple safe/unsafe decision.

    This is a practical hackathon version. For the final firewall evaluation,
    keep the measured round-based evaluation script as the source of truth.
    """
    if _use_meta():
        from mequa.agents import meta_client

        model = model or meta_client.get_guard_model()
        out = meta_client.generate(prompt=prompt, model=model, temperature=0.0, max_tokens=32, timeout=timeout)
        if "error" in out:
            return {"decision": "unsafe", "raw": "", "model": model,
                    "base_url": (base_url or get_base_url()).rstrip("/"), "error": out["error"]}
        raw = str(out.get("response", "")).strip().lower()
        return {
            "decision": "unsafe" if "unsafe" in raw else ("safe" if "safe" in raw else "unsafe"),
            "raw": raw,
            "model": model,
            "base_url": (base_url or get_base_url()).rstrip("/"),
        }
    base_url = (base_url or get_base_url()).rstrip("/")
    model = model or os.getenv("GUARD_MODEL", "llama-guard3:1b")

    prompt_text = f"""
You are a safety classifier for a research biomechanics simulation tool.

Decide whether the user request is allowed as RESEARCH or should be blocked because it asks for clinical, diagnostic, treatment, or individual medical advice.

Answer with exactly one word:
safe
or
unsafe

User request:
{prompt}
""".strip()

    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 32,
        },
    }

    url = base_url + "/api/generate"
    response = _post_json(url, payload, timeout=timeout)

    if "error" in response:
        # Fail closed: if Guard cannot be reached, treat as blocked.
        return {
            "decision": "unsafe",
            "raw": "",
            "model": model,
            "base_url": base_url,
            "error": response.get("error"),
        }

    raw = str(response.get("response", "")).strip().lower()

    if not raw:
        decision = "unsafe"
    elif "unsafe" in raw:
        decision = "unsafe"
    elif "safe" in raw:
        decision = "safe"
    else:
        # Fail closed if output is unexpected.
        decision = "unsafe"

    return {
        "decision": decision,
        "raw": raw,
        "model": model,
        "base_url": base_url,
    }