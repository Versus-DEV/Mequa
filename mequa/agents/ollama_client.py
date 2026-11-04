"""Base client for Mequa agents (Ollama-compatible shim over the Meta client).

Signatures are unchanged so existing callers and test fakes keep working.
When LLM_PROVIDER=meta (the default), calls are forwarded to
mequa.agents.meta_client with legacy Ollama tags mapped to Meta models;
otherwise the original Ollama HTTP path is used.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_TIMEOUT_SECONDS = 120.0

_LEGACY_MODELS = {"llama-guard3:1b", "llama3.2:3b", "llama3.1:latest", "llama3.1:8b"}


def _use_meta() -> bool:
    from mequa.agents import meta_client

    return meta_client.provider() == "meta"


def _resolve_model(model: str) -> str:
    from mequa.agents import meta_client

    if not model or model in _LEGACY_MODELS:
        return meta_client.get_agent_model()
    return model


def get_main_url() -> str:
    if _use_meta():
        from mequa.agents import meta_client

        return meta_client.get_base_url()
    return os.getenv("OLLAMA_MAIN_URL", "http://127.0.0.1:11434").rstrip("/")


def get_guard_url() -> str:
    if _use_meta():
        from mequa.agents import meta_client

        return meta_client.get_base_url()
    from mequa import settings

    return settings.effective_guard_url()


def _post_json(
    url: str,
    payload: Dict[str, Any],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Send JSON to Ollama and return the response."""
    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"error": str(exc), "body": body}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        # Ollama stopped, wrong host, timeout, or a non-JSON reply: report it, do not crash the caller.
        return {"error": f"{type(exc).__name__}: {exc}", "body": ""}


def generate(
    prompt: str,
    model: str,
    base_url: Optional[str] = None,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Generate text from the configured LLM (Meta by default, Ollama legacy)."""
    if _use_meta():
        from mequa.agents import meta_client

        return meta_client.generate(
            prompt=prompt,
            model=_resolve_model(model),
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    base_url = base_url or get_main_url()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if system:
        payload["system"] = system

    url = base_url + "/api/generate"
    return _post_json(url, payload, timeout=timeout)


def chat(
    messages: List[Dict[str, str]],
    model: str,
    base_url: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Chat with the configured LLM, optionally with tool definitions."""
    if _use_meta():
        from mequa.agents import meta_client

        return meta_client.chat(
            messages=messages,
            model=_resolve_model(model),
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    base_url = base_url or get_main_url()

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if tools:
        payload["tools"] = tools

    url = base_url + "/api/chat"
    return _post_json(url, payload, timeout=timeout)