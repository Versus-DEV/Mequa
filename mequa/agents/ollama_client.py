"""Base Ollama client for Mequa agents."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_TIMEOUT_SECONDS = 120.0


def get_main_url() -> str:
    return os.getenv("OLLAMA_MAIN_URL", "http://127.0.0.1:11434").rstrip("/")


def get_guard_url() -> str:
    return os.getenv("OLLAMA_GUARD_URL", get_main_url()).rstrip("/")


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
    """Generate text from Ollama."""
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
    """Chat with Ollama, optionally with tool definitions."""
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