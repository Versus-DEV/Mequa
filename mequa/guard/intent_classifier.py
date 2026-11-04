"""Intent classifier for research pivot (Meta model by default, Llama legacy)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from mequa.agents.ollama_client import generate, get_guard_url


def classify_intent(
    prompt: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Classify the user's intent as research, clinical, or other.

    Args:
        prompt: The user's prompt.
        model: Intent classifier model name.
        base_url: Ollama base URL (usually the guard node).
        timeout: Request timeout.

    Returns:
        Dict with intent classification.
    """
    if model is None:
        from mequa import firewall as _fw

        model = _fw.INTENT_MODEL
    base_url = base_url or get_guard_url()

    system_prompt = """You are an intent classifier for a biomechanics research tool.

Classify the user's intent into one of these categories:
- RESEARCH: The user wants to simulate tissue mechanics, analyze cell geometry, or study biomechanical properties.
- CLINICAL: The user is asking for medical diagnosis, treatment advice, or patient-specific guidance.
- OTHER: The request is unclear or unrelated.

Respond with exactly one word: RESEARCH, CLINICAL, or OTHER."""

    full_prompt = f"{system_prompt}\n\nUser request: {prompt}\n\nIntent:"

    response = generate(
        prompt=full_prompt,
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=32,
        timeout=timeout,
    )

    if "error" in response:
        return {
            "intent": "OTHER",
            "raw": "",
            "error": response["error"],
        }

    raw = str(response.get("response", "")).strip().upper()

    if "RESEARCH" in raw:
        intent = "RESEARCH"
    elif "CLINICAL" in raw:
        intent = "CLINICAL"
    else:
        intent = "OTHER"

    return {
        "intent": intent,
        "raw": raw,
        "model": model,
    }