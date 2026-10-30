"""Llama translates plain English into Simulink/Xcos block specifications."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from mequa.agents.ollama_client import generate, get_main_url


def translate_to_block_spec(
    user_prompt: str,
    model: Optional[str] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Translate a plain-English description into a block-spec JSON.

    Args:
        user_prompt: Plain English description of the system.
        model: Llama model name.
        timeout: Request timeout.

    Returns:
        Dict with block_spec JSON or error.
    """
    model = model or os.getenv("PLANNER_MODEL", "llama3.2:3b")

    system_prompt = """You are a control systems engineer. Convert the user's description into a block-diagram specification.

The specification must be valid JSON with this structure:

{
  "blocks": [
    {
      "id": "block_1",
      "type": "transfer_function",
      "parameters": {
        "numerator": [1.0],
        "denominator": [1.0, 2.0, 1.0]
      }
    },
    {
      "id": "block_2",
      "type": "step_input",
      "parameters": {
        "step_time": 0.0,
        "initial_value": 0.0,
        "final_value": 1.0
      }
    }
  ],
  "connections": [
    {
      "from": "block_2",
      "to": "block_1"
    }
  ]
}

Supported block types (one chain starting at a single step_input; no sums, no feedback):
- transfer_function: parameters are numerator and denominator coefficient lists.
- step_input: parameters are step_time, initial_value, final_value.
- gain: parameters are gain_value.

Respond with ONLY the JSON. No explanation."""

    full_prompt = f"{system_prompt}\n\nUser description: {user_prompt}\n\nBlock specification JSON:"

    response = generate(
        prompt=full_prompt,
        model=model,
        temperature=0.0,
        max_tokens=1024,
        timeout=timeout,
    )

    if "error" in response:
        return {
            "success": False,
            "error": response["error"],
        }

    raw = str(response.get("response", "")).strip()

    # Try to extract JSON from the response.
    try:
        # Find JSON block.
        start = raw.find("{")
        end = raw.rfind("}") + 1

        if start == -1 or end == 0:
            return {
                "success": False,
                "error": "No JSON found in response.",
                "raw": raw,
            }

        json_str = raw[start:end]
        block_spec = json.loads(json_str)

        return {
            "success": True,
            "block_spec": block_spec,
            "raw": raw,
        }

    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "error": f"JSON decode error: {exc}",
            "raw": raw,
        }