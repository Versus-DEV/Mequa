"""Llama translates plain English into Simulink/Xcos block specifications."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from mequa.agents.ollama_client import generate, get_main_url
from mequa.blockspec.block_validator import validate_block_spec
from mequa.blockspec.normalize import normalize_block_spec


def translate_to_block_spec(
    user_prompt: str,
    model: Optional[str] = None,
    timeout: float = 120.0,
    generate_fn=None,
    repair: bool = True,
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
- transfer_function: EITHER a standard form (preferred, the program does the arithmetic) OR coefficient lists in descending powers of s.
    {"form": "first_order", "K": 5.0, "tau": 0.25}                 means K / (tau s + 1)
    {"form": "second_order", "K": 1.0, "wn": 1.0, "zeta": 0.9}     means K wn^2 / (s^2 + 2 zeta wn s + wn^2)
    {"form": "integrator", "K": 1.0}                               means K / s
    {"numerator": [3.0], "denominator": [2.0, 1.0]}                means 3 / (2 s + 1)
  Copy K, tau, wn, zeta from the description exactly; "unit DC gain" means K is 1.
- step_input: parameters are step_time, initial_value, final_value.
- gain: parameters are gain_value. A gain stated as a separate block is a gain block; do not fold it into a transfer function.
- Chain order follows the description: step_input first, then each block in the order described. Use ids block_1, block_2, ... and connect each block to the next.

Respond with ONLY the JSON. No explanation."""

    gen = generate_fn or generate

    def parse(raw: str) -> Dict[str, Any]:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {"success": False, "error": "No JSON found in response.", "raw": raw}
        try:
            return {"success": True, "block_spec": normalize_block_spec(json.loads(raw[start:end])), "raw": raw}
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"JSON decode error: {exc}", "raw": raw}

    prompt = f"{system_prompt}\n\nUser description: {user_prompt}\n\nBlock specification JSON:"
    attempts = 0
    while True:
        attempts += 1
        response = gen(prompt=prompt, model=model, temperature=0.0, max_tokens=1024, timeout=timeout)
        if "error" in response:
            return {"success": False, "error": response["error"], "attempts": attempts}
        result = parse(str(response.get("response", "")).strip())
        result["attempts"] = attempts
        problem = result.get("error") if not result["success"] else (
            None if (v := validate_block_spec(result["block_spec"]))["valid"] else "; ".join(v["errors"]))
        if problem is None or not repair or attempts >= 2:
            return result
        prompt = (f"{system_prompt}\n\nUser description: {user_prompt}\n\nYour previous answer was rejected: {problem}\n"
                  f"Previous answer:\n{result.get('raw', '')[:800]}\n\nCorrected block specification JSON:")
