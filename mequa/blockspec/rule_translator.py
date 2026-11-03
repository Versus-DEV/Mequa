"""Offline plain-English -> block-spec parser (no Llama needed).

Used when Ollama is not running, so the Simulink editor still works for the common forms. It is
deliberately narrow: it understands a step input followed by first-order, second-order and
integrator transfer functions and gain blocks, and refuses anything else with a clear reason
instead of guessing. The result is labelled source="rules" so nobody mistakes it for the model.

Examples it understands:
  "first-order transfer function with gain 2 and time constant 1, driven by a step input"
  "second-order system, natural frequency 3 rad/s, damping ratio 0.4, step of amplitude 2"
  "an integrator with gain 5 followed by a gain block of 3"
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from mequa.blockspec.block_validator import validate_block_spec
from mequa.blockspec.normalize import normalize_block_spec

_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
_GAIN_BLOCK = re.compile(
    r"(?:followed by|then|and then|cascaded with|in series with|into|feeding|plus)\s+(?:an?\s+|the\s+)?(?:extra\s+)?"
    r"(?:gain(?:\s+block)?(?:\s+(?:of|=|with value))?|amplifier\s+(?:of|with gain))\s*" + _NUM, re.I)


def _find(text: str, patterns: List[str]) -> Optional[float]:
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return float(m.group(1))
    return None


def parse_description(text: str) -> Dict[str, Any]:
    t = " ".join(text.replace("τ", " tau ").replace("ζ", " zeta ").replace("ω", " w ").split())
    gain_blocks = [float(m.group(1)) for m in _GAIN_BLOCK.finditer(t)]
    core = _GAIN_BLOCK.sub(" ", t)  # remove separate gain clauses so they are not read as the transfer function's K

    amp = _find(core, [r"step(?: input)?\s+(?:of\s+)?(?:amplitude|height|size|magnitude)\s*(?:of|=)?\s*" + _NUM,
                       r"amplitude\s*(?:of|=)?\s*" + _NUM, r"step(?: input)?\s+of\s+" + _NUM,
                       r"step(?: input)?\s+(?:to|up to)\s+" + _NUM])
    step_t = _find(core, [r"step\s+(?:occurs\s+|starts\s+|happens\s+)?at\s+(?:t\s*=\s*)?" + _NUM, r"at\s+t\s*=\s*" + _NUM,
                          r"step\s+time\s*(?:of|=)?\s*" + _NUM])
    K = _find(core, [r"(?:dc\s+)?gain\s*(?:of|=|k\s*=|is)?\s*" + _NUM, r"\bk\s*=\s*" + _NUM])
    if re.search(r"unit(?:y)?\s+(?:dc\s+)?gain|gain\s+(?:of\s+)?(?:one|unity)\b", core, re.I):
        K = 1.0

    form: Optional[Dict[str, Any]] = None
    if re.search(r"second[\s-]*order", core, re.I) or re.search(r"damping|zeta|natural\s+freq", core, re.I):
        wn = _find(core, [r"natural\s+freq\w*\s*(?:of|=|is)?\s*" + _NUM, r"\bw\s*n\s*(?:=|of)\s*" + _NUM, r"omega[\s_]*n?\s*(?:=|of)\s*" + _NUM])
        z = _find(core, [r"damping(?:\s+ratio)?\s*(?:of|=|is)?\s*" + _NUM, r"zeta\s*(?:=|of|is)?\s*" + _NUM])
        if wn is None or z is None:
            return {"success": False, "error": "A second-order system needs a natural frequency and a damping ratio, for example "
                                               "'natural frequency 3 and damping ratio 0.4'."}
        form = {"form": "second_order", "K": 1.0 if K is None else K, "wn": wn, "zeta": z}
    elif re.search(r"integrat", core, re.I):
        form = {"form": "integrator", "K": 1.0 if K is None else K}
    elif re.search(r"first[\s-]*order|time\s+constant|\btau\b|low[\s-]*pass|lag\b", core, re.I):
        tau = _find(core, [r"time\s+constant\s*(?:of|=|is|tau\s*=)?\s*" + _NUM, r"\btau\s*(?:=|of|is)?\s*" + _NUM])
        if tau is None:
            return {"success": False, "error": "A first-order system needs a time constant, for example 'time constant 0.5'."}
        form = {"form": "first_order", "K": 1.0 if K is None else K, "tau": tau}

    if form is None and not gain_blocks:
        if K is not None and re.search(r"\bgain\b", core, re.I):
            gain_blocks = [K]
        else:
            return {"success": False, "error": "I can parse first-order, second-order and integrator systems with an optional step input and "
                                               "gain blocks. Name the system type and its parameters (time constant, natural frequency, damping ratio, gain)."}

    blocks: List[Dict[str, Any]] = [{"id": "block_1", "type": "step_input",
                                     "parameters": {"step_time": 0.0 if step_t is None else step_t, "initial_value": 0.0,
                                                    "final_value": 1.0 if amp is None else amp}}]
    if form is not None:
        blocks.append({"id": f"block_{len(blocks) + 1}", "type": "transfer_function", "parameters": form})
    for g in gain_blocks:
        blocks.append({"id": f"block_{len(blocks) + 1}", "type": "gain", "parameters": {"gain_value": g}})
    conns = [{"from": blocks[i]["id"], "to": blocks[i + 1]["id"]} for i in range(len(blocks) - 1)]
    return {"success": True, "block_spec": normalize_block_spec({"blocks": blocks, "connections": conns}),
            "raw": "(parsed by the built-in rules, no language model used)", "source": "rules"}


def translate_offline(text: str) -> Dict[str, Any]:
    res = parse_description(text)
    if res.get("success"):
        check = validate_block_spec(res["block_spec"])
        if not check["valid"]:
            return {"success": False, "error": "; ".join(check["errors"])}
    return res
