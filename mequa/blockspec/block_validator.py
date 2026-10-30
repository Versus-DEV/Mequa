"""Validate a block specification (for example one written by Llama) before it is simulated.

Supported: step_input -> (gain | transfer_function)* as a single chain. Anything else, including
`sum` blocks and feedback loops, is rejected with a reason instead of being silently ignored.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

SUPPORTED = ("step_input", "gain", "transfer_function")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _num_list(x) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(_is_num(v) for v in x)


def validate_block_spec(block_spec: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    if not isinstance(block_spec, dict):
        return {"valid": False, "errors": ["block_spec must be a dict"]}
    blocks = block_spec.get("blocks", [])
    conns = block_spec.get("connections", [])
    if not isinstance(blocks, list) or not isinstance(conns, list):
        return {"valid": False, "errors": ["blocks and connections must be lists"]}
    if not blocks:
        return {"valid": False, "errors": ["no blocks"]}

    by_id: Dict[str, Dict[str, Any]] = {}
    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            errors.append(f"block {i} must be an object")
            continue
        bid, btype, params = b.get("id"), b.get("type"), b.get("parameters", {}) or {}
        if not bid or not isinstance(bid, str):
            errors.append(f"block {i} needs a string id")
            continue
        if bid in by_id:
            errors.append(f"duplicate block id {bid}")
            continue
        by_id[bid] = b
        if btype not in SUPPORTED:
            errors.append(f"block {bid}: type {btype!r} is not supported (supported: {', '.join(SUPPORTED)})")
            continue
        if btype == "gain" and not _is_num(params.get("gain_value")):
            errors.append(f"block {bid}: gain_value must be a number")
        if btype == "step_input":
            for key in ("step_time", "initial_value", "final_value"):
                if key in params and not _is_num(params[key]):
                    errors.append(f"block {bid}: {key} must be a number")
            if _is_num(params.get("step_time")) and params["step_time"] < 0:
                errors.append(f"block {bid}: step_time must be >= 0")
        if btype == "transfer_function":
            num, den = params.get("numerator"), params.get("denominator")
            if not _num_list(num) or not _num_list(den):
                errors.append(f"block {bid}: numerator and denominator must be non-empty number lists")
            else:
                if den[0] == 0:
                    errors.append(f"block {bid}: leading denominator coefficient must be non-zero")
                if len(num) > len(den):
                    errors.append(f"block {bid}: transfer function must be proper (numerator order <= denominator order)")

    out_deg: Dict[str, int] = {k: 0 for k in by_id}
    in_deg: Dict[str, int] = {k: 0 for k in by_id}
    for i, c in enumerate(conns):
        if not isinstance(c, dict) or c.get("from") not in by_id or c.get("to") not in by_id:
            errors.append(f"connection {i} must name existing blocks in 'from' and 'to'")
            continue
        out_deg[c["from"]] += 1
        in_deg[c["to"]] += 1
    for k in by_id:
        if out_deg[k] > 1 or in_deg[k] > 1:
            errors.append(f"block {k}: only a single chain is supported (one input, one output per block)")
    sources = [k for k, b in by_id.items() if b.get("type") == "step_input"]
    if len(sources) != 1:
        errors.append("exactly one step_input block is required")
    elif in_deg[sources[0]] != 0:
        errors.append("step_input cannot have an incoming connection")
    if not errors and sources:
        nxt = {c["from"]: c["to"] for c in conns}
        seen, cur = {sources[0]}, sources[0]
        while cur in nxt:
            cur = nxt[cur]
            if cur in seen:
                errors.append("feedback loops are not supported")
                break
            seen.add(cur)
        if not errors and len(seen) != len(by_id):
            errors.append("every block must be on the chain that starts at the step_input")
    return {"valid": not errors, "errors": errors}
