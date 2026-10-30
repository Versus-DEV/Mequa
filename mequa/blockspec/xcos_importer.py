"""Simulate a validated block specification: step_input -> gain / transfer_function chain (SciPy lsim)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from mequa.blockspec.block_validator import validate_block_spec

try:
    from scipy import signal
except ImportError:  # pragma: no cover
    signal = None


def simulate_block_spec(block_spec: Dict[str, Any], t_end: float = 10.0, n: int = 1001) -> Dict[str, Any]:
    if signal is None:
        return {"success": False, "error": "scipy is not installed"}
    check = validate_block_spec(block_spec)
    if not check["valid"]:
        return {"success": False, "error": "invalid block spec: " + "; ".join(check["errors"])}
    by_id = {b["id"]: b for b in block_spec["blocks"]}
    nxt = {c["from"]: c["to"] for c in block_spec.get("connections", [])}
    src = next(k for k, b in by_id.items() if b["type"] == "step_input")
    p = by_id[src].get("parameters", {}) or {}
    t = np.linspace(0.0, float(t_end), int(n))
    u = np.where(t >= float(p.get("step_time", 0.0)), float(p.get("final_value", 1.0)), float(p.get("initial_value", 0.0)))
    signal_now, cur, order = u, src, [src]
    try:
        while cur in nxt:
            cur = nxt[cur]
            order.append(cur)
            b = by_id[cur]
            bp = b.get("parameters", {}) or {}
            if b["type"] == "gain":
                signal_now = signal_now * float(bp["gain_value"])
            else:
                system = signal.TransferFunction(bp["numerator"], bp["denominator"])
                _, signal_now, _ = signal.lsim(system, U=signal_now, T=t)
    except Exception as exc:
        return {"success": False, "error": f"simulation error: {exc}"}
    return {"success": True, "time": t.tolist(), "input": u.tolist(), "output": np.asarray(signal_now).tolist(), "chain": order}
