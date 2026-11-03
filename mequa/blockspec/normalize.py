"""Let the model name a standard form and its parameters; code does the arithmetic.

A language model is good at reading "damping ratio 0.3, natural frequency 2" and poor at computing
2*0.3*2 and 2**2 reliably. A transfer_function block may therefore carry a "form" instead of
coefficient lists, and this module expands it into numerator/denominator (descending powers of s).

  first_order   K / (tau s + 1)                       needs tau, optional K (default 1)
  second_order  K wn^2 / (s^2 + 2 zeta wn s + wn^2)   needs wn, zeta, optional K (default 1)
  integrator    K / s                                 optional K (default 1)

Blocks that already carry numerator/denominator are left untouched.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def normalize_block_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(spec)
    if not isinstance(out, dict) or not isinstance(out.get("blocks"), list):
        return out
    for b in out["blocks"]:
        if not isinstance(b, dict) or b.get("type") != "transfer_function":
            continue
        p = b.get("parameters")
        if not isinstance(p, dict) or "form" not in p or ("numerator" in p and "denominator" in p):
            continue
        form, K = p.get("form"), p.get("K", 1.0)
        if not _num(K):
            continue
        if form == "first_order" and _num(p.get("tau")) and p["tau"] > 0:
            p["numerator"], p["denominator"] = [float(K)], [float(p["tau"]), 1.0]
        elif form == "second_order" and _num(p.get("wn")) and _num(p.get("zeta")) and p["wn"] > 0 and p["zeta"] >= 0:
            wn, z = float(p["wn"]), float(p["zeta"])
            p["numerator"], p["denominator"] = [float(K) * wn * wn], [1.0, 2.0 * z * wn, wn * wn]
        elif form == "integrator":
            p["numerator"], p["denominator"] = [float(K)], [1.0, 0.0]
    return out
