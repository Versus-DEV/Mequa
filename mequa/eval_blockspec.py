"""Measure how often Llama writes a block specification that is valid AND correct.

Each prompt describes a system whose step response has a known analytic answer, so the check is
not "does the JSON parse" but "does the simulated output match the maths":
  first-order lag     y(10) = K * h * (1 - exp(-10/tau))
  gain then lag       y(10) = g * h * (1 - exp(-10/tau))
  second-order        overshoot = exp(-pi*zeta/sqrt(1-zeta^2)), measured on y / y_final
Stages per prompt: JSON found -> validator passes -> simulation runs -> matches analytic answer ->
output drives the strain solver (coupling) and strain responds to the signal.

Run:  python -m mequa.eval_blockspec --models llama3.2:3b llama3.1:latest
Smoke test without Ollama:  python -m mequa.eval_blockspec --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

T_END = 10.0
REL_TOL = 0.02       # relative tolerance on final values
OVERSHOOT_TOL = 0.03  # absolute tolerance on overshoot fraction


def build_cases() -> List[Dict[str, Any]]:
    cases = []
    for K, tau in [(2.0, 1.5), (0.5, 0.8), (3.0, 2.0), (1.0, 1.0), (4.0, 0.5)]:
        cases.append({
            "kind": "first_order",
            "prompt": f"A first-order system with DC gain {K} and time constant {tau} seconds, driven by a unit step.",
            "expected_final": K * (1 - math.exp(-T_END / tau)),
        })
    for h, g, tau in [(2.0, 3.0, 1.0), (1.5, 0.5, 2.0), (5.0, 2.0, 0.7)]:
        cases.append({
            "kind": "gain_lag",
            "prompt": f"A step input of height {h} goes into a gain of {g}, followed by a first-order lag with time constant {tau} seconds and unit DC gain.",
            "expected_final": g * h * (1 - math.exp(-T_END / tau)),
        })
    for wn, zeta in [(2.0, 0.3), (3.0, 0.5), (1.5, 0.2), (4.0, 0.6)]:
        cases.append({
            "kind": "second_order",
            "prompt": f"A second-order system with natural frequency {wn} rad/s and damping ratio {zeta}, unit DC gain, driven by a unit step.",
            "expected_overshoot": math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)),
        })
    return cases


def check_case(case: Dict[str, Any], spec: Dict[str, Any], points, backend) -> Dict[str, Any]:
    from mequa.blockspec.block_validator import validate_block_spec
    from mequa.blockspec.xcos_importer import simulate_block_spec
    from mequa.couple import drive_strain

    stages = {"valid": False, "simulated": False, "matches_maths": False, "coupled": False}
    detail: Dict[str, Any] = {}
    check = validate_block_spec(spec)
    detail["validator_errors"] = check["errors"]
    if not check["valid"]:
        return {"stages": stages, "detail": detail}
    stages["valid"] = True
    sim = simulate_block_spec(spec, t_end=T_END)
    if not sim["success"]:
        detail["sim_error"] = sim["error"]
        return {"stages": stages, "detail": detail}
    stages["simulated"] = True
    t, y = np.asarray(sim["time"]), np.asarray(sim["output"])
    if case["kind"] == "second_order":
        final = float(y[-1])
        measured = (float(y.max()) / final - 1.0) if abs(final) > 1e-9 else float("nan")
        detail.update({"measured_overshoot": round(measured, 4), "expected_overshoot": round(case["expected_overshoot"], 4)})
        stages["matches_maths"] = bool(abs(measured - case["expected_overshoot"]) <= OVERSHOOT_TOL)
    else:
        exp = case["expected_final"]
        detail.update({"final": round(float(y[-1]), 4), "expected_final": round(exp, 4)})
        stages["matches_maths"] = bool(abs(y[-1] - exp) <= REL_TOL * abs(exp))
    if np.ptp(y) > 1e-9:
        rows = drive_strain(t, y, points, backend, samples=6)
        strains = [r["mean_strain"] for r in rows]
        detail["strain_range"] = [round(min(strains), 5), round(max(strains), 5)]
        stages["coupled"] = bool(len(rows) >= 2 and max(strains) > min(strains))
    return {"stages": stages, "detail": detail}


def run_model(model, translate_fn: Callable, points, backend, log=print) -> List[Dict[str, Any]]:
    rows = []
    for case in build_cases():
        t0 = time.perf_counter()
        out = translate_fn(case["prompt"], model=model)
        secs = round(time.perf_counter() - t0, 1)
        if not out.get("success"):
            res = {"stages": {"json": False, "valid": False, "simulated": False, "matches_maths": False, "coupled": False},
                   "detail": {"error": out.get("error")}}
        else:
            res = check_case(case, out["block_spec"], points, backend)
            res["stages"] = {"json": True, **res["stages"]}
        res.update({"model": model, "kind": case["kind"], "prompt": case["prompt"], "seconds": secs})
        rows.append(res)
        s = res["stages"]
        log(f"[{model}] {case['kind']:<12} json={s['json']} valid={s['valid']} sim={s['simulated']} maths={s['matches_maths']} coupled={s['coupled']} {secs}s")
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {"prompts": n}
    for stage in ("json", "valid", "simulated", "matches_maths", "coupled"):
        out[stage] = round(sum(1 for r in rows if r["stages"].get(stage)) / n, 3) if n else None
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        out[f"matches_maths_{kind}"] = f"{sum(1 for r in sub if r['stages'].get('matches_maths'))}/{len(sub)}"
    return out


def _dry_translate(prompt, model=None):
    """Scripted stand-in: parses the numbers back out of the prompt and writes a correct spec."""
    import re

    nums = [float(x) for x in re.findall(r"\d+\.?\d*", prompt)]
    if "second-order" in prompt:
        wn, zeta = nums[0], nums[1]
        tf = {"numerator": [wn ** 2], "denominator": [1.0, 2 * zeta * wn, wn ** 2]}
        blocks = [{"id": "s", "type": "step_input", "parameters": {"step_time": 0.0, "initial_value": 0.0, "final_value": 1.0}},
                  {"id": "tf", "type": "transfer_function", "parameters": tf}]
    elif "gain of" in prompt:
        h, g, tau = nums[0], nums[1], nums[2]
        blocks = [{"id": "s", "type": "step_input", "parameters": {"step_time": 0.0, "initial_value": 0.0, "final_value": h}},
                  {"id": "g", "type": "gain", "parameters": {"gain_value": g}},
                  {"id": "tf", "type": "transfer_function", "parameters": {"numerator": [1.0], "denominator": [tau, 1.0]}}]
    else:
        K, tau = nums[0], nums[1]
        blocks = [{"id": "s", "type": "step_input", "parameters": {"step_time": 0.0, "initial_value": 0.0, "final_value": 1.0}},
                  {"id": "tf", "type": "transfer_function", "parameters": {"numerator": [K], "denominator": [tau, 1.0]}}]
    conns = [{"from": a["id"], "to": b["id"]} for a, b in zip(blocks, blocks[1:])]
    return {"success": True, "block_spec": {"blocks": blocks, "connections": conns}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:3b", "llama3.1:latest"])
    ap.add_argument("--out", default="results/blockspec_eval.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from mequa import backend as backend_mod
    from mequa import numpy_backend as nb

    points = nb.synthetic_cells(300, seed=1, clusters=1)
    backend = backend_mod.load("numpy")
    if args.dry_run:
        translate, models = _dry_translate, ["scripted-dry-run"]
    else:
        from mequa.blockspec.semantic_block_translator import translate_to_block_spec as translate
        models = args.models
    all_rows, summary = [], {}
    for m in models:
        rows = run_model(m, translate, points, backend)
        all_rows.extend(rows)
        summary[m] = summarize(rows)
        print(json.dumps({m: summary[m]}, indent=2))
    if not args.dry_run:
        Path(args.out).parent.mkdir(exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": all_rows}, indent=2), encoding="utf-8")
        print("saved", args.out)
    return summary


if __name__ == "__main__":
    main()