"""Measure the Llama tool-calling investigator against a ground truth it never sees.

Ground truth: a fine, serial load sweep on the same mesh, with the same threshold rule
(parallel.find_threshold, 95th percentile strain reaching the strain limit). The agent has to
find that threshold by calling the solver tools on its own.

Temperature is 0, so repeating one question gives the same answer every time. Variety therefore
comes from different meshes (seeds) and different phrasings of the question, not from repeats.

Per run it records: status, whether the model called sweep_load, whether the sweep bracketed the
crossing, the threshold the tool computed, the error against ground truth, whether that threshold
appears in the final summary, the number-grounding outcome on the first attempt, whether the template
fallback fired, rounds used and seconds.

Run (Ollama must be up):
  python -m mequa.eval_agent --models llama3.2:3b llama3.1:latest --seeds 1 2 3 --phrasings 3
Smoke test without Ollama:
  python -m mequa.eval_agent --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mequa import numpy_backend as nb
from mequa import parallel
from mequa.agents.adaptive_investigator import AdaptiveInvestigator
from mequa.session import Session

PHRASINGS = [
    "At what load fraction does the 95th percentile strain reach the strain limit in this dataset?",
    "Find the load at which this cell mesh reaches the strain limit.",
    "Sweep the load and tell me the threshold where strain crosses the limit.",
    "How much load can this mesh take before the strain limit is reached?",
    "Determine the failure-threshold load for the dataset using the solver.",
]

CELLS = 600
KNN = 8
LIMIT = 0.05
TOLERANCE = 0.005  # absolute load-fraction tolerance; the agent sweeps a coarser grid than the ground truth
NUMBER = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def ground_truth(points, stiffness=None, limit=LIMIT, auto_limit=False) -> Dict[str, Any]:
    loads = np.linspace(0.005, 0.2, 40).tolist()
    out = parallel.sweep_loads(points, loads, k=3, knn=KNN, stiffness=stiffness, comm=None)
    if auto_limit:  # real meshes: pick a limit the strain actually crosses, the p95 strain at load 0.06
        xs = [r["load_fraction"] for r in out["rows"]]
        ys = [r["p95_strain"] for r in out["rows"]]
        limit = round(float(np.interp(0.06, xs, ys)), 3)
    return {"threshold": parallel.find_threshold(out["rows"], limit), "limit": limit, "solves": len(loads)}


def load_points_csv(path):
    import pandas as pd

    df = pd.read_csv(path)
    cols = [c for c in ("x", "y", "z") if c in df.columns] or list(df.select_dtypes("number").columns[:3])
    return np.ascontiguousarray(df[cols].to_numpy(dtype=np.float64))


def numbers_in(text: str) -> List[float]:
    out = []
    for m in NUMBER.findall(text or ""):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def score_run(result: Dict[str, Any], truth: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    trace = result.get("trace", [])
    calls = [s for s in trace if s.get("type") == "tool_call"]
    sweeps = [s for s in calls if s["tool"] == "sweep_load" and "rows" in s["result"]]
    errors = [s for s in calls if "error" in s["result"]]
    model_sweeps = [s for s in sweeps if not s.get("forced")]
    row: Dict[str, Any] = {
        "status": result.get("status"),
        "tool_calls": len(calls),
        "tool_errors": len(errors),
        "called_sweep": bool(model_sweeps),
        "forced_sweep": bool(result.get("forced_sweep")),
        "diagnostics": None if model_sweeps else result.get("diagnostics"),
        "iterations": result.get("iterations"),
        "seconds": round(seconds, 1),
        "truth_threshold": truth["threshold"],
        "tool_threshold": None,
        "threshold_error": None,
        "threshold_correct": False,
        "threshold_correct_model_initiated": False,
        "threshold_in_summary": False,
        "first_attempt_ungrounded": None,
        "template_fallback": None,
    }
    if sweeps:
        t = sweeps[-1]["result"].get("threshold_load")
        row["tool_threshold"] = t
        if t is not None and truth["threshold"] is not None:
            row["threshold_error"] = round(abs(t - truth["threshold"]), 5)
            row["threshold_correct"] = row["threshold_error"] <= TOLERANCE
            row["threshold_correct_model_initiated"] = row["threshold_correct"] and bool(model_sweeps)
    if result.get("status") == "ok":
        g = result["grounding"]
        row["first_attempt_ungrounded"] = bool(g["attempts"][0]["ungrounded"])
        row["template_fallback"] = bool(g["fallback"])
        if truth["threshold"] is not None:
            row["threshold_in_summary"] = any(abs(v - truth["threshold"]) <= TOLERANCE for v in numbers_in(result["summary"]))
    return row


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]

    def frac(pred, pool=rows):
        return round(sum(1 for r in pool if pred(r)) / len(pool), 3) if pool else None

    secs = sorted(r["seconds"] for r in rows)
    errs = [r["threshold_error"] for r in rows if r["threshold_error"] is not None]
    return {
        "runs": n,
        "completed": frac(lambda r: r["status"] == "ok"),
        "called_sweep": frac(lambda r: r["called_sweep"]),
        "forced_sweep_rate": frac(lambda r: r.get("forced_sweep", False)),
        "threshold_correct": frac(lambda r: r["threshold_correct"]),
        "threshold_correct_model_initiated": frac(lambda r: r.get("threshold_correct_model_initiated", r["threshold_correct"] and r["called_sweep"])),
        "threshold_in_final_summary": frac(lambda r: r["threshold_in_summary"]),
        "tool_error_rate": frac(lambda r: r["tool_errors"] > 0),
        "first_attempt_invented_number_rate": frac(lambda r: r["first_attempt_ungrounded"], ok),
        "template_fallback_rate": frac(lambda r: r["template_fallback"], ok),
        "median_threshold_error": round(float(np.median(errs)), 5) if errs else None,
        "median_iterations": float(np.median([r["iterations"] for r in ok])) if ok else None,
        "median_seconds": secs[len(secs) // 2] if secs else None,
        "tolerance": TOLERANCE,
    }


def run_model(model, meshes, phrasings, chat_fn=None, check=None, log=print):
    """meshes: list of (label, points, truth)."""
    rows = []
    for label, pts, truth in meshes:
        sess = Session(pts, knn=KNN)
        log(f"[{model}] {label}: ground-truth threshold {truth['threshold']} (limit {truth['limit']})")
        for p in phrasings:
            kw = {}
            if check is not None:
                kw = {"check_request": check, "check_reply": lambda q, t: check(q)}
            inv = AdaptiveInvestigator(sess, model=model, strain_limit=truth["limit"], chat_fn=chat_fn, **kw)
            t0 = time.perf_counter()
            res = inv.run(PHRASINGS[p % len(PHRASINGS)])
            row = score_run(res, truth, time.perf_counter() - t0)
            row.update({"model": model, "mesh": label, "phrasing": p})
            rows.append(row)
            log(f"  phrasing {p}: {row['status']} model_sweep={row['called_sweep']} forced={row['forced_sweep']} tool_thr={row['tool_threshold']} "
                f"err={row['threshold_error']} inv1={row['first_attempt_ungrounded']} fb={row['template_fallback']} {row['seconds']}s")
            if not row["called_sweep"] and row["diagnostics"]:
                log(f"    why: {row['diagnostics']}")
    return rows


def _dry_run_chat():
    """A scripted stand-in for a model: sweeps once, then reports the tool's own threshold."""
    state = {"i": 0}

    def chat(messages, model, tools=None, **kw):
        state["i"] += 1
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if not tool_msgs:
            return {"message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "sweep_load", "arguments": {"start": 0.01, "stop": 0.12, "steps": 10}}}]}}
        res = json.loads(tool_msgs[-1]["content"])
        return {"message": {"role": "assistant", "content": f"The 95th percentile strain reaches the limit at a load fraction of {res['threshold_load']}."}}

    return chat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:3b", "llama3.1:latest"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--phrasings", type=int, default=3, help="how many of the %d phrasings to use per seed" % len(PHRASINGS))
    ap.add_argument("--points", default=None, help="CSV of real cell coordinates (x,y[,z]) to use instead of synthetic meshes; the strain limit is then chosen automatically")
    ap.add_argument("--out", default="results/agent_eval.json")
    ap.add_argument("--dry-run", action="store_true", help="scripted model, no Ollama, no firewall; checks the harness only")
    args = ap.parse_args(argv)

    from mequa.firewall import Verdict

    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    models = ["scripted-dry-run"] if args.dry_run else args.models
    meshes = []
    if args.points:
        pts = load_points_csv(args.points)
        meshes.append((f"real:{Path(args.points).name}", pts, ground_truth(pts, auto_limit=True)))
    else:
        for seed in (args.seeds[:1] if args.dry_run else args.seeds):
            pts = nb.synthetic_cells(CELLS, seed=seed, clusters=1)
            meshes.append((f"synthetic seed {seed}", pts, ground_truth(pts)))
    for model in models:
        if args.dry_run:
            rows = run_model(model, meshes, range(min(args.phrasings, 2)), chat_fn=_dry_run_chat(),
                             check=lambda *a, **k: Verdict(allowed=True))
        else:
            rows = run_model(model, meshes, range(args.phrasings))
        all_rows.extend(rows)
        summary[model] = summarize(rows)
        print(json.dumps({model: summary[model]}, indent=2))
    out = {"config": {"cells": CELLS, "knn": KNN, "strain_limit": LIMIT, "seeds": args.seeds,
                      "phrasings": args.phrasings, "dry_run": args.dry_run,
                      "note": "temperature 0: variety comes from seeds and phrasings, not repeats"},
           "summary": summary, "rows": all_rows}
    if not args.dry_run:
        Path(args.out).parent.mkdir(exist_ok=True)
        old = Path(args.out)
        if old.exists():
            prev = json.loads(old.read_text(encoding="utf-8"))
            out["summary"] = {**prev.get("summary", {}), **summary}
            out["rows"] = [r for r in prev.get("rows", []) if r.get("model") not in summary] + all_rows
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("saved", args.out)
    return out


if __name__ == "__main__":
    main()
