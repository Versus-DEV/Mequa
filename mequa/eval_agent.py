"""Measure the Llama tool-calling investigator against a ground truth it never sees.

Ground truth: a fine, serial load sweep on the same mesh, with the same threshold rule
(parallel.find_threshold, 95th percentile strain reaching the strain limit). The agent has to
find that threshold by calling the solver tools on its own.

Temperature is 0, so repeating one question gives the same answer every time. Variety therefore
comes from different meshes (seeds) and different phrasings of the question, not from repeats.

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


def ground_truth(points, stiffness=None, limit=LIMIT) -> Dict[str, Any]:
    loads = np.linspace(0.005, 0.2, 40).tolist()
    out = parallel.sweep_loads(points, loads, k=3, knn=KNN, stiffness=stiffness, comm=None)
    return {"threshold": parallel.find_threshold(out["rows"], limit), "limit": limit, "solves": len(loads)}


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
    row: Dict[str, Any] = {
        "status": result.get("status"),
        "tool_calls": len(calls),
        "tool_errors": len(errors),
        "called_sweep": bool(sweeps),
        "iterations": result.get("iterations"),
        "seconds": round(seconds, 1),
        "truth_threshold": truth["threshold"],
        "tool_threshold": None,
        "threshold_error": None,
        "threshold_correct": False,
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
        "threshold_correct": frac(lambda r: r["threshold_correct"]),
        "threshold_in_final_summary": frac(lambda r: r["threshold_in_summary"]),
        "tool_error_rate": frac(lambda r: r["tool_errors"] > 0),
        "first_attempt_invented_number_rate": frac(lambda r: r["first_attempt_ungrounded"], ok),
        "template_fallback_rate": frac(lambda r: r["template_fallback"], ok),
        "median_threshold_error": round(float(np.median(errs)), 5) if errs else None,
        "median_iterations": float(np.median([r["iterations"] for r in ok])) if ok else None,
        "median_seconds": secs[len(secs) // 2] if secs else None,
        "tolerance": TOLERANCE,
    }


def run_model(model, seeds, phrasings, chat_fn=None, check=None, log=print):
    rows = []
    for seed in seeds:
        pts = nb.synthetic_cells(CELLS, seed=seed, clusters=1)
        sess = Session(pts, knn=KNN)
        truth = ground_truth(pts)
        log(f"[{model}] seed {seed}: ground-truth threshold {truth['threshold']}")
        for p in phrasings:
            kw = {}
            if check is not None:
                kw = {"check_request": check, "check_reply": lambda q, t: check(q)}
            inv = AdaptiveInvestigator(sess, model=model, strain_limit=LIMIT, chat_fn=chat_fn, **kw)
            t0 = time.perf_counter()
            res = inv.run(PHRASINGS[p % len(PHRASINGS)])
            row = score_run(res, truth, time.perf_counter() - t0)
            row.update({"model": model, "seed": seed, "phrasing": p})
            rows.append(row)
            log(f"  phrasing {p}: {row['status']} sweep={row['called_sweep']} tool_thr={row['tool_threshold']} "
                f"err={row['threshold_error']} inv1={row['first_attempt_ungrounded']} fb={row['template_fallback']} {row['seconds']}s")
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
    ap.add_argument("--out", default="results/agent_eval.json")
    ap.add_argument("--dry-run", action="store_true", help="scripted model, no Ollama, no firewall; checks the harness only")
    args = ap.parse_args(argv)

    from mequa.firewall import Verdict

    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    models = ["scripted-dry-run"] if args.dry_run else args.models
    for model in models:
        if args.dry_run:
            rows = run_model(model, args.seeds[:1], range(min(args.phrasings, 2)), chat_fn=_dry_run_chat(),
                             check=lambda *a, **k: Verdict(allowed=True))
        else:
            rows = run_model(model, args.seeds, range(args.phrasings))
        all_rows.extend(rows)
        summary[model] = summarize(rows)
        print(json.dumps({model: summary[model]}, indent=2))
    out = {"config": {"cells": CELLS, "knn": KNN, "strain_limit": LIMIT, "seeds": args.seeds,
                      "phrasings": args.phrasings, "dry_run": args.dry_run,
                      "note": "temperature 0: variety comes from seeds and phrasings, not repeats"},
           "summary": summary, "rows": all_rows}
    if not args.dry_run:
        Path(args.out).parent.mkdir(exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("saved", args.out)
    return out


if __name__ == "__main__":
    main()