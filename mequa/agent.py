import argparse
import json
import os
import time

import numpy as np

from mequa import backend as backend_mod
from mequa.agents.number_grounding import template_summary, ungrounded_numbers
from mequa.firewall import _ollama_chat, check_reply, check_request
from mequa.session import ensure_connectivity

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "llama3.2:3b")

PLAN_SYSTEM = (
    "You plan numerical experiments for Mequa, a research workbench that analyzes cell point clouds. "
    "Return only a JSON object with these keys: "
    "\"zones\" (integer from 2 to 8, number of density zones), "
    "\"neighbors\" (integer from 3 to 12, neighbors per cell in the mechanical graph), "
    "\"load\" (number from 0.01 to 0.15, fraction of tissue width the right edge is pulled), "
    "and \"reason\" (one short sentence). "
    "If previous attempts list issues, change the parameters to fix them."
)

SUMMARY_SYSTEM = (
    "You write short factual summaries of numerical results for researchers. "
    "Use only the numbers provided and never invent metrics or definitions. "
    "Glossary: strain is the dimensionless relative change in length of the mechanical links between neighboring cells. "
    "SPHI (Systemic Propagating Hazard Index) is the leading eigenvalue of the neighbor graph weighted by the mean strain "
    "of connected cells; a higher value means strain is both larger and more widely connected. "
    "Failure probability is a logistic function of cell strain centered on the 90th percentile strain (s_crit); "
    "it is a simulation output, not a clinical prediction. "
    "Do not give medical advice or refer to individual patients. Do not use qualitative risk labels such as low, moderate or high; "
    "state the numbers and compare zones to each other. Write 3 to 5 sentences covering the zones, "
    "the strain pattern and the SPHI value."
)


def _num(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, low, high):
    return max(low, min(high, value))


def plan_experiment(request, n_cells, history):
    attempts = json.dumps(history[-2:]) if history else "none"
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {
            "role": "user",
            "content": f"Request: {request}\nCells in dataset: {n_cells}\nPrevious attempts: {attempts}",
        },
    ]
    raw = _ollama_chat(PLANNER_MODEL, messages, fmt="json", num_predict=120)
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "zones": int(_clamp(_num(data.get("zones"), 3), 2, 8)),
        "neighbors": int(_clamp(_num(data.get("neighbors"), 6), 3, 12)),
        "load": float(_clamp(_num(data.get("load"), 0.05), 0.01, 0.15)),
        "reason": str(data.get("reason", ""))[:200],
    }


def zone_stats(result):
    labels = result["labels"]
    strain = result["node_strain"]
    fail = result["failure_prob"]
    rows = []
    for c in sorted(set(int(v) for v in labels)):
        mask = labels == c
        rows.append({
            "zone": c,
            "cells": int(mask.sum()),
            "mean_strain": round(float(strain[mask].mean()), 5),
            "share_high_failure_prob": round(float((fail[mask] > 0.5).mean()), 4),
        })
    return rows


def critique(result, n_cells):
    issues = []
    zones = zone_stats(result)
    sizes = [z["cells"] for z in zones]
    if sizes and min(sizes) < 0.05 * n_cells:
        issues.append("a zone holds under 5 percent of cells; try fewer zones")
    if len(sizes) > 2 and max(sizes) > 0.7 * n_cells:
        issues.append("one zone holds over 70 percent of cells; try more zones")
    if not np.isfinite(result["sphi"]):
        issues.append("SPHI is not finite")
    if result.get("load_path") is False:
        issues.append("the pulled edge is not mechanically connected to the held edge; raise neighbors")
    return issues, zones


def build_facts(result, plan, n_cells, zones):
    strain = result["node_strain"]
    fail = result["failure_prob"]
    return {
        "cells": n_cells,
        "zones": plan["zones"],
        "neighbors": plan["neighbors"],
        "load_fraction": plan["load"],
        "mechanical_links": result["n_edges"],
        "sphi": round(float(result["sphi"]), 6),
        "mean_strain": round(float(strain.mean()), 5),
        "max_strain": round(float(strain.max()), 5),
        "s_crit": round(float(result["s_crit"]), 5),
        "share_cells_failure_prob_above_half": round(float((fail > 0.5).mean()), 4),
        "per_zone": zones,
        "graph_components": result.get("n_components"),
        "load_path_connected": result.get("load_path"),
    }


def summarize(request, facts, feedback=""):
    note = f"\n{feedback}" if feedback else ""
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"Request: {request}\nResults: {json.dumps(facts)}{note}"},
    ]
    return _ollama_chat(PLANNER_MODEL, messages, num_predict=260)


def summarize_grounded(request, facts, max_attempts=3, writer=None):
    """Llama writes the summary; every number in it must trace to the solver facts.

    A summary with an untraceable number is rejected and rewritten with the offending numbers
    named. After max_attempts the deterministic template is used, so an invented number never
    reaches the user. Returns (text, info) where info records each attempt for measurement.
    """
    writer = writer or summarize
    zone_ids = [z["zone"] for z in facts.get("per_zone", [])]
    attempts, feedback = [], ""
    for i in range(max_attempts):
        text = writer(request, facts, feedback)
        bad = ungrounded_numbers(text, facts, zone_ids)
        attempts.append({"attempt": i + 1, "ungrounded": bad})
        if not bad:
            return text, {"attempts": attempts, "fallback": False}
        feedback = (
            f"Your previous summary contained numbers that are not in Results: {bad}. "
            "Rewrite it using only numbers that appear in Results."
        )
    return template_summary(facts), {"attempts": attempts, "fallback": True}


def run_agent(request, points=None, max_rounds=3, backend=None, stiffness=None):
    be = backend or backend_mod.load("julia")
    trace = []

    def step(name, start, detail=""):
        trace.append({"step": name, "seconds": round(time.perf_counter() - start, 2), "detail": detail})

    t = time.perf_counter()
    verdict = check_request(request)
    step("firewall.request", t, ",".join(verdict.blocked_by) or "allowed")
    if not verdict.allowed:
        return {
            "status": "blocked",
            "blocked_by": verdict.blocked_by,
            "categories": verdict.categories,
            "trace": trace,
            "backend": be.name,
        }
    if points is None:
        points = be.synthetic_cells()
    n_cells = len(points)
    history = []
    t = time.perf_counter()
    plan = plan_experiment(request, n_cells, history)
    step("llama.plan", t, json.dumps(plan))
    facts = None
    for _ in range(max_rounds):
        t = time.perf_counter()
        result = backend_mod.analyze_routed(be, points, k=plan["zones"], knn=plan["neighbors"], load=plan["load"], stiffness=stiffness)
        ensure_connectivity(result, points, plan["neighbors"])
        step("solver." + result["backend_used"], t, f"zones={plan['zones']} neighbors={plan['neighbors']} load={plan['load']}")
        issues, zones = critique(result, n_cells)
        facts = build_facts(result, plan, n_cells, zones)
        history.append({
            "zones": plan["zones"],
            "neighbors": plan["neighbors"],
            "load": plan["load"],
            "zone_sizes": [z["cells"] for z in zones],
            "sphi": facts["sphi"],
            "issues": issues,
        })
        if not issues:
            break
        t = time.perf_counter()
        plan = plan_experiment(request, n_cells, history)
        step("llama.replan", t, json.dumps(plan))
    t = time.perf_counter()
    summary, grounding = summarize_grounded(request, facts)
    step("llama.summary", t, f"attempts={len(grounding['attempts'])} fallback={grounding['fallback']}")
    t = time.perf_counter()
    reply_verdict = check_reply(request, summary)
    step("firewall.reply", t, ",".join(reply_verdict.blocked_by) or "allowed")
    if not reply_verdict.allowed:
        return {
            "status": "reply_blocked",
            "blocked_by": reply_verdict.blocked_by,
            "history": history,
            "trace": trace,
            "backend": be.name,
        }
    return {
        "status": "ok",
        "history": history,
        "facts": facts,
        "summary": summary,
        "grounding": grounding,
        "trace": trace,
        "backend": be.name,
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="*")
    parser.add_argument("--points", default=None)
    parser.add_argument("--backend", choices=["julia", "rust", "numpy"], default="julia")
    args = parser.parse_args()
    request = " ".join(args.request) or "Cluster the cells into density zones and report strain risk under a 5 percent load"
    points = None
    if args.points:
        from mequa.ingest import load_points_csv

        points = load_points_csv(args.points)
    try:
        out = run_agent(request, points=points, backend=backend_mod.load(args.backend))
    except OSError:
        print("Ollama is not reachable. Start it and pull llama3.2:3b and llama-guard3:1b.")
        return
    out.pop("result", None)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
