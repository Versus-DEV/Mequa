import json
import sys

import numpy as np

from mequa.firewall import _ollama_chat, check_request, check_reply
from mequa.julia_bridge import analyze, synthetic_cells

PLANNER_MODEL = "llama3.2:3b"

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
    "Use only the numbers provided. Do not interpret results clinically, give medical advice, "
    "or refer to individual patients. Write 3 to 4 sentences."
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
    raw = _ollama_chat(PLANNER_MODEL, messages, fmt="json")
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


def critique(result, n_cells):
    issues = []
    sizes = np.bincount(result["labels"])[1:]
    if sizes.size and sizes.min() < 0.05 * n_cells:
        issues.append("a zone holds under 5 percent of cells; try fewer zones")
    if not np.isfinite(result["sphi"]):
        issues.append("SPHI is not finite")
    return issues, [int(s) for s in sizes]


def summarize(request, entry):
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {
            "role": "user",
            "content": f"Request: {request}\nResults: {json.dumps(entry)}",
        },
    ]
    return _ollama_chat(PLANNER_MODEL, messages)


def run_agent(request, points=None, max_rounds=3):
    verdict = check_request(request)
    if not verdict.allowed:
        return {"status": "blocked", "blocked_by": verdict.blocked_by, "categories": verdict.categories}
    if points is None:
        points = synthetic_cells()
    n_cells = len(points)
    history = []
    plan = plan_experiment(request, n_cells, history)
    for _ in range(max_rounds):
        result = analyze(points, k=plan["zones"], knn=plan["neighbors"], load=plan["load"])
        issues, sizes = critique(result, n_cells)
        history.append({
            "zones": plan["zones"],
            "neighbors": plan["neighbors"],
            "load": plan["load"],
            "zone_sizes": sizes,
            "sphi": round(float(result["sphi"]), 6),
            "issues": issues,
        })
        if not issues:
            break
        plan = plan_experiment(request, n_cells, history)
    summary = summarize(request, history[-1])
    reply_verdict = check_reply(request, summary)
    if not reply_verdict.allowed:
        return {"status": "reply_blocked", "blocked_by": reply_verdict.blocked_by, "history": history}
    return {"status": "ok", "history": history, "summary": summary, "flags": verdict.flags + reply_verdict.flags}


def main():
    request = " ".join(sys.argv[1:]) or "Cluster the cells into density zones and report strain risk under a 5 percent load"
    try:
        out = run_agent(request)
    except OSError:
        print("Ollama is not reachable. Start it and pull llama3.2:3b and llama-guard3:1b.")
        return
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()