"""Llama as a tool-using scientist: it calls the verified solver, reads the numbers, and decides
what to run next. Nothing here computes physics; every number comes from mequa.session / parallel.

Guard rails: the request passes the firewall first and the final reply passes it again; tool
arguments are validated and clamped; the loop is bounded; the final summary must pass the
number-grounding check or it is rewritten, and after repeated failure replaced by a template.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mequa import parallel
from mequa.agents.llama_tool_schemas import get_solver_tools
from mequa.agents.number_grounding import ungrounded_numbers
from mequa.agents.ollama_client import chat

SYSTEM = (
    "You are a computational scientist using a cell spring-network solver for research datasets. "
    "Work by calling tools, then answer. Strategy: first call run_solver once at a moderate load to see the "
    "zones and the strain pattern; if the question is about a failure threshold, call sweep_load. "
    "The strain limit used by sweep_load is {limit}. If a result reports load_path false, the pulled edge is not "
    "connected to the held edge: call run_solver again with more neighbors before trusting any strain value. "
    "Final answer: 3 to 6 sentences, using only numbers that appear in tool results, no medical advice, "
    "no statements about individuals, no qualitative risk labels."
)

TEMPLATE_MIN = "No solver result was produced, so there is nothing to report."


def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _parse_args(raw) -> Dict[str, Any]:
    """Ollama returns arguments as an object; some servers return a JSON string. Accept both."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except ValueError:
            return {}
    return {}


def flatten_facts(trace: List[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> List[Any]:
    facts: List[Any] = [extra or {}]
    for step in trace:
        if step.get("type") == "tool_call":
            facts.append(step.get("arguments"))
            facts.append(step.get("result"))
    return facts


def trace_summary(trace: List[Dict[str, Any]], extra: Dict[str, Any]) -> str:
    """Deterministic summary built only from tool results (used when the model keeps inventing numbers)."""
    runs = [s["result"] for s in trace if s.get("type") == "tool_call" and s["tool"] == "run_solver" and "sphi" in s["result"]]
    sweeps = [s["result"] for s in trace if s.get("type") == "tool_call" and s["tool"] == "sweep_load" and "rows" in s["result"]]
    if not runs and not sweeps:
        return TEMPLATE_MIN
    parts = [f"The dataset has {extra['cells']} cells in {extra['dim']} dimensions."]
    if runs:
        r = runs[-1]
        parts.append(
            f"At a load fraction of {r['load_fraction']}, SPHI is {r['sphi']}, mean strain is {r['mean_strain']} "
            f"and maximum strain is {r['max_strain']}."
        )
        if r.get("load_path") is False:
            parts.append("The pulled edge is not mechanically connected to the held edge, so this strain is not a loaded equilibrium.")
    if sweeps:
        s = sweeps[-1]
        if s["threshold_load"] is None:
            parts.append(f"The 95th percentile strain did not reach {s['strain_limit']} between loads {s['rows'][0]['load_fraction']} and {s['rows'][-1]['load_fraction']}.")
        else:
            parts.append(f"The 95th percentile strain reaches {s['strain_limit']} at a load fraction of {s['threshold_load']}.")
    return " ".join(parts)


class AdaptiveInvestigator:
    def __init__(
        self,
        session,
        model: Optional[str] = None,
        max_iterations: int = 6,
        strain_limit: float = 0.05,
        chat_fn: Optional[Callable] = None,
        check_request: Optional[Callable] = None,
        check_reply: Optional[Callable] = None,
        max_grounding_retries: int = 2,
    ) -> None:
        self.session = session
        self.model = model or os.getenv("PLANNER_MODEL", "llama3.2:3b")
        self.max_iterations = max_iterations
        self.strain_limit = strain_limit
        self.chat = chat_fn or chat
        if check_request is None or check_reply is None:
            from mequa import firewall

            check_request = check_request or firewall.check_request
            check_reply = check_reply or firewall.check_reply
        self.check_request = check_request
        self.check_reply = check_reply
        self.max_grounding_retries = max_grounding_retries
        self.trace: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ tools
    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "run_solver":
                load = _clamp(_num(args.get("load_fraction"), 0.05), 0.005, 0.2)
                k = int(_clamp(_num(args.get("zones"), self.session.k), 2, 8))
                knn = int(_clamp(_num(args.get("neighbors"), self.session.knn), 3, 14))
                return self.session.run(load, k=k, knn=knn)
            if name == "sweep_load":
                start = _clamp(_num(args.get("start"), 0.01), 0.005, 0.2)
                stop = _clamp(_num(args.get("stop"), 0.12), 0.005, 0.2)
                if stop <= start:
                    return {"error": "stop must be greater than start"}
                steps = int(_clamp(_num(args.get("steps"), 8), 3, 16))
                loads = np.linspace(start, stop, steps).tolist()
                out = parallel.sweep_loads(
                    self.session.points, loads, k=self.session.k, knn=self.session.knn,
                    stiffness=self.session.stiffness, comm=None,
                )
                rows = [{k_: round(v, 5) for k_, v in r.items()} for r in out["rows"]]
                return {
                    "rows": rows,
                    "threshold_load": parallel.find_threshold(rows, self.strain_limit),
                    "strain_limit": self.strain_limit,
                    "solves": len(rows),
                }
            return {"error": f"unknown tool {name}"}
        except Exception as exc:  # a failing tool must not end the investigation
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ------------------------------------------------------------------- loop
    def run(self, question: str) -> Dict[str, Any]:
        self.trace = []
        verdict = self.check_request(question)
        if not verdict.allowed:
            return {"status": "blocked", "blocked_by": verdict.blocked_by, "trace": self.trace}
        extra = {"cells": int(len(self.session.points)), "dim": self.session.dim, "strain_limit": self.strain_limit}
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM.format(limit=self.strain_limit)},
            {"role": "user", "content": f"{question}\nDataset: {extra['cells']} cells, {extra['dim']}D."},
        ]
        tools = get_solver_tools()
        nudged = False
        final_text = None
        for iteration in range(1, self.max_iterations + 1):
            resp = self.chat(messages=messages, model=self.model, tools=tools, temperature=0.0, max_tokens=700)
            if "error" in resp:
                return {"status": "model_unavailable", "error": resp["error"], "trace": self.trace}
            msg = resp.get("message", {})
            calls = msg.get("tool_calls") or []
            if not calls:
                have_result = any(s.get("type") == "tool_call" for s in self.trace)
                if not have_result and not nudged:
                    nudged = True
                    messages.append({"role": "assistant", "content": msg.get("content", "")})
                    messages.append({"role": "user", "content": "Call run_solver first, then answer from its result."})
                    continue
                final_text = (msg.get("content") or "").strip()
                self.trace.append({"iteration": iteration, "type": "final", "content": final_text})
                break
            messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {})
                name, args = fn.get("name", ""), _parse_args(fn.get("arguments"))
                result = self.execute(name, args)
                self.trace.append({"iteration": iteration, "type": "tool_call", "tool": name, "arguments": args, "result": result})
                messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
        if final_text is None:
            final_text = ""  # ran out of iterations without a final answer
        return self._finish(question, final_text, messages, extra)

    def _finish(self, question, text, messages, extra):
        facts = flatten_facts(self.trace, extra)
        attempts = []
        for attempt in range(self.max_grounding_retries + 1):
            bad = ungrounded_numbers(text, facts) if text else ["<empty summary>"]
            attempts.append({"attempt": attempt + 1, "ungrounded": bad})
            if not bad:
                break
            if attempt == self.max_grounding_retries:
                text = ""
                break
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"These numbers are not in the tool results: {bad}. Rewrite the answer using only numbers from the tool results."})
            resp = self.chat(messages=messages, model=self.model, tools=None, temperature=0.0, max_tokens=500)
            if "error" in resp:
                text = ""
                break
            text = (resp.get("message", {}).get("content") or "").strip()
        fallback = False
        if not text:
            text, fallback = trace_summary(self.trace, extra), True
        verdict = self.check_reply(question, text)
        if not verdict.allowed:
            return {"status": "reply_blocked", "blocked_by": verdict.blocked_by, "trace": self.trace}
        return {
            "status": "ok",
            "summary": text,
            "trace": self.trace,
            "iterations": max((s["iteration"] for s in self.trace), default=0),
            "grounding": {"attempts": attempts, "fallback": fallback},
            "model": self.model,
        }
