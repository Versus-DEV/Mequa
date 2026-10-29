from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mequa import agent as agent_mod
from mequa import backend as backend_mod
from mequa import simulink
from mequa.firewall import check_request

mcp = FastMCP("mequa")
_backends = {}


def _backend(kind):
    if kind not in _backends:
        _backends[kind] = backend_mod.load(kind)
    return _backends[kind]


@mcp.tool()
def compliance_check(text: str) -> dict:
    """Check a request with the Mequa compliance firewall (rules, Llama Guard, Llama intent classifier). Returns whether it is allowed and which layers blocked it."""
    verdict = check_request(text)
    return {
        "allowed": verdict.allowed,
        "blocked_by": verdict.blocked_by,
        "categories": verdict.categories,
        "degraded": verdict.degraded,
    }


@mcp.tool()
def analyze_cells(cells: int = 2000, zones: int = 3, neighbors: int = 6, load: float = 0.05, backend: str = "numpy") -> dict:
    """Run the strain analysis on synthetic cells. backend is one of julia, rust or numpy. Returns SPHI, critical strain and per-zone statistics."""
    be = _backend(backend)
    points = be.synthetic_cells(max(100, min(int(cells), 20000)))
    result = be.analyze(points, k=zones, knn=neighbors, load=load)
    return {
        "backend": be.name,
        "cells": int(len(points)),
        "sphi": float(result["sphi"]),
        "s_crit": float(result["s_crit"]),
        "mechanical_links": int(result["n_edges"]),
        "zones": agent_mod.zone_stats(result),
    }


@mcp.tool()
def simulate_simulink(model_path: str, variables: str = "", t_end: float = 6.0) -> dict:
    """Import a Simulink .slx file without MATLAB and simulate it. variables is a string such as 'R=1.0, C=0.3'. Returns summary statistics of each recorded signal."""
    path = Path(model_path)
    if path.suffix.lower() != ".slx" or not path.is_file():
        return {"error": "model_path must point to an existing .slx file"}
    try:
        model = simulink.load_slx(path)
        sim = simulink.simulate(model, t_end=max(0.1, min(float(t_end), 60.0)), variables=simulink.parse_assignments(variables))
    except simulink.ModelError as exc:
        return {"error": str(exc)}
    return {
        "model": model.name,
        "blocks": len(model.blocks),
        "signals": {
            name: {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}
            for name, v in sim.signals.items()
        },
    }


@mcp.tool()
def run_experiment(request: str, backend: str = "numpy") -> dict:
    """Run the full Mequa agent: the request passes the firewall, Llama plans the experiment, the solver runs, Llama summarizes, and the reply is checked. Requires Ollama with llama3.2:3b and llama-guard3:1b."""
    try:
        out = agent_mod.run_agent(request, backend=_backend(backend))
    except OSError:
        return {"error": "Ollama is not reachable"}
    out.pop("result", None)
    return out


if __name__ == "__main__":
    mcp.run()
