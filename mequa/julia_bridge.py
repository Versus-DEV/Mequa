import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

from juliacall import Main as jl

_SOLVERS = Path(__file__).resolve().parent.parent / "solvers"
jl.include(str(_SOLVERS / "MequaCore.jl"))
core = jl.MequaCore


def synthetic_cells(n=2000, seed=1, clusters=3):
    return np.array(core.synthetic_cells(n, seed=seed, clusters=clusters))


def analyze(points, k=3, knn=6, load=0.05):
    pts = np.ascontiguousarray(points, dtype=np.float64)
    res = core.run_pipeline(pts, k, knn, load)
    return {
        "labels": np.array(res.labels),
        "centroids": np.array(res.centroids),
        "positions": np.array(res.positions),
        "node_strain": np.array(res.node_strain),
        "failure_prob": np.array(res.failure_prob),
        "sphi": float(res.sphi),
        "s_crit": float(res.s_crit),
        "n_edges": int(res.n_edges),
    }