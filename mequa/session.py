"""One analysis session: cell points (2D or 3D), optional per-cell stiffness, and the solver.

Every consumer (dashboard pages, the Llama investigator, the MPI sweep) goes through this class,
so there is exactly one solver path and it is the verified one (Julia, Rust or NumPy spring
network from mequa.backend). Results are plain floats so they can be passed to an LLM and logged.
"""

import numpy as np

from mequa import backend as backend_mod


def summarize_result(res, load, k):
    strain = np.asarray(res["node_strain"], dtype=float)
    labels = np.asarray(res["labels"])
    fail = np.asarray(res["failure_prob"], dtype=float)
    zones = []
    for z in sorted(set(int(v) for v in labels)):
        m = labels == z
        zones.append({
            "zone": z,
            "cells": int(m.sum()),
            "mean_strain": round(float(strain[m].mean()), 5),
            "max_strain": round(float(strain[m].max()), 5),
            "share_high_failure_prob": round(float((fail[m] > 0.5).mean()), 4),
        })
    return {
        "load_fraction": round(float(load), 6),
        "zones": int(k),
        "cells": int(len(strain)),
        "dim": int(res.get("dim", 2)),
        "mechanical_links": int(res["n_edges"]),
        "sphi": round(float(res["sphi"]), 6),
        "mean_strain": round(float(strain.mean()), 5),
        "p95_strain": round(float(np.percentile(strain, 95)), 5),
        "max_strain": round(float(strain.max()), 5),
        "s_crit": round(float(res["s_crit"]), 5),
        "per_zone": zones,
        "backend_used": res.get("backend_used", "unknown"),
        "n_components": res.get("n_components"),
        "load_path": res.get("load_path"),
        "residual": None if res.get("residual") is None else float(f"{res['residual']:.3g}"),
    }


def ensure_connectivity(res, points, knn):
    """Julia and Rust results do not carry the graph check; compute it on the same edge set."""
    if res.get("load_path") is None:
        from mequa import numpy_backend as nb

        ei, ej = nb.build_knn_edges(np.asarray(points, dtype=np.float64), knn)
        res["n_components"], res["load_path"] = nb.connectivity(np.asarray(points, dtype=np.float64), ei, ej)
    return res


class Session:
    def __init__(self, points, stiffness=None, backend=None, k=3, knn=6):
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        if stiffness is not None:
            stiffness = np.ascontiguousarray(stiffness, dtype=np.float64)
            if stiffness.shape != (len(self.points),):
                raise ValueError(
                    f"stiffness has {stiffness.shape} values but there are {len(self.points)} cells; "
                    "it must be one value per cell"
                )
        self.stiffness = stiffness
        self.backend = backend or backend_mod.load("numpy")
        self.k = k
        self.knn = knn

    @property
    def dim(self):
        return int(self.points.shape[1])

    def run(self, load, k=None, knn=None, full=False):
        k = k or self.k
        knn = knn or self.knn
        res = backend_mod.analyze_routed(self.backend, self.points, k=k, knn=knn, load=load, stiffness=self.stiffness)
        ensure_connectivity(res, self.points, knn)
        out = summarize_result(res, load, k)
        return (out, res) if full else out
