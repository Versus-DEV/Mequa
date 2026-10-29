import numpy as np
from scipy.special import expit

import mequa_rs
from mequa import numpy_backend as nb

name = "rust"
synthetic_cells = nb.synthetic_cells


def as_f64_matrix(data):
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.ascontiguousarray(data, dtype=np.float64)


def analyze(points, k=3, knn=6, load=0.05):
    pts = as_f64_matrix(points)
    labels, centroids = nb.kmeans_zones(pts, k)
    ei, ej = mequa_rs.knn_edges(pts, knn)
    pos, node_strain, _ = mequa_rs.relax_strain(pts, ei, ej, load, 0.1, 1500, 0.05)
    index = mequa_rs.sphi(node_strain, ei, ej)
    ordered = np.sort(node_strain)
    s_crit = float(ordered[min(max(int(np.ceil(0.9 * len(ordered))), 1), len(ordered)) - 1])
    failure = expit((node_strain - s_crit) / max(s_crit * 0.25, 1e-9))
    return {
        "labels": labels,
        "centroids": centroids,
        "positions": pos,
        "node_strain": node_strain,
        "failure_prob": failure,
        "sphi": float(index),
        "s_crit": s_crit,
        "n_edges": int(len(ei)),
    }
