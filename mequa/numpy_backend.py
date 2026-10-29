import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.special import expit

name = "numpy"


def synthetic_cells(n=2000, seed=1, clusters=3):
    rng = np.random.default_rng(seed)
    centers = rng.random((clusters, 2)) * 10
    pick = rng.integers(0, clusters, n)
    return centers[pick] + 0.8 * rng.standard_normal((n, 2))


def kmeans_zones(points, k, iters=100, seed=1):
    rng = np.random.default_rng(seed)
    n = len(points)
    centroids = points[rng.permutation(n)[:k]].copy()
    labels = np.full(n, -1)
    for _ in range(iters):
        dist = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = points[mask].mean(axis=0)
    return labels + 1, centroids


def build_knn_edges(points, k):
    n = len(points)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    a = np.repeat(np.arange(n), k)
    b = idx[:, 1:].ravel()
    keep = a != b
    a = a[keep]
    b = b[keep]
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys = np.unique(lo * n + hi)
    return keys // n, keys % n


def relax_strain(points, ei, ej, load=0.05, band=0.1, iters=1500, step=0.05):
    n, d = points.shape
    rest = np.maximum(np.linalg.norm(points[ei] - points[ej], axis=1), 1e-9)
    x = points[:, 0]
    lo, hi = x.min(), x.max()
    width = hi - lo
    fixed = x <= lo + band * width
    pulled = (~fixed) & (x >= hi - band * width)
    free = ~(fixed | pulled)
    pos = points.copy()
    pos[pulled, 0] += load * width
    for _ in range(iters):
        diff = pos[ej] - pos[ei]
        length = np.linalg.norm(diff, axis=1)
        positive = length > 0
        f = np.zeros_like(length)
        f[positive] = (length[positive] - rest[positive]) / length[positive]
        fv = diff * f[:, None]
        force = np.empty_like(pos)
        for j in range(d):
            force[:, j] = np.bincount(ei, weights=fv[:, j], minlength=n) - np.bincount(
                ej, weights=fv[:, j], minlength=n
            )
        pos[free] += step * force[free]
    length = np.linalg.norm(pos[ej] - pos[ei], axis=1)
    edge_strain = np.abs(length - rest) / rest
    node_sum = np.bincount(ei, weights=edge_strain, minlength=n) + np.bincount(
        ej, weights=edge_strain, minlength=n
    )
    node_cnt = np.bincount(ei, minlength=n) + np.bincount(ej, minlength=n)
    node_strain = np.divide(node_sum, node_cnt, out=np.zeros(n), where=node_cnt > 0)
    return pos, node_strain, edge_strain


def sphi(node_strain, ei, ej, iters=300, tol=1e-10):
    n = len(node_strain)
    w = (node_strain[ei] + node_strain[ej]) / 2
    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    mat = coo_matrix((np.concatenate([w, w]), (rows, cols)), shape=(n, n)).tocsr()
    v = np.full(n, 1.0 / np.sqrt(n))
    lam = 0.0
    for _ in range(iters):
        wv = mat @ v
        nrm = np.linalg.norm(wv)
        if nrm == 0:
            return 0.0
        v = wv / nrm
        new = float(v @ (mat @ v))
        if abs(new - lam) < tol:
            return new
        lam = new
    return lam


def analyze(points, k=3, knn=6, load=0.05):
    pts = np.ascontiguousarray(points, dtype=np.float64)
    labels, centroids = kmeans_zones(pts, k)
    ei, ej = build_knn_edges(pts, knn)
    pos, node_strain, _ = relax_strain(pts, ei, ej, load=load)
    index = sphi(node_strain, ei, ej)
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
