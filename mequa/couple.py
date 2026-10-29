import numpy as np


def drive_strain(times, signal, points, backend, base_load=0.08, samples=20, k=3, knn=6):
    smin = float(np.min(signal))
    smax = float(np.max(signal))
    span = max(smax - smin, 1e-12)
    idx = np.unique(np.linspace(0, len(times) - 1, samples).astype(int))
    lo = 0.01
    rows = []
    for i in idx:
        level = (signal[i] - smin) / span
        load = lo + (base_load - lo) * level
        res = backend.analyze(points, k=k, knn=knn, load=load)
        rows.append({
            "time": float(times[i]),
            "signal": float(signal[i]),
            "load": float(load),
            "sphi": float(res["sphi"]),
            "mean_strain": float(np.mean(res["node_strain"])),
            "peak_strain": float(np.max(res["node_strain"])),
        })
    return rows
