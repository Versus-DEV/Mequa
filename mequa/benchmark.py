import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from mequa import numpy_backend as nb


def best_of(fn, repeats):
    best = float("inf")
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


def numpy_stages(points, k, knn, load, repeats):
    t_km, _ = best_of(lambda: nb.kmeans_zones(points, k), repeats)
    t_knn, edges = best_of(lambda: nb.build_knn_edges(points, knn), repeats)
    ei, ej = edges
    t_relax, relaxed = best_of(lambda: nb.relax_strain(points, ei, ej, load=load), repeats)
    node_strain = relaxed[1]
    t_sphi, value = best_of(lambda: nb.sphi(node_strain, ei, ej), repeats)
    return {"kmeans": t_km, "knn": t_knn, "relax": t_relax, "sphi": t_sphi, "sphi_value": float(value)}


def rust_stages(rs, points, k, knn, load, repeats):
    pts = np.ascontiguousarray(points, dtype=np.float64)
    t_km, _ = best_of(lambda: nb.kmeans_zones(pts, k), repeats)
    t_knn, edges = best_of(lambda: rs.knn_edges(pts, knn), repeats)
    ei, ej = edges
    t_relax, relaxed = best_of(lambda: rs.relax_strain(pts, ei, ej, load, 0.1, 1500, 0.05), repeats)
    node_strain = relaxed[1]
    t_sphi, value = best_of(lambda: rs.sphi(node_strain, ei, ej), repeats)
    return {"kmeans": t_km, "knn": t_knn, "relax": t_relax, "sphi": t_sphi, "sphi_value": float(value)}


def julia_stages(jb, points, k, knn, load, repeats):
    to_mat = jb.jl.seval("Matrix{Float64}")
    mat = to_mat(np.ascontiguousarray(points, dtype=np.float64))
    best = None
    for _ in range(repeats):
        r = jb.jl.MequaBench.stage_times(mat, k, knn, load)
        cur = {
            "kmeans": float(r.kmeans),
            "knn": float(r.knn),
            "relax": float(r.relax),
            "sphi": float(r.sphi_seconds),
            "sphi_value": float(r.sphi_value),
        }
        if best is None:
            best = cur
        else:
            for key in ("kmeans", "knn", "relax", "sphi"):
                best[key] = min(best[key], cur[key])
    return best


def total(stages):
    return stages["kmeans"] + stages["knn"] + stages["relax"] + stages["sphi"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numpy-only", action="store_true")
    parser.add_argument("--no-julia", action="store_true")
    parser.add_argument("--no-rust", action="store_true")
    args = parser.parse_args()

    jb = None
    rs = None
    threads = None
    if not args.numpy_only and not args.no_rust:
        try:
            import mequa_rs as rs
        except ImportError:
            print("mequa_rs not installed; skipping the Rust column")
    if not args.numpy_only and not args.no_julia:
        from mequa import julia_bridge as jb

        solvers = Path(__file__).resolve().parent.parent / "solvers"
        jb.jl.include(str(solvers / "Bench.jl"))
        threads = int(jb.jl.seval("Threads.nthreads()"))
        warm = nb.synthetic_cells(300)
        julia_stages(jb, warm, 3, 6, 0.05, 1)

    k, knn, load = 3, 6, 0.05
    rows = []
    for n in args.sizes:
        pts = nb.synthetic_cells(n)
        np_stages = numpy_stages(pts, k, knn, load, args.repeats)
        row = {"cells": n, "numpy": np_stages, "numpy_total_s": total(np_stages)}
        if rs is not None:
            rs_stages = rust_stages(rs, pts, k, knn, load, args.repeats)
            row["rust"] = rs_stages
            row["rust_total_s"] = total(rs_stages)
            row["rust_speedup_vs_numpy"] = row["numpy_total_s"] / row["rust_total_s"]
            row["rust_sphi_relative_difference"] = abs(np_stages["sphi_value"] - rs_stages["sphi_value"]) / max(abs(np_stages["sphi_value"]), 1e-12)
        if jb is not None:
            jl_stages = julia_stages(jb, pts, k, knn, load, args.repeats)
            row["julia"] = jl_stages
            row["julia_total_s"] = total(jl_stages)
            row["speedup_total"] = row["numpy_total_s"] / row["julia_total_s"]
            denom = max(abs(np_stages["sphi_value"]), 1e-12)
            row["sphi_relative_difference"] = abs(np_stages["sphi_value"] - jl_stages["sphi_value"]) / denom
        rows.append(row)
        line = f"n={n:>6}  numpy {row['numpy_total_s']:.3f}s"
        if rs is not None:
            line += f"  rust {row['rust_total_s']:.3f}s ({row['rust_speedup_vs_numpy']:.2f}x vs numpy, sphi diff {row['rust_sphi_relative_difference']:.1e})"
        if jb is not None:
            line += f"  julia {row['julia_total_s']:.3f}s  speedup {row['speedup_total']:.2f}x  sphi diff {row['sphi_relative_difference']:.2e}"
        print(line)

    report = {
        "machine": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "julia_threads": threads,
        "rust_threads": rs.num_threads() if rs is not None else None,
        "settings": {"zones": k, "neighbors": knn, "load": load, "repeats": args.repeats, "stat": "best of repeats"},
        "baseline_note": "NumPy baseline is vectorized with scipy cKDTree for neighbors; the Rust column uses NumPy k-means plus Rust kNN, relaxation and SPHI; speedup below 1 means NumPy was faster.",
        "rows": rows,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved results/benchmark.json")


if __name__ == "__main__":
    main()
