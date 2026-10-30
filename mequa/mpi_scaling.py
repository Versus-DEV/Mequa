"""Measure MPI scaling: runs mpi_sweep at several rank counts and writes results/mpi_scaling.json.

    python -m mequa.mpi_scaling --cells 3000 --loads 24 --repeats 3
Needs mpiexec on PATH (MS-MPI on Windows, Open MPI or MPICH elsewhere) and mpi4py installed.

Repeats are interleaved (1, 2, 4 ranks, then 1, 2, 4 again) so that a laptop that heats up or
throttles during the test affects every rank count alike. All raw times are saved, speedup uses
the median of the repeats, and a warning is printed when the repeats of the same rank count differ
by more than 20 percent, because then the speedup number is not trustworthy. For a clean run: plug
in, choose the best-performance power mode, close other programs, stop Streamlit and Ollama.
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=3000)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--loads", type=int, default=24)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/mpi_scaling.json")
    args = ap.parse_args()
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if not mpiexec:
        sys.exit("mpiexec not found. Install MS-MPI (Windows) or Open MPI, then pip install mpi4py.")
    walls = {n: [] for n in args.ranks}
    last = {}
    for rep in range(args.repeats):
        for n in args.ranks:
            tmp = Path(args.out).with_name(f"_mpi_{n}_{rep}.json")
            cmd = [mpiexec, "-n", str(n), sys.executable, "-m", "mequa.mpi_sweep", "--cells", str(args.cells),
                   "--dim", str(args.dim), "--loads", str(args.loads), "--out", str(tmp)]
            subprocess.run(cmd, check=True, capture_output=True)
            data = json.loads(tmp.read_text())
            tmp.unlink()
            walls[n].append(data["wall_seconds"])
            last[n] = data
            print(f"repeat {rep + 1} ranks={n} wall={data['wall_seconds']}s")
    base = statistics.median(walls[args.ranks[0]])
    ref = last[args.ranks[0]]["rows"]
    summary = []
    for n in args.ranks:
        med = statistics.median(walls[n])
        spread = (max(walls[n]) - min(walls[n])) / med if med else 0.0
        diff = max(abs(a[k] - b[k]) for a, b in zip(last[n]["rows"], ref) for k in ref[0])
        summary.append({
            "ranks": n,
            "wall_seconds_median": round(med, 3),
            "wall_seconds_all": walls[n],
            "spread_pct": round(100 * spread, 1),
            "speedup": round(base / med, 2),
            "efficiency": round(base / med / n, 2),
            "max_abs_diff_vs_first": diff,
            "noisy": spread > 0.2,
        })
    noisy = [s["ranks"] for s in summary if s["noisy"]]
    if noisy:
        print(f"WARNING: repeats differ by more than 20% for ranks {noisy}; the machine was not steady. Re-run on power with other programs closed.")
    out = {"cells": args.cells, "dim": args.dim, "loads": args.loads, "repeats": args.repeats,
           "machine": last[args.ranks[0]]["machine"], "scaling": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
