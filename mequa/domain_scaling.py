"""Measure the domain-decomposed solve at several rank counts, with repeats and a serial check.

    python -m mequa.domain_scaling --cells 20000 --dim 3 --ranks 1 2 4 8 --repeats 3

Each run launches mequa.domain_mpi with --verify, so every measurement also compares the result
with the serial solve and times the serial solve in the same process. Repeats are interleaved
across rank counts; speedup uses medians; runs whose repeats differ by more than 20 percent are
flagged, because then the speedup is not trustworthy. For a clean run: plug in, best-performance
power mode, close other programs, stop Streamlit and Ollama.
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
    ap.add_argument("--cells", type=int, default=20000)
    ap.add_argument("--dim", type=int, choices=[2, 3], default=3)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/domain_scaling.json")
    args = ap.parse_args()
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if not mpiexec:
        sys.exit("mpiexec not found. Install MS-MPI (Windows) or Open MPI, then pip install mpi4py.")
    runs = {n: [] for n in args.ranks}
    for rep in range(args.repeats):
        for n in args.ranks:
            tmp = Path(args.out).with_name(f"_dom_{n}_{rep}.json")
            cmd = [mpiexec, "-n", str(n), sys.executable, "-m", "mequa.domain_mpi", "--cells", str(args.cells), "--dim", str(args.dim),
                   "--iters", str(args.iters), "--verify", "--out", str(tmp)]
            subprocess.run(cmd, check=True, capture_output=True)
            d = json.loads(tmp.read_text())
            tmp.unlink()
            runs[n].append(d)
            print(f"repeat {rep + 1} ranks={n} wall={d['wall_seconds']}s serial={d['serial_seconds']}s "
                  f"max position diff={d['max_abs_position_diff_vs_serial']}")
    serial_med = statistics.median(d["serial_seconds"] for n in args.ranks for d in runs[n])
    one = statistics.median(d["wall_seconds"] for d in runs[args.ranks[0]])
    summary = []
    for n in args.ranks:
        walls = [d["wall_seconds"] for d in runs[n]]
        med = statistics.median(walls)
        spread = (max(walls) - min(walls)) / med if med else 0.0
        summary.append({
            "ranks": n, "wall_seconds_median": round(med, 3), "wall_seconds_all": walls, "spread_pct": round(100 * spread, 1),
            "speedup_vs_serial": round(serial_med / med, 2), f"speedup_vs_{args.ranks[0]}_rank": round(one / med, 2),
            "efficiency_vs_serial": round(serial_med / med / n, 2),
            "max_abs_position_diff_vs_serial": max(d["max_abs_position_diff_vs_serial"] for d in runs[n]),
            "max_abs_strain_diff_vs_serial": max(d["max_abs_strain_diff_vs_serial"] for d in runs[n]),
            "halo_nodes_rank0": runs[n][-1]["halo_nodes_rank0"], "noisy": spread > 0.2,
        })
    if any(s["noisy"] for s in summary):
        print("WARNING: repeats differ by more than 20% for some rank counts; re-run on power with other programs closed.")
    out = {"cells": args.cells, "dim": args.dim, "iters": args.iters, "repeats": args.repeats,
           "serial_seconds_median": round(serial_med, 3), "scaling": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
