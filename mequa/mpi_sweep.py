"""MPI load sweep CLI.  mpiexec -n 4 python -m mequa.mpi_sweep --cells 3000 --loads 24 --verify"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per rank, otherwise ranks fight over cores

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from mequa import numpy_backend, parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=3000)
    ap.add_argument("--dim", type=int, choices=[2, 3], default=2)
    ap.add_argument("--loads", type=int, default=24)
    ap.add_argument("--max-load", type=float, default=0.12)
    ap.add_argument("--strain-limit", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify", action="store_true", help="also solve serially on rank 0 and compare")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comm = parallel.get_comm()
    rank = 0 if comm is None else comm.Get_rank()
    points = loads = None
    if rank == 0:
        points = numpy_backend.synthetic_cells(args.cells, seed=args.seed, dim=args.dim)
        loads = np.linspace(0.01, args.max_load, args.loads).tolist()
    result = parallel.sweep_loads(points, loads, comm=comm)
    if rank != 0:
        return
    rows = result["rows"]
    result["threshold_load_p95"] = parallel.find_threshold(rows, args.strain_limit)
    result["strain_limit"] = args.strain_limit
    result["machine"] = {"node": platform.node(), "processor": platform.processor(), "python": platform.python_version()}
    if args.verify:
        t0 = time.perf_counter()
        serial_rows, _ = parallel.compute_rows(points, loads, list(range(len(loads))))
        serial = [r for _, r in serial_rows]
        result["serial_seconds"] = round(time.perf_counter() - t0, 4)
        diff = max(abs(a[k] - b[k]) for a, b in zip(rows, serial) for k in parallel.ROW_KEYS)
        result["max_abs_diff_vs_serial"] = diff
        result["speedup_vs_serial_in_same_run"] = round(result["serial_seconds"] / result["wall_seconds"], 2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
