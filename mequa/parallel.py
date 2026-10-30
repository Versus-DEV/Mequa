"""Distributed load sweeps with MPI (mpi4py), with a serial fallback.

What is parallelised: a sweep is many independent relaxations (one per load value). Rank r takes
loads r, r+size, r+2*size, ... , every rank solves its share with the NumPy spring-network
backend, and rank 0 gathers the rows. No domain decomposition and no halo exchange: that is a
deliberate scope limit, stated in the README. Each rank is a separate process, so the speed-up
comes from using more CPU cores, not from a faster kernel.

Run it:  mpiexec -n 4 python -m mequa.mpi_sweep --cells 3000 --loads 24 --verify
Without mpi4py, or with one rank, the same code runs serially and says so.
"""

import time

import numpy as np

from mequa import numpy_backend
from mequa.session import summarize_result

ROW_KEYS = ("load_fraction", "sphi", "mean_strain", "p95_strain", "max_strain")


def get_comm():
    """Return the MPI world communicator, or None when mpi4py is not installed."""
    try:
        from mpi4py import MPI
    except Exception:
        return None
    return MPI.COMM_WORLD


def plan(n_items, rank, size):
    """Indices of the sweep this rank solves (round robin)."""
    return list(range(rank, n_items, size))


def _solve_one(points, load, k, knn, stiffness):
    res = numpy_backend.analyze(points, k=k, knn=knn, load=load, stiffness=stiffness)
    row = summarize_result(res, load, k)
    return {key: row[key] for key in ROW_KEYS}


def compute_rows(points, loads, indices, k=3, knn=6, stiffness=None):
    t0 = time.perf_counter()
    rows = [(i, _solve_one(points, loads[i], k, knn, stiffness)) for i in indices]
    return rows, time.perf_counter() - t0


def merge(parts, n_items):
    """Combine per-rank (index, row) lists into one list ordered by load index."""
    out = [None] * n_items
    for part in parts:
        for i, row in part:
            if out[i] is not None:
                raise ValueError(f"load index {i} was solved by more than one rank")
            out[i] = row
    missing = [i for i, r in enumerate(out) if r is None]
    if missing:
        raise ValueError(f"load indices never solved: {missing[:5]}")
    return out


def sweep_loads(points=None, loads=None, k=3, knn=6, stiffness=None, comm=None):
    """Solve every load in `loads`. Returns a dict on rank 0 and None on the other ranks.

    On a multi-rank communicator only rank 0 needs to pass points, loads and stiffness; they are
    broadcast to the other ranks.
    """
    size = 1 if comm is None else comm.Get_size()
    rank = 0 if comm is None else comm.Get_rank()
    if size > 1:
        points, loads, stiffness, k, knn = comm.bcast((points, loads, stiffness, k, knn), root=0)
        comm.Barrier()
    points = np.ascontiguousarray(points, dtype=np.float64)
    loads = [float(x) for x in loads]
    t0 = time.perf_counter()
    mine, busy = compute_rows(points, loads, plan(len(loads), rank, size), k, knn, stiffness)
    if size > 1:
        gathered = comm.gather((mine, busy), root=0)
        if rank != 0:
            return None
        parts = [g[0] for g in gathered]
        per_rank = [round(g[1], 4) for g in gathered]
    else:
        parts, per_rank = [mine], [round(busy, 4)]
    wall = time.perf_counter() - t0
    return {
        "rows": merge(parts, len(loads)),
        "ranks": size,
        "mode": "mpi" if size > 1 else "serial",
        "wall_seconds": round(wall, 4),
        "per_rank_busy_seconds": per_rank,
        "cells": int(len(points)),
        "dim": int(points.shape[1]),
    }


def find_threshold(rows, strain_limit, key="p95_strain"):
    """Smallest load at which `key` reaches strain_limit, by linear interpolation between sweep
    points. Returns None when the limit is never reached within the swept range. The limit is a
    user-chosen modelling assumption, not a measured tissue property."""
    for a, b in zip(rows, rows[1:]):
        if a[key] >= strain_limit:
            return a["load_fraction"]
        if b[key] >= strain_limit:
            span = b[key] - a[key]
            frac = 0.0 if span <= 0 else (strain_limit - a[key]) / span
            return round(a["load_fraction"] + frac * (b["load_fraction"] - a["load_fraction"]), 6)
    if rows and rows[0][key] >= strain_limit:
        return rows[0]["load_fraction"]
    return None
