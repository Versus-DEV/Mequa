"""Domain-decomposed strain relaxation: one mesh split across MPI ranks, halo exchange each step.

The existing mpi_sweep parallelises over loads (every rank solves a different load on the whole
mesh). This module parallelises ONE solve: nodes are split into contiguous slabs along x, each rank
updates only the nodes it owns, and after every relaxation step ranks swap the positions of the
nodes that sit across a slab boundary (the halo).

What is and is not decomposed: the mesh (k-nearest-neighbour edges) is built identically on every
rank; the relaxation, which dominates the cost, is distributed. The update is the same explicit
Jacobi step as numpy_backend.relax_strain, so results agree with the serial solve up to
floating-point summation order. In the tests the difference came out exactly 0.0 because each rank
keeps the global edge order, but the guarantee we state is agreement to rounding, not bit-identity.
No speedup is claimed: at laptop sizes the halo exchange in Python costs about what it saves.

  mpiexec -n 4 python -m mequa.domain_mpi --cells 20000 --dim 3 --verify
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np

from mequa import numpy_backend as nb


def partition(points, size):
    """Owner rank of every node: contiguous slabs along x with equal node counts."""
    order = np.argsort(points[:, 0], kind="stable")
    owner = np.empty(len(points), dtype=np.int64)
    for r, chunk in enumerate(np.array_split(order, size)):
        owner[chunk] = r
    return owner


def relax_strain_domain(points, ei, ej, comm, load=0.05, band=0.1, iters=1500, step=0.05, stiffness=None):
    """Same contract as numpy_backend.relax_strain. Every rank returns the full result."""
    rank, size = comm.Get_rank(), comm.Get_size()
    n, d = points.shape
    kedge = None if stiffness is None else nb.edge_stiffness(stiffness, ei, ej)
    rest = np.maximum(np.linalg.norm(points[ei] - points[ej], axis=1), 1e-9)
    x = points[:, 0]
    lo, hi = x.min(), x.max()
    width = hi - lo
    fixed = x <= lo + band * width
    pulled = (~fixed) & (x >= hi - band * width)
    free = ~(fixed | pulled)
    pos = points.copy()
    pos[pulled, 0] += load * width

    owner = partition(points, size)
    mine = owner == rank
    local_edges = np.flatnonzero(mine[ei] | mine[ej])
    lei, lej, lrest = ei[local_edges], ej[local_edges], rest[local_edges]
    lk = None if kedge is None else kedge[local_edges]

    ghost = np.unique(np.concatenate([lei[~mine[lei]], lej[~mine[lej]]])) if len(lei) else np.empty(0, dtype=np.int64)
    need = [ghost[owner[ghost] == r] for r in range(size)]
    send_ids = comm.alltoall(need)  # send_ids[r] = my nodes that rank r needs
    updatable = mine & free
    halo_nodes = int(sum(len(a) for a in need))

    for _ in range(iters):
        diff = pos[lej] - pos[lei]
        length = np.linalg.norm(diff, axis=1)
        positive = length > 0
        f = np.zeros_like(length)
        f[positive] = (length[positive] - lrest[positive]) / length[positive]
        if lk is not None:
            f = f * lk
        fv = diff * f[:, None]
        force = np.zeros_like(pos)
        for j in range(d):
            force[:, j] = np.bincount(lei, weights=fv[:, j], minlength=n) - np.bincount(lej, weights=fv[:, j], minlength=n)
        pos[updatable] += step * force[updatable]
        outgoing = [pos[ids] for ids in send_ids]
        incoming = comm.alltoall(outgoing)
        for r in range(size):
            if len(need[r]):
                pos[need[r]] = incoming[r]

    parts = comm.allgather((np.flatnonzero(mine), pos[mine]))
    full = pos.copy()
    for idx, p in parts:
        full[idx] = p
    length = np.linalg.norm(full[ej] - full[ei], axis=1)
    edge_strain = np.abs(length - rest) / rest
    node_sum = np.bincount(ei, weights=edge_strain, minlength=n) + np.bincount(ej, weights=edge_strain, minlength=n)
    node_cnt = np.bincount(ei, minlength=n) + np.bincount(ej, minlength=n)
    node_strain = np.divide(node_sum, node_cnt, out=np.zeros(n), where=node_cnt > 0)
    relax_strain_domain.last_halo_nodes = halo_nodes
    return full, node_strain, edge_strain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=20000)
    ap.add_argument("--dim", type=int, choices=[2, 3], default=3)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--load", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    points = nb.synthetic_cells(args.cells, seed=args.seed, clusters=1, dim=args.dim) if rank == 0 else None
    points = comm.bcast(points, root=0)
    ei, ej = nb.build_knn_edges(points, args.knn)
    comm.Barrier()
    t0 = time.perf_counter()
    pos, ns, es = relax_strain_domain(points, ei, ej, comm, load=args.load, iters=args.iters)
    comm.Barrier()
    wall = time.perf_counter() - t0
    if rank != 0:
        return
    out = {"ranks": size, "cells": args.cells, "dim": args.dim, "edges": int(len(ei)), "iters": args.iters,
           "wall_seconds": round(wall, 3), "halo_nodes_rank0": int(relax_strain_domain.last_halo_nodes),
           "max_strain": float(ns.max())}
    if args.verify:
        t1 = time.perf_counter()
        spos, sns, _ = nb.relax_strain(points, ei, ej, load=args.load, iters=args.iters)
        out["serial_seconds"] = round(time.perf_counter() - t1, 3)
        out["max_abs_position_diff_vs_serial"] = float(np.abs(pos - spos).max())
        out["max_abs_strain_diff_vs_serial"] = float(np.abs(ns - sns).max())
    print(json.dumps(out, indent=2))
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
