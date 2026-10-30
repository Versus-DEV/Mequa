import copy
import threading

import numpy as np

from mequa import numpy_backend as nb
from mequa import parallel


class Hub:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.box = None
        self.slots = [None] * size


class FakeComm:
    """Thread-based stand-in for an MPI communicator (tests the logic, not real MPI)."""

    def __init__(self, hub, rank):
        self.hub, self.rank = hub, rank

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.hub.size

    def Barrier(self):
        self.hub.barrier.wait()

    def bcast(self, obj, root=0):
        if self.rank == root:
            self.hub.box = copy.deepcopy(obj)
        self.hub.barrier.wait()
        out = copy.deepcopy(self.hub.box)
        self.hub.barrier.wait()
        return out

    def gather(self, obj, root=0):
        self.hub.slots[self.rank] = obj
        self.hub.barrier.wait()
        out = list(self.hub.slots) if self.rank == root else None
        self.hub.barrier.wait()
        return out


def run_fake_world(size, points, loads, **kw):
    hub = Hub(size)
    results = [None] * size

    def work(rank):
        comm = FakeComm(hub, rank)
        p, l = (points, loads) if rank == 0 else (None, None)  # only rank 0 holds the data
        results[rank] = parallel.sweep_loads(p, l, comm=comm, **kw)

    threads = [threading.Thread(target=work, args=(r,)) for r in range(size)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return results


def test_plan_covers_every_index_exactly_once():
    for n in (1, 5, 24, 25):
        for size in (1, 2, 3, 4, 7):
            seen = sorted(i for r in range(size) for i in parallel.plan(n, r, size))
            assert seen == list(range(n))


def test_merge_rejects_duplicates_and_gaps():
    row = {k: 0.0 for k in parallel.ROW_KEYS}
    try:
        parallel.merge([[(0, row)], [(0, row)]], 1)
        assert False, "duplicate index not detected"
    except ValueError:
        pass
    try:
        parallel.merge([[(0, row)]], 2)
        assert False, "missing index not detected"
    except ValueError:
        pass


def test_multi_rank_equals_serial_exactly():
    pts = nb.synthetic_cells(300, seed=3)
    loads = np.linspace(0.01, 0.1, 7).tolist()
    serial = parallel.sweep_loads(pts, loads)
    assert serial["mode"] == "serial" and serial["ranks"] == 1
    for size in (2, 3):
        out = run_fake_world(size, pts, loads)
        assert out[0]["mode"] == "mpi" and out[0]["ranks"] == size
        assert all(o is None for o in out[1:])
        assert out[0]["rows"] == serial["rows"]  # same floats, not just close
        assert len(out[0]["per_rank_busy_seconds"]) == size


def test_more_ranks_than_loads_still_works():
    pts = nb.synthetic_cells(200, seed=2)
    out = run_fake_world(4, pts, [0.02, 0.05])
    assert [r["load_fraction"] for r in out[0]["rows"]] == [0.02, 0.05]


def test_3d_and_stiffness_through_the_sweep():
    pts = nb.synthetic_cells(250, seed=4, dim=3)
    stiff = nb.stiffness_from_areas(np.random.default_rng(0).uniform(50, 200, 250))
    out = parallel.sweep_loads(pts, [0.02, 0.06], stiffness=stiff)
    assert out["dim"] == 3
    assert out["rows"][1]["mean_strain"] > out["rows"][0]["mean_strain"]


def test_find_threshold_interpolates_and_reports_not_reached():
    rows = [{"load_fraction": 0.02 * i, "p95_strain": 0.01 * i} for i in range(1, 6)]
    assert abs(parallel.find_threshold(rows, 0.035) - 0.07) < 1e-9
    assert parallel.find_threshold(rows, 0.5) is None
    assert parallel.find_threshold(rows, 0.001) == rows[0]["load_fraction"]
