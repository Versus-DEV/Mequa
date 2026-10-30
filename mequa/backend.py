import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

PREFIX = "@@MEQUA@@ "
ROOT = Path(__file__).resolve().parent.parent


class JuliaWorker:
    name = "julia"

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mequa.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        msg = self._read()
        if not msg.get("ready"):
            raise RuntimeError("worker did not report ready")

    def _read(self):
        import json

        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("Julia worker exited")
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX):])

    def _call(self, req):
        import json

        with self.lock:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            msg = self._read()
        if not msg.get("ok"):
            raise RuntimeError(msg.get("error", "worker error"))

    def analyze(self, points, k=3, knn=6, load=0.05):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "points.npy")
            dst = os.path.join(tmp, "result.npz")
            np.save(src, np.ascontiguousarray(points, dtype=np.float64))
            self._call({"op": "analyze", "points": src, "out": dst, "k": int(k), "knn": int(knn), "load": float(load)})
            with np.load(dst) as data:
                return {key: (data[key].item() if data[key].ndim == 0 else data[key].copy()) for key in data.files}

    def synthetic_cells(self, n=2000, seed=1, clusters=3):
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, "points.npy")
            self._call({"op": "synthetic", "n": int(n), "seed": int(seed), "clusters": int(clusters), "out": dst})
            return np.load(dst)

    def close(self):
        try:
            self._call({"op": "quit"})
        except Exception:
            pass


class _Wrapper:
    def __init__(self, module, name):
        self.analyze = module.analyze
        self.synthetic_cells = module.synthetic_cells
        self.name = name


def _warn(reason):
    print(f"[mequa] WARNING: Requested backend unavailable ({reason}). Using the slower NumPy fallback.", file=sys.stderr)


def load(mode="julia"):
    require = bool(os.environ.get("MEQUA_REQUIRE_JULIA"))
    if os.environ.get("MEQUA_FORCE_NUMPY"):
        mode = "numpy"
    if mode == "julia_worker":
        try:
            return JuliaWorker()
        except Exception as exc:
            if require:
                raise
            _warn(f"worker failed: {exc}")
            mode = "numpy"
    if mode == "rust":
        try:
            from mequa import rust_backend

            return _Wrapper(rust_backend, "rust")
        except Exception as exc:
            if require:
                raise
            _warn(f"rust module unavailable: {type(exc).__name__}: {exc}")
            mode = "numpy"
    if mode == "julia":
        try:
            from mequa import julia_bridge

            return _Wrapper(julia_bridge, "julia")
        except Exception as exc:
            if require:
                raise
            _warn(f"{type(exc).__name__}: {exc}")
            mode = "numpy"
    from mequa import numpy_backend

    return _Wrapper(numpy_backend, "numpy")


def analyze_routed(be, points, k=3, knn=6, load=0.05, stiffness=None):
    """Run an analysis on the requested backend when it supports the job, otherwise on NumPy.

    The Julia and Rust kernels currently handle 2D points with uniform stiffness. Jobs with
    per-cell stiffness (SAM-driven mechanics) or 3D points run on the NumPy backend. The result
    always names the backend that actually ran, so nothing is attributed to the wrong solver.
    """
    from mequa import numpy_backend

    points = np.asarray(points)
    needs_numpy = stiffness is not None or points.shape[1] != 2
    if needs_numpy and be.name != "numpy":
        res = numpy_backend.analyze(points, k=k, knn=knn, load=load, stiffness=stiffness)
        res["backend_used"] = "numpy"
        res["routed_from"] = be.name
        return res
    res = be.analyze(points, k=k, knn=knn, load=load) if stiffness is None else be.analyze(points, k=k, knn=knn, load=load, stiffness=stiffness)
    res["backend_used"] = be.name
    return res


def synthetic_any(be, n=2000, seed=1, clusters=3, dim=2):
    if dim == 2:
        return be.synthetic_cells(n, seed=seed, clusters=clusters)
    from mequa import numpy_backend

    return numpy_backend.synthetic_cells(n, seed=seed, clusters=clusters, dim=dim)
