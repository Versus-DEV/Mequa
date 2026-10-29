import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mequa import numpy_backend as nb

ROOT = Path(__file__).resolve().parent.parent
julia = shutil.which("julia")
pytestmark = pytest.mark.skipif(julia is None, reason="julia not installed")


def test_smoke_script_runs():
    out = subprocess.run([julia, "--threads", "2", str(ROOT / "solvers" / "smoke_test.jl")], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    assert "pipeline SPHI" in out.stdout


def test_kernel_self_checks():
    out = subprocess.run([julia, "--threads", "2", str(ROOT / "solvers" / "test_kernels.jl")], capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "grid == bruteforce: true" in out.stdout


def test_julia_agrees_with_numpy(tmp_path):
    pts = nb.synthetic_cells(1200)
    csv = tmp_path / "pts.csv"
    np.savetxt(csv, pts, delimiter=",", fmt="%.17g")
    script = tmp_path / "ref.jl"
    script.write_text(
        f'include("{(ROOT / "solvers" / "MequaCore.jl").as_posix()}")\n'
        "using .MequaCore\nusing DelimitedFiles\n"
        f'pts = Matrix{{Float64}}(readdlm("{csv.as_posix()}", \',\'))\n'
        "ei, ej = build_knn_edges(pts, 6)\n"
        "pos, ns, es = relax_strain(pts, ei, ej; load=0.05)\n"
        'println("SPHI ", sphi(ns, ei, ej))\n'
    )
    out = subprocess.run([julia, str(script)], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    value = float(out.stdout.strip().split()[-1])
    ref = nb.analyze(pts, 3, 6, 0.05)["sphi"]
    assert abs(value - ref) / ref < 1e-8
