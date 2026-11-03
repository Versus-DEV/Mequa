import os
import shutil
import subprocess
import sys

import pytest

mpi = shutil.which("mpiexec") or shutil.which("mpirun")
pytest.importorskip("mpi4py")


@pytest.mark.skipif(mpi is None, reason="no MPI launcher on PATH")
@pytest.mark.parametrize("ranks,dim", [(2, 2), (3, 3), (4, 3)])
def test_domain_decomposed_relaxation_matches_serial(ranks, dim, tmp_path):
    out = tmp_path / "r.json"
    cmd = [mpi, "-n", str(ranks), sys.executable, "-m", "mequa.domain_mpi", "--cells", "1500", "--dim", str(dim),
           "--iters", "120", "--verify", "--out", str(out)]
    if os.name != "nt":  # Open MPI: CI runners have 2 to 4 cores, so always allow more ranks than slots
        cmd[1:1] = ["--oversubscribe"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        cmd[1:1] = ["--allow-run-as-root"]
    done = subprocess.run(cmd, capture_output=True, timeout=300, text=True,
                          env={**os.environ, "OMPI_MCA_rmaps_base_oversubscribe": "1"})
    assert done.returncode == 0, f"mpiexec failed ({done.returncode}):\n{done.stderr[-2000:]}"
    import json

    r = json.loads(out.read_text())
    assert r["ranks"] == ranks
    assert r["max_abs_position_diff_vs_serial"] < 1e-9
    assert r["max_abs_strain_diff_vs_serial"] < 1e-9
