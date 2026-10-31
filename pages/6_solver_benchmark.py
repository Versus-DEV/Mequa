"""Measured results: the cross-backend benchmark and the MPI scaling run (read from results/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mequa import ui

ui.page_header('Benchmarks and MPI Scaling', 'Nothing here is computed in the page. It shows result files written by the measurement scripts, with the machine that produced them.')

bench = ROOT / "results" / "benchmark.json"
st.subheader("Julia, Rust and NumPy on identical points")
if bench.exists():
    data = json.loads(bench.read_text(encoding="utf-8"))
    st.json(data, expanded=False)
else:
    st.info("Run: python -m mequa.benchmark")

mpi = ROOT / "results" / "mpi_scaling.json"
st.subheader("MPI load sweep scaling")
if mpi.exists():
    m = json.loads(mpi.read_text(encoding="utf-8"))
    st.caption(f"{m['cells']} cells, {m['dim']}D, {m['loads']} loads, median of {m.get('repeats', m.get('repeats_best_of'))} interleaved runs. Machine: {m['machine'].get('node')} ({m['machine'].get('processor')})")
    st.dataframe(pd.DataFrame(m["scaling"]), use_container_width=True)
    st.caption("max_abs_diff_vs_1_rank is the largest difference in any reported number between that rank count and the single-rank run.")
else:
    st.info("Run: python -m mequa.mpi_scaling --cells 3000 --loads 24")
