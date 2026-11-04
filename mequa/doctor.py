import importlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXPECTED = [
    "app.py",
    "mequa/__init__.py",
    "mequa/julia_bridge.py",
    "mequa/numpy_backend.py",
    "mequa/backend.py",
    "mequa/worker.py",
    "mequa/firewall.py",
    "mequa/agent.py",
    "mequa/ingest.py",
    "mequa/benchmark.py",
    "mequa/eval_all.py",
    "mequa/eval_firewall.py",
    "mequa/eval_heldout.py",
    "mequa/simulink.py",
    "mequa/slx_demo.py",
    "mequa/slx_cli.py",
    "mequa/couple.py",
    "mequa/doctor.py",
    "mequa/rust_backend.py",
    "mequa/mcp_server.py",
    "tests/test_numpy_backend.py",
    "tests/test_rust.py",
    ".github/workflows/ci.yml",
    "mequa/zero_copy_check.py",
    "mequa/crosscheck.py",
    "rust/Cargo.toml",
    "rust/src/lib.rs",
    "rust/build.ps1",
    "solvers/MequaCore.jl",
    "solvers/Bench.jl",
    "solvers/test_kernels.jl",
    "solvers/smoke_test.jl",
]
MODULES = ["mequa_rs", "mcp", "torchvision", "numpy", "scipy", "pandas", "PIL", "plotly", "streamlit", "juliacall", "torch", "segment_anything"]


def main():
    print(f"repo root: {ROOT}")
    print("\nFiles")
    missing = 0
    for rel in EXPECTED:
        ok = (ROOT / rel).exists()
        missing += 0 if ok else 1
        print(f"  {'OK     ' if ok else 'MISSING'} {rel}")
    known = {Path(p).name for p in EXPECTED}
    for folder in ("mequa", "solvers"):
        d = ROOT / folder
        if d.exists():
            extra = sorted(p.name for p in d.iterdir() if p.is_file() and p.name not in known)
            if extra:
                print(f"  unrecognized files in {folder}/ (misnamed or leftover): {', '.join(extra)}")
    print(f"\n{missing} expected file(s) missing")

    print("\nPython packages")
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"  OK      {mod}")
        except Exception as exc:
            print(f"  MISSING {mod} ({type(exc).__name__})")

    print("\nJulia backend")
    try:
        importlib.import_module("mequa.julia_bridge")
        print("  OK      mequa.julia_bridge imports and Julia starts")
    except Exception as exc:
        print(f"  FAILED  {type(exc).__name__}: {exc}")

    print("\nSAM checkpoints in models/")
    found = sorted((ROOT / "models").glob("*.pth")) if (ROOT / "models").exists() else []
    if not found:
        print("  none found (the first --method sam run will download ViT-B)")
    for f in found:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")

    print("\nRust toolchain (only needed to rebuild the module)")
    import shutil

    for tool in ("cargo", "rustc", "maturin"):
        print(f"  {'OK     ' if shutil.which(tool) else 'MISSING'} {tool}")

    print("\nMeta Model API")
    import os

    from mequa.agents import meta_client
    from mequa.utils.env import load_env_file

    load_env_file()

    key = os.getenv("META_API_KEY", "").strip()
    print(f"  {'OK     ' if key else 'MISSING'} META_API_KEY")
    print(f"  base: {meta_client.get_base_url()}")
    print(f"  agent model: {meta_client.get_agent_model()} · guard model: {meta_client.get_guard_model()} "
          f"· planner model: {meta_client.get_planner_model()}")
    try:
        out = meta_client.status(timeout=10.0)
        if out.get("up"):
            print(f"  reachable, models: {', '.join(out.get('models', [])) or '(list unavailable)'}")
        else:
            print(f"  NOT REACHABLE ({out.get('error', 'unknown')}); check META_API_KEY and META_BASE_URL")
    except Exception as exc:
        print(f"  NOT REACHABLE ({type(exc).__name__}); check META_API_KEY and META_BASE_URL")


if __name__ == "__main__":
    main()
