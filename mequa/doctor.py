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
OLLAMA_MODELS = ["llama3.2:3b", "llama-guard3:1b"]


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

    print("\nOllama")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            tags = [m["name"] for m in json.loads(resp.read().decode("utf-8")).get("models", [])]
        print(f"  reachable, models: {', '.join(tags) or 'none'}")
        for need in OLLAMA_MODELS:
            have = any(t == need or t.startswith(need.split(":")[0] + ":") and need.split(":")[1] in t for t in tags)
            print(f"  {'OK     ' if have else 'MISSING'} {need}")
    except Exception as exc:
        print(f"  NOT REACHABLE ({type(exc).__name__}); start Ollama")


if __name__ == "__main__":
    main()
