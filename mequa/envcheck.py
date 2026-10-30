"""Environment report for one machine. Run it on BOTH laptops and send back results/env_<name>.json.

    python -m mequa.envcheck
It never changes anything; it only reads versions, hardware, and which services answer.
"""

import importlib
import json
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["numpy", "scipy", "pandas", "PIL", "plotly", "streamlit", "skimage", "cv2", "torch", "segment_anything", "mpi4py", "juliacall", "psutil", "mcp"]


def _version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:
        return f"missing ({type(exc).__name__})"


def _ollama(url):
    if not url:
        return {"url": None, "reachable": False}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=4) as r:
            models = [m.get("name") for m in json.loads(r.read().decode()).get("models", [])]
        return {"url": url, "reachable": True, "models": models}
    except Exception as exc:
        return {"url": url, "reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    try:
        from mequa.utils.env import load_env_file

        load_env_file(ROOT / ".env")
    except Exception:
        pass
    ram = None
    try:
        import psutil

        ram = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    sam = ROOT / "models" / "sam"
    report = {
        "node": platform.node(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": ram,
        "python": sys.version.split()[0],
        "packages": {p: _version(p) for p in PACKAGES},
        "tools": {t: shutil.which(t) for t in ["julia", "cargo", "mpiexec", "mpirun", "ollama", "git"]},
        "sam_checkpoints": sorted(p.name for p in sam.glob("*.pth")) if sam.exists() else [],
        "ollama_main": _ollama(os.environ.get("OLLAMA_MAIN_URL", "http://127.0.0.1:11434")),
        "ollama_guard": _ollama(os.environ.get("OLLAMA_GUARD_URL")),
        "env": {k: os.environ.get(k) for k in ["PLANNER_MODEL", "GUARD_MODEL", "INTENT_MODEL", "SAM_CHECKPOINT"]},
    }
    out = ROOT / "results" / f"env_{platform.node() or 'machine'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
