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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ["numpy", "scipy", "pandas", "PIL", "plotly", "streamlit", "skimage", "cv2", "torch", "segment_anything", "mpi4py", "juliacall", "psutil", "mcp"]


def _version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception as exc:
        return f"missing ({type(exc).__name__})"


def _meta():
    from mequa.agents import meta_client

    base = meta_client.get_base_url()
    if not os.environ.get("META_API_KEY", "").strip():
        return {"url": base, "reachable": False, "key_set": False,
                "models": {"agent": meta_client.get_agent_model(),
                           "guard": meta_client.get_guard_model(),
                           "planner": meta_client.get_planner_model()}}
    try:
        out = meta_client.status(timeout=8.0)
        return {"url": base, "reachable": bool(out.get("up")), "key_set": True,
                "models": out.get("models", []),
                **({"error": out["error"]} if "error" in out else {})}
    except Exception as exc:
        return {"url": base, "reachable": False, "key_set": True,
                "error": f"{type(exc).__name__}: {exc}"}


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
        "tools": {t: shutil.which(t) for t in ["julia", "cargo", "mpiexec", "mpirun", "git"]},
        "sam_checkpoints": sorted(p.name for p in sam.glob("*.pth")) if sam.exists() else [],
        "meta_api": _meta(),
        "env": {k: os.environ.get(k) for k in ["LLM_PROVIDER", "META_BASE_URL", "META_AGENT_MODEL",
                                              "META_GUARD_MODEL", "META_PLANNER_MODEL", "SAM_CHECKPOINT"]},
    }
    out = ROOT / "results" / f"env_{platform.node() or 'machine'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
