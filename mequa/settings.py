"""User settings that outlive a session, and the one decision this module exists for: which machine
runs the guard models (Llama Guard and the intent classifier).

Guard mode
  auto       Use the dedicated guard machine while it answers; if it does not, run the guard models on
             the main machine instead and say so. Recommended: nothing stops working when the second
             laptop is asleep.
  dedicated  Guard machine only. If it does not answer, requests and replies are refused with a clear
             reason (fail closed). Use this when the guard must never share the main machine.
  main       The main machine runs everything. The guard URL is ignored. Use this with one laptop.

The firewall always runs both guard layers; this setting only chooses where. Settings are stored in
config/mequa_settings.json. MEQUA_GUARD_MODE in the environment overrides the file (CLI and CI).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "config" / "mequa_settings.json"
MODES = ("auto", "dedicated", "main")
MODE_LABELS = {"auto": "Auto: guard machine, else main",
               "dedicated": "Guard machine only",
               "main": "Main machine does everything"}
DEFAULT_MAIN = "http://127.0.0.1:11434"
_PROBE_TTL = 8.0
_probe_cache: Dict[str, Any] = {}
_lock = threading.Lock()


def _read() -> Dict[str, Any]:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load() -> Dict[str, Any]:
    d = {"guard_mode": "auto", "guard_url": "", "theme": "light"}
    d.update({k: v for k, v in _read().items() if k in d})
    if d["guard_mode"] not in MODES:
        d["guard_mode"] = "auto"
    return d


def save(**changes: Any) -> Dict[str, Any]:
    d = load()
    for k, v in changes.items():
        if k == "guard_mode" and v not in MODES:
            raise ValueError(f"guard_mode must be one of {MODES}")
        if k == "theme" and v not in ("light", "green"):
            raise ValueError("theme must be 'light' or 'green'")
        if k in d:
            d[k] = str(v).strip().rstrip("/") if k == "guard_url" else v
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    with _lock:
        _probe_cache.clear()
    return d


def guard_mode() -> str:
    env = os.environ.get("MEQUA_GUARD_MODE", "").strip().lower()
    return env if env in MODES else load()["guard_mode"]


def main_url() -> str:
    return (os.environ.get("OLLAMA_MAIN_URL") or DEFAULT_MAIN).rstrip("/")


def configured_guard_url() -> str:
    """The guard machine address: the Settings value, else OLLAMA_GUARD_URL, else empty."""
    return (load()["guard_url"] or os.environ.get("OLLAMA_GUARD_URL", "")).rstrip("/")


def reachable(url: str, timeout: float = 0.8, cache: bool = True) -> bool:
    if not url:
        return False
    now = time.monotonic()
    with _lock:
        hit = _probe_cache.get(url)
        if cache and hit and now - hit[0] < _PROBE_TTL:
            return hit[1]
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout):
            ok = True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        ok = False
    with _lock:
        _probe_cache[url] = (now, ok)
    return ok


def guard_status() -> Dict[str, Any]:
    """Where the guard models run right now and why."""
    mode, main, guard = guard_mode(), main_url(), configured_guard_url()
    separate = bool(guard) and guard != main
    if mode == "main" or not separate:
        return {"mode": mode, "effective": main, "configured": guard, "on_main": True, "fallback": False,
                "guard_up": None, "ok": True,
                "note": "Main machine runs the guard models." if mode == "main" else "No separate guard machine is set, so the main machine runs the guard models."}
    up = reachable(guard)
    if up:
        return {"mode": mode, "effective": guard, "configured": guard, "on_main": False, "fallback": False, "guard_up": True, "ok": True,
                "note": "Guard machine is answering."}
    if mode == "auto":
        return {"mode": mode, "effective": main, "configured": guard, "on_main": True, "fallback": True, "guard_up": False, "ok": True,
                "note": f"Guard machine {guard} is not answering, so the main machine is running the guard models."}
    return {"mode": mode, "effective": guard, "configured": guard, "on_main": False, "fallback": False, "guard_up": False, "ok": False,
            "note": f"Guard machine {guard} is not answering and the mode is 'Guard machine only', so requests are refused. "
                    "Start Ollama on that machine, or switch to 'Auto' or 'Main machine does everything' in Machines."}


def effective_guard_url() -> str:
    return guard_status()["effective"]
