"""Check whether both Ollama nodes are reachable.

Usage:
    python scripts/check_two_machine_ollama.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mequa.guard.remote_guard import list_models


def load_env_file(path: Path = Path(".env")) -> None:
    """Very small .env loader so we do not require python-dotenv yet."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def check_node(name: str, url: str) -> bool:
    print(f"[{name}] {url}")

    try:
        models = list_models(url, timeout=10.0)
        print("  PASS: Ollama server reachable.")
        print("  Models:")

        if models:
            for model in models:
                print(f"    - {model}")
        else:
            print("    - no models found")

        return True

    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False


def main() -> int:
    load_env_file()

    main_url = os.getenv("OLLAMA_MAIN_URL", "http://127.0.0.1:11434")
    guard_url = os.getenv("OLLAMA_GUARD_URL", main_url)

    print("Mequa two-machine Ollama check")
    print("=" * 50)

    results = []

    results.append(check_node("MAIN", main_url))
    results.append(check_node("GUARD", guard_url))

    print("=" * 50)

    if all(results):
        print("PASS: both nodes reachable.")
        return 0

    print("FAIL: one or more nodes unreachable.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())