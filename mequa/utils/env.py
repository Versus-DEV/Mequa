"""Small environment-file loader.

This avoids requiring python-dotenv for hackathon scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union


PathLike = Union[str, Path]


def load_env_file(path: Optional[PathLike] = None) -> Dict[str, str]:
    """Load key=value pairs from a .env file.

    Existing environment variables are not overwritten.
    """
    if path is None:
        path = Path.cwd() / ".env"

    path = Path(path)
    values: Dict[str, str] = {}

    if not path.exists():
        return values

    text = path.read_text(encoding="utf-8")

    for line in text.splitlines():
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

        values[key] = value
        os.environ.setdefault(key, value)

    return values