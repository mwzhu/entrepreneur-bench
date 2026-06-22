from __future__ import annotations

import os
from pathlib import Path


DISABLED_VALUES = {"0", "false", "no", "off"}


def load_dotenv(path: Path | None = None, *, override: bool = False) -> int:
    """Load simple KEY=value pairs from a local .env file.

    This intentionally supports only the small subset Solvent needs for local
    credentials, avoiding a runtime dependency while keeping shell exports in
    charge by default.
    """
    if os.environ.get("SOLVENT_DOTENV", "").strip().lower() in DISABLED_VALUES:
        return 0
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return 0

    loaded = 0
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or (key in os.environ and not override):
            continue
        os.environ[key] = _strip_quotes(value.strip())
        loaded += 1
    return loaded


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
