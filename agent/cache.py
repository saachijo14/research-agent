import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from agent.tools.base import ToolResult

CACHE_DIR = Path("cache")
VALID_MODES = {"off", "auto", "record", "replay"}
_mode = os.getenv("CACHE_MODE", "auto")


def set_mode(mode: str) -> None:
    global _mode
    if mode not in VALID_MODES:
        raise ValueError(f"cache mode must be one of {VALID_MODES}")
    _mode = mode


def get_mode() -> str:
    return _mode


def make_key(tool: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps(
        {"tool": tool, "args": list(args), "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def load(key: str) -> Optional[ToolResult]:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return ToolResult.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None  # corrupted cache entry: behave as a miss


def save(key: str, result: ToolResult) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(result.model_dump_json(), encoding="utf-8")