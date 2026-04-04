"""Centralised configuration loader with attribute-style access."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class _AttrDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"No config key '{key}'") from None
        if isinstance(value, dict):
            return _AttrDict(value)
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _load_config(path: Path) -> _AttrDict:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _AttrDict(raw)


def _find_config_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = current / "configs" / "config.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError("configs/config.yaml not found in any parent directory")


_CONFIG_PATH = Path(os.environ.get("SELF_HEALING_CONFIG", str(_find_config_root())))
_PROJECT_ROOT = _CONFIG_PATH.parent.parent

settings = _load_config(_CONFIG_PATH)


def resolve(relative_path: str) -> Path:
    """Return an absolute path relative to the project root, creating parent dirs."""
    p = _PROJECT_ROOT / relative_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
