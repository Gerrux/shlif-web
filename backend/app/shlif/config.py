"""Configuration loading — a thin, dotted-access wrapper over the YAML config."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Config(dict):
    """dict with attribute access; nested dicts are wrapped recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path | None = None) -> Config:
    """Load the pipeline config. Falls back to config/default.yaml."""
    path = Path(path) if path else _DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(copy.deepcopy(data))
