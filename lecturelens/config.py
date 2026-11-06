"""Configuration: `config.yaml` + optional overrides + keys from `.env`."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from .utils import PROJECT_ROOT

log = logging.getLogger(__name__)

KEY_NAMES = ("GEMINI_API_KEY", "GROQ_API_KEY")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Thin wrapper around the YAML dict with dotted-path access."""

    def __init__(self, data: dict, root: Path = PROJECT_ROOT):
        self.data = data
        self.root = root

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, key: str) -> Path:
        """Resolve a `paths.*` entry relative to the project root."""
        p = Path(self.get(f"paths.{key}"))
        return p if p.is_absolute() else self.root / p

    def with_overrides(self, overrides: dict) -> "Config":
        return Config(_deep_merge(self.data, overrides), self.root)

    def snapshot(self) -> dict:
        return copy.deepcopy(self.data)

    @staticmethod
    def api_key(name: str) -> str | None:
        v = os.environ.get(name, "").strip()
        return v or None

    @classmethod
    def keys_found(cls) -> dict[str, bool]:
        """Which keys are present (never the values)."""
        return {k: cls.api_key(k) is not None for k in KEY_NAMES}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover
        log.warning("python-dotenv not installed; reading keys from the environment only")
    p = Path(path or os.environ.get("LECTURELENS_CONFIG") or PROJECT_ROOT / "config.yaml")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(data)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    for k in ("runs_dir", "cache_dir", "results_dir"):
        cfg.path(k).mkdir(parents=True, exist_ok=True)
    return cfg
