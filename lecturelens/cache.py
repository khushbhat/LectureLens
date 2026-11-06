"""Persistent LLM response cache.

Key = sha256(provider, model, system, prompt, params). Re-running the app,
resuming a run, or re-running the evaluation never spends quota twice on
the same request.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .utils import sha256_text


def cache_key(provider: str, model: str, prompt: str, system: str | None, params: dict) -> str:
    payload = json.dumps(
        {"provider": provider, "model": model, "system": system or "", "prompt": prompt, "params": params},
        sort_keys=True, ensure_ascii=False,
    )
    return sha256_text(payload)


class LLMCache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, provider TEXT, model TEXT, "
                "response TEXT, meta TEXT, created REAL)"
            )
        self.hits = 0
        self.misses = 0

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)

    def get(self, key: str) -> tuple[str, dict] | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT response, meta FROM responses WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return row[0], json.loads(row[1] or "{}")

    def put(self, key: str, provider: str, model: str, response: str, meta: dict) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?)",
                (key, provider, model, response, json.dumps(meta), time.time()),
            )
