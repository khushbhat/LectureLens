"""Small shared helpers: hashing, timestamps, token estimates, JSON IO."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, n: int = 8) -> str:
    return sha256_text(text)[:n]


def file_hash(path: str | Path, n: int = 8, max_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash of file size + first `max_bytes` (fast for big videos)."""
    p = Path(path)
    h = hashlib.sha256(str(p.stat().st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()[:n]


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (s[:max_len].strip("-")) or "video"


def fmt_ts(seconds: float | None) -> str:
    """12.5 -> '0:12', 3725 -> '1:02:05'."""
    if seconds is None:
        return "?"
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def srt_ts(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def estimate_tokens(text: str) -> int:
    """Cheap provider-agnostic token estimate (~4 chars/token for English)."""
    return math.ceil(len(text) / 4) if text else 0


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    """Atomic JSON write (a crash never leaves a half-written cache file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating ```json fences and prose around it."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object found in model output")
    start = min(starts)
    end = max(t.rfind("}"), t.rfind("]"))
    return json.loads(t[start : end + 1])


def repair_truncated_json(text: str, max_tries: int = 60) -> Any:
    """Locally fix JSON cut off mid-way (e.g. max_tokens reached): cut back to a
    complete value and close all open brackets. Raises ValueError if impossible."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*)", t, re.S)
    if fence:
        t = fence.group(1).replace("```", "")
    start = min([i for i in (t.find("{"), t.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError("no JSON start")
    t = t[start:]
    cuts = [i for i, ch in enumerate(t) if ch in "}]"][::-1][:max_tries]
    for cut in cuts:
        cand = t[: cut + 1]
        stack, in_str, esc = [], False, False
        for ch in cand:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack:
                    stack.pop()
        if in_str:
            continue
        try:
            return json.loads(cand + "".join(reversed(stack)))
        except json.JSONDecodeError:
            continue
    raise ValueError("could not repair truncated JSON")
