"""Common LLM provider interface and error taxonomy."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ProviderUnavailable(Exception):
    """Provider cannot be used at all (no key, package missing, server down, bad model)."""


class RateLimitError(Exception):
    """429 / RESOURCE_EXHAUSTED. `daily` => quota gone for today; `retry_after` hint in s."""

    def __init__(self, message: str, retry_after: float | None = None, daily: bool = False):
        super().__init__(message)
        self.retry_after, self.daily = retry_after, daily


class TransientError(Exception):
    """5xx, timeouts, connection resets — worth retrying with backoff."""


class FatalRequestError(Exception):
    """4xx other than rate limits (bad request, too large, safety block) — don't retry."""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    meta: dict = field(default_factory=dict)


class LLMProvider(ABC):
    """`generate(prompt, system=None, json_schema=None, temperature=0.2) -> str`.

    `complete()` returns an `LLMResult` with token usage; `generate()` is the
    plain-string convenience wrapper required by the brief.
    """

    name: str = "base"

    def __init__(self, model: str, max_output_tokens: int = 4096, timeout_s: float = 180):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, json_schema: dict | None = None,
                 temperature: float = 0.2, max_tokens: int | None = None) -> LLMResult: ...

    def generate(self, prompt: str, system: str | None = None, json_schema: dict | None = None,
                 temperature: float = 0.2) -> str:
        return self.complete(prompt, system, json_schema, temperature).text

    def available(self) -> tuple[bool, str]:
        """(ok, message). Cheap check; may call a free list-models endpoint once."""
        return True, "ok"

    def list_models(self) -> list[str]:
        return []


_RETRY_PATTERNS = [
    r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",
    r"retry in (\d+(?:\.\d+)?)\s*s",
    r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s",
]


def parse_retry_after(message: str) -> float | None:
    """Extract a retry hint like 'retryDelay': '34s' or 'try again in 1m2.5s'."""
    m = re.search(_RETRY_PATTERNS[2], message, re.I)
    if m:
        return float(m.group(1) or 0) * 60 + float(m.group(2))
    for pat in _RETRY_PATTERNS[:2]:
        m = re.search(pat, message, re.I)
        if m:
            return float(m.group(1))
    return None


def looks_daily(message: str) -> bool:
    """Heuristic: is this 429 about a per-day quota (don't wait, fall back)?"""
    m = message.lower()
    return any(s in m for s in ("perday", "per day", "per_day", "requests per day", "tokens per day", "(rpd)", "(tpd)", "daily"))
