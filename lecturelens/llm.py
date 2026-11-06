"""Quota-aware LLM router: cache -> throttle -> call -> retry/backoff -> fallback.

Order of operations for one request:
1. Look the request up in the response cache for *every* provider in the
   chain (so a resumed run reuses an answer even if it came from a fallback).
2. For each available provider in order: reserve quota (client-side throttle,
   daily caps), call, retry transient errors / short rate limits with
   exponential backoff + jitter (honouring retry-after hints), and on daily
   exhaustion or repeated failure move on to the next provider.
3. If every provider is exhausted, raise `AllProvidersExhausted`; the
   pipeline then saves state so `lecturelens resume <run_id>` can continue.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from .cache import LLMCache, cache_key
from .providers import build_provider
from .providers.base import (FatalRequestError, LLMProvider, LLMResult, ProviderUnavailable, RateLimitError,
                             TransientError)
from .quota import QuotaExhausted, QuotaManager
from .utils import estimate_tokens

log = logging.getLogger(__name__)

_AVAILABILITY: dict[tuple, tuple[bool, str]] = {}  # per-process cache of model checks


class AllProvidersExhausted(Exception):
    """No provider could serve the request. `quota_related` => resume later can help."""

    def __init__(self, reasons: dict[str, str], quota_related: bool, too_large: bool = False):
        self.reasons, self.quota_related, self.too_large = reasons, quota_related, too_large
        super().__init__("; ".join(f"{k}: {v}" for k, v in reasons.items()) or "no LLM provider available")


@dataclass
class RouterStats:
    calls: int = 0              # real API calls that succeeded
    attempts: int = 0           # including failed attempts
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    fallbacks: int = 0
    wait_s: float = 0.0
    by_provider: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class LLMRouter:
    def __init__(self, cfg, quota: QuotaManager | None = None, cache: LLMCache | None = None,
                 order: list[str] | None = None, providers: dict[str, LLMProvider] | None = None,
                 on_wait: Callable[[float, str], None] | None = None, run_id: str | None = None,
                 sleeper: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.quota = quota or QuotaManager(cfg, on_wait=on_wait)
        self.cache = cache or LLMCache(cfg.path("cache_dir") / "llm_cache.sqlite")
        self.order = list(order or cfg.get("llm.order", ["gemini", "groq", "ollama"]))
        self._providers: dict[str, LLMProvider] = dict(providers or {})
        self.on_wait = on_wait
        self.run_id = run_id
        self.sleeper = sleeper
        self.stats = RouterStats()
        self.max_retries = int(cfg.get("quota.max_retries", 4))
        self.backoff_base = float(cfg.get("quota.backoff_base_s", 2.0))
        self.backoff_max = float(cfg.get("quota.backoff_max_s", 90.0))
        self.max_wait = float(cfg.get("quota.max_wait_s", 120))

    # -------------------------------------------------------------- providers
    def provider(self, name: str) -> LLMProvider:
        if name not in self._providers:
            self._providers[name] = build_provider(name, self.cfg)
        return self._providers[name]

    def check(self, name: str) -> tuple[bool, str]:
        """Is provider usable (key present, model exists)? Cached per process."""
        try:
            p = self.provider(name)
        except Exception as e:
            return False, str(e)
        key = (name, p.model, getattr(p, "api_key", None) is not None)
        if key not in _AVAILABILITY or name == "mock":
            if not self.cfg.get("llm.check_models_on_startup", True) and name in ("gemini", "groq"):
                ok = getattr(p, "api_key", None) is not None
                _AVAILABILITY[key] = (ok, "ok" if ok else "no API key")
            else:
                _AVAILABILITY[key] = p.available()
        return _AVAILABILITY[key]

    def available_chain(self) -> list[str]:
        chain = []
        for name in self.order:
            ok, msg = self.check(name)
            if ok:
                chain.append(name)
            else:
                log.info("LLM provider %s unavailable: %s", name, msg)
        return chain

    def status(self) -> list[dict]:
        rows = []
        for name in self.order:
            ok, msg = self.check(name)
            p = self.provider(name)
            rows.append({"provider": name, "model": p.model, "available": ok, "message": msg})
        return rows

    def budget(self, name: str, key: str, default: int) -> int:
        return int(self.cfg.get(f"llm.{name}.{key}", default) or default)

    # ---------------------------------------------------------------- calling
    def _params(self, p: LLMProvider, json_mode: bool, temperature: float, max_tokens: int | None) -> dict:
        return {"json": bool(json_mode), "temperature": temperature, "max_tokens": max_tokens or p.max_output_tokens}

    def lookup_cache(self, prompt: str, system: str | None, json_mode: bool, temperature: float,
                     max_tokens: int | None, chain: list[str]) -> LLMResult | None:
        for name in chain:
            p = self.provider(name)
            mt = max_tokens if max_tokens else p.max_output_tokens
            hit = self.cache.get(cache_key(p.name, p.model, prompt, system, self._params(p, json_mode, temperature, mt)))
            if hit:
                text, meta = hit
                self.stats.cache_hits += 1
                return LLMResult(text, p.name, p.model, meta.get("input_tokens", 0), meta.get("output_tokens", 0), cached=True)
        return None

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = True,
                 temperature: float | None = None, max_tokens: int | None = None, task: str = "",
                 chain: list[str] | None = None) -> LLMResult:
        temperature = self.cfg.get("llm.temperature", 0.2) if temperature is None else temperature
        chain = [c for c in (chain or self.available_chain()) if self.check(c)[0]]
        if not chain:
            raise AllProvidersExhausted({"llm": "no LLM provider available (no keys / Ollama not running)"}, False)
        cached = self.lookup_cache(prompt, system, json_mode, temperature, max_tokens, chain)
        if cached:
            log.info("[%s] cache hit (%s/%s) — no quota used", task, cached.provider, cached.model)
            self.stats.events.append({"task": task, "provider": cached.provider, "model": cached.model, "cached": True})
            return cached

        reasons, quota_related, too_large = {}, False, False
        for i, name in enumerate(chain):
            p = self.provider(name)
            if i > 0:
                self.stats.fallbacks += 1
                prev = chain[i - 1]
                log.warning("[%s] %s failed (%s) -> falling back to %s/%s", task, prev, reasons.get(prev, "?")[:200], name, p.model)
            try:
                return self._call_with_retries(p, prompt, system, json_mode, temperature, max_tokens, task)
            except QuotaExhausted as e:
                reasons[name] = e.reason
                quota_related = True
                too_large = too_large or "exceeds" in e.reason
            except RateLimitError as e:
                reasons[name] = f"rate limited: {str(e)[:150]}"
                quota_related = True
            except TransientError as e:
                reasons[name] = f"server error after retries: {str(e)[:150]}"
                quota_related = True  # likely temporary: resume later
            except ProviderUnavailable as e:
                reasons[name] = str(e)
                key = (name, p.model, getattr(p, "api_key", None) is not None)
                _AVAILABILITY[key] = (False, str(e))   # don't retry an unusable model on every call
            except FatalRequestError as e:
                reasons[name] = f"request rejected: {str(e)[:200]}"
                too_large = too_large or "too large" in str(e).lower() or "413" in str(e)
        raise AllProvidersExhausted(reasons, quota_related, too_large)

    def _backoff(self, attempt: int, hint: float | None) -> float:
        if hint is not None:
            return hint + random.uniform(0.5, 2.0)
        return min(self.backoff_max, self.backoff_base * (2 ** attempt)) * random.uniform(0.75, 1.25)

    def _wait(self, seconds: float, reason: str) -> None:
        self.stats.wait_s += seconds
        if self.on_wait:
            self.on_wait(seconds, reason)
        self.sleeper(seconds)

    def _call_with_retries(self, p: LLMProvider, prompt, system, json_mode, temperature, max_tokens, task) -> LLMResult:
        mt = max_tokens or p.max_output_tokens
        est_in = estimate_tokens(prompt) + estimate_tokens(system or "")
        # TPM accounting: input + requested output (Groq counts max_tokens up-front)
        est_total = est_in + mt
        attempt = 0
        while True:
            event_id = self.quota.acquire(p.name, p.model, tokens=est_total, run_id=self.run_id)
            self.stats.attempts += 1
            t0 = time.time()
            try:
                res = p.complete(prompt, system, {} if json_mode else None, temperature, mt)
            except RateLimitError as e:
                self.quota.commit(event_id, est_in)
                if e.daily:
                    self.quota.mark_exhausted(p.name, p.model, str(e))
                    raise QuotaExhausted(p.name, p.model, "daily quota exhausted (server 429)") from e
                if attempt >= self.max_retries or (e.retry_after or 0) > self.max_wait:
                    raise
                delay = self._backoff(attempt, e.retry_after)
                log.warning("[%s] %s/%s rate limited; retry %d in %.0fs", task, p.name, p.model, attempt + 1, delay)
                self.stats.retries += 1
                self._wait(delay, f"{p.name} rate limit (retry {attempt + 1}/{self.max_retries})")
                attempt += 1
                continue
            except TransientError:
                self.quota.commit(event_id, 0)
                if attempt >= self.max_retries:
                    raise
                delay = self._backoff(attempt, None)
                log.warning("[%s] %s transient error; retry %d in %.0fs", task, p.name, attempt + 1, delay)
                self.stats.retries += 1
                self._wait(delay, f"{p.name} server error (retry {attempt + 1}/{self.max_retries})")
                attempt += 1
                continue
            except (FatalRequestError, ProviderUnavailable):
                self.quota.commit(event_id, 0)
                raise
            used = (res.input_tokens or est_in) + (res.output_tokens or estimate_tokens(res.text))
            self.quota.commit(event_id, used)
            res.meta["latency_s"] = round(time.time() - t0, 2)
            self.cache.put(cache_key(p.name, p.model, prompt, system, self._params(p, json_mode, temperature, mt)),
                           p.name, p.model, res.text,
                           {"input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "task": task})
            self.stats.calls += 1
            self.stats.input_tokens += res.input_tokens or est_in
            self.stats.output_tokens += res.output_tokens or estimate_tokens(res.text)
            bp = self.stats.by_provider.setdefault(f"{p.name}/{p.model}", {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            bp["calls"] += 1
            bp["input_tokens"] += res.input_tokens
            bp["output_tokens"] += res.output_tokens
            self.stats.events.append({"task": task, "provider": p.name, "model": p.model, "cached": False,
                                      "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                                      "latency_s": res.meta["latency_s"], "retries": attempt})
            log.info("[%s] %s/%s ok: %d in / %d out tokens, %.1fs", task, p.name, p.model,
                     res.input_tokens, res.output_tokens, res.meta["latency_s"])
            return res
