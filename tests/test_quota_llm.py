import datetime as dt

import pytest

from lecturelens.cache import LLMCache
from lecturelens.llm import AllProvidersExhausted, LLMRouter
from lecturelens.providers.base import RateLimitError, TransientError, parse_retry_after
from lecturelens.providers.mock import MockProvider
from lecturelens.quota import QuotaExhausted, QuotaManager


def _quota(cfg, tmp_path, clock, limits):
    c = cfg.with_overrides({"quota": {"limits": limits}})
    return QuotaManager(c, tmp_path / "q.db", clock=clock, sleeper=clock.sleep)


def test_rpm_throttle_waits_instead_of_429(cfg, tmp_path, clock):
    q = _quota(cfg, tmp_path, clock, {"mock/m": {"rpm": 3}})
    for _ in range(3):
        q.acquire("mock", "m")
    assert clock.slept == []
    q.acquire("mock", "m")
    assert len(clock.slept) == 1 and 59 < clock.slept[0] < 61


def test_tpm_throttle_and_oversized_request(cfg, tmp_path, clock):
    q = _quota(cfg, tmp_path, clock, {"groq/x": {"tpm": 12000}})
    q.acquire("groq", "x", tokens=9000)
    q.acquire("groq", "x", tokens=5000)     # must wait for the window
    assert clock.slept and clock.slept[0] > 59
    with pytest.raises(QuotaExhausted, match="exceeds tokens/min"):
        q.acquire("groq", "x", tokens=13000)


def test_daily_cap_and_pacific_reset(cfg, tmp_path, clock):
    # 2026-03-10 23:30 Pacific (PDT) == 06:30 UTC next day
    clock.t = dt.datetime(2026, 3, 11, 6, 30, tzinfo=dt.timezone.utc).timestamp()
    q = _quota(cfg, tmp_path, clock, {"gemini/g": {"rpd": 2, "day_reset_tz": "America/Los_Angeles"}})
    assert q.day_key("gemini", "g") == "2026-03-10"
    q.acquire("gemini", "g")
    q.acquire("gemini", "g")
    with pytest.raises(QuotaExhausted, match="daily request cap"):
        q.acquire("gemini", "g")
    assert q.remaining("gemini", "g")["requests_day"] == 0
    clock.t += 31 * 60   # 00:01 Pacific -> new day
    assert q.day_key("gemini", "g") == "2026-03-11"
    q.acquire("gemini", "g")
    assert q.remaining("gemini", "g")["requests_day"] == 1


def test_mark_exhausted_blocks_until_reset(cfg, tmp_path, clock):
    q = _quota(cfg, tmp_path, clock, {"mock/m": {}})
    q.mark_exhausted("mock", "m", "429 PerDay")
    with pytest.raises(QuotaExhausted):
        q.acquire("mock", "m")
    assert not q.can_afford("mock", "m", 1)


def test_ledger_persists(cfg, tmp_path, clock):
    q = _quota(cfg, tmp_path, clock, {"mock/m": {"rpd": 10}})
    eid = q.acquire("mock", "m", tokens=100)
    q.commit(eid, 250)
    q2 = _quota(cfg, tmp_path, clock, {"mock/m": {"rpd": 10}})
    assert q2.usage("mock", "m")["tokens_day"] == 250


def test_retry_after_parsing():
    assert parse_retry_after("... 'retryDelay': '34s' ...") == 34
    assert parse_retry_after("Please try again in 1m2.5s.") == 62.5
    assert parse_retry_after("nothing") is None


def _router(cfg, tmp_path, clock, providers, limits=None):
    q = _quota(cfg, tmp_path, clock, limits or {})
    r = LLMRouter(cfg, q, LLMCache(tmp_path / "c.db"), order=list(providers), providers=providers, sleeper=clock.sleep)
    r.check = lambda name: (True, "ok")
    return r


PROMPT = "TASK: map\n=== INPUT ===\n[SEGMENT 0 | 0:00–1:00]\nSLIDE: Intro\nSPEECH: Hello there. This is it.\n=== END INPUT ==="


def test_retry_then_success_honours_retry_after(cfg, tmp_path, clock):
    m = MockProvider("m")
    m.fail_queue = [RateLimitError("429", retry_after=7.0), TransientError("503")]
    r = _router(cfg, tmp_path, clock, {"mock": m})
    res = r.complete(PROMPT, task="t")
    assert res.provider == "mock" and r.stats.retries == 2 and r.stats.calls == 1
    assert clock.slept[0] >= 7.0


def test_daily_429_falls_back_and_cache_prevents_repeat(cfg, tmp_path, clock):
    a, b = MockProvider("a"), MockProvider("b")
    a.name, b.name = "pa", "pb"
    a.fail_queue = [RateLimitError("429 GenerateRequestsPerDayPerProjectPerModel", daily=True)]
    r = _router(cfg, tmp_path, clock, {"pa": a, "pb": b})
    res = r.complete(PROMPT, task="t")
    assert res.provider == "pb" and r.stats.fallbacks == 1
    assert r.quota.is_blocked("pa", "a")
    res2 = r.complete(PROMPT, task="t")          # resume / re-run: served from cache
    assert res2.cached and len(b.calls) == 1 and r.stats.cache_hits == 1


def test_all_exhausted_raises_quota_related(cfg, tmp_path, clock):
    m = MockProvider("m")
    r = _router(cfg, tmp_path, clock, {"mock": m}, {"mock/m": {"rpd": 1}})
    r.complete(PROMPT, task="1")
    with pytest.raises(AllProvidersExhausted) as ei:
        r.complete(PROMPT + " different", task="2")
    assert ei.value.quota_related
