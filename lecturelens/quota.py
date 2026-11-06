"""Free-tier quota manager.

* Persistent usage ledger (SQLite, `.cache/quota.db`): one row per request with
  provider, model, timestamp, tokens and audio-seconds.
* Client-side throttling *before* a limit is hit: sliding-window rate limiter
  (equivalent to a token bucket refilled continuously over 60 s / 3600 s) for
  requests/min, tokens/min and audio-seconds/hour.
* Daily limits with configurable reset time zone (Gemini: midnight Pacific).
  When a daily limit would be exceeded, `QuotaExhausted` is raised so the
  router can fall back to the next provider or pause the run.
* Pre-flight estimates: "this run needs ~X calls / Y tokens; you have ~Z left".
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

WaitCallback = Callable[[float, str], None]


class QuotaExhausted(Exception):
    """A provider/model cannot serve more requests now (daily cap, blocked, too large)."""

    def __init__(self, provider: str, model: str, reason: str, retry_at: float | None = None):
        self.provider, self.model, self.reason, self.retry_at = provider, model, reason, retry_at
        when = ""
        if retry_at:
            when = f" (resets ~{dt.datetime.fromtimestamp(retry_at).strftime('%Y-%m-%d %H:%M')} local time)"
        super().__init__(f"{provider}/{model}: {reason}{when}")


@dataclass
class Limits:
    rpm: float | None = None
    tpm: float | None = None
    rpd: float | None = None
    tpd: float | None = None
    audio_s_per_hour: float | None = None
    audio_s_per_day: float | None = None
    day_reset_tz: str = "UTC"


class QuotaManager:
    def __init__(
        self,
        cfg,
        db_path: str | Path | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        on_wait: WaitCallback | None = None,
    ):
        self.cfg = cfg
        self.safety = float(cfg.get("quota.safety_factor", 0.9) or 1.0)
        self.db_path = Path(db_path) if db_path else cfg.path("cache_dir") / "quota.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.clock, self.sleeper, self.on_wait = clock, sleeper, on_wait
        self._lock = threading.RLock()
        self.waited_s = 0.0          # client-side throttle time in this process
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "provider TEXT, model TEXT, ts REAL, day_key TEXT, tokens INTEGER DEFAULT 0, "
                "audio_s REAL DEFAULT 0, run_id TEXT)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS ix_ev ON events(provider, model, ts)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS blocks (provider TEXT, model TEXT, day_key TEXT, "
                "reason TEXT, PRIMARY KEY (provider, model, day_key))"
            )

    # ------------------------------------------------------------------ infra
    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)

    def limits(self, provider: str, model: str) -> Limits:
        table = self.cfg.get("quota.limits", {}) or {}
        raw = table.get(f"{provider}/{model}") or table.get(f"{provider}/*") or {}
        return Limits(**{k: v for k, v in raw.items() if k in Limits.__dataclass_fields__})

    def _cap(self, value: float | None) -> float | None:
        return None if not value else max(1.0, value * self.safety)

    def day_key(self, provider: str, model: str, ts: float | None = None) -> str:
        tz = ZoneInfo(self.limits(provider, model).day_reset_tz or "UTC")
        return dt.datetime.fromtimestamp(ts if ts is not None else self.clock(), tz).strftime("%Y-%m-%d")

    def next_reset(self, provider: str, model: str) -> float:
        tz = ZoneInfo(self.limits(provider, model).day_reset_tz or "UTC")
        now = dt.datetime.fromtimestamp(self.clock(), tz)
        tomorrow = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return tomorrow.timestamp()

    # ------------------------------------------------------------------ usage
    def _window(self, c, provider: str, model: str, seconds: float) -> list[tuple[float, int, float]]:
        since = self.clock() - seconds
        return c.execute(
            "SELECT ts, tokens, audio_s FROM events WHERE provider=? AND model=? AND ts>? ORDER BY ts",
            (provider, model, since),
        ).fetchall()

    def usage(self, provider: str, model: str) -> dict:
        with self._lock, self._conn() as c:
            minute = self._window(c, provider, model, 60)
            hour = self._window(c, provider, model, 3600)
            day = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens),0), COALESCE(SUM(audio_s),0) FROM events "
                "WHERE provider=? AND model=? AND day_key=?",
                (provider, model, self.day_key(provider, model)),
            ).fetchone()
        return {
            "requests_min": len(minute),
            "tokens_min": sum(r[1] for r in minute),
            "audio_s_hour": sum(r[2] for r in hour),
            "requests_day": day[0],
            "tokens_day": day[1],
            "audio_s_day": day[2],
        }

    def remaining(self, provider: str, model: str) -> dict:
        """Remaining *today* (after the safety factor). None = unlimited."""
        lim, use = self.limits(provider, model), self.usage(provider, model)

        def rem(cap, used):
            cap = self._cap(cap)
            return None if cap is None else max(0, int(cap - used))

        return {
            "requests_day": rem(lim.rpd, use["requests_day"]),
            "tokens_day": rem(lim.tpd, use["tokens_day"]),
            "audio_s_day": rem(lim.audio_s_per_day, use["audio_s_day"]),
            "blocked": self.is_blocked(provider, model),
        }

    def can_afford(self, provider: str, model: str, calls: int, tokens: int = 0) -> bool:
        r = self.remaining(provider, model)
        if r["blocked"]:
            return False
        if r["requests_day"] is not None and r["requests_day"] < calls:
            return False
        if r["tokens_day"] is not None and r["tokens_day"] < tokens:
            return False
        return True

    # ------------------------------------------------------------- blocking
    def mark_exhausted(self, provider: str, model: str, reason: str) -> None:
        """Server said the daily quota is gone: don't try this model again today."""
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO blocks VALUES (?,?,?,?)",
                (provider, model, self.day_key(provider, model), reason[:300]),
            )
        log.warning("Marked %s/%s exhausted for today: %s", provider, model, reason[:120])

    def is_blocked(self, provider: str, model: str) -> bool:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM blocks WHERE provider=? AND model=? AND day_key=?",
                (provider, model, self.day_key(provider, model)),
            ).fetchone()
        return row is not None

    # -------------------------------------------------------------- acquire
    def _minute_wait(self, events, cap_req, cap_tok, tokens, now, span=60.0) -> float:
        """Seconds to wait until adding one request with `tokens` fits the window."""
        wait = 0.0
        if cap_req is not None and len(events) + 1 > cap_req:
            idx = int(len(events) + 1 - cap_req) - 1
            wait = max(wait, events[idx][0] + span - now)
        if cap_tok is not None and tokens > 0:
            total = sum(e[1] for e in events)
            i = 0
            while total + tokens > cap_tok and i < len(events):
                total -= events[i][1]
                wait = max(wait, events[i][0] + span - now)
                i += 1
        return max(0.0, wait)

    def acquire(self, provider: str, model: str, tokens: int = 0, audio_s: float = 0.0, run_id: str | None = None) -> int:
        """Block until the request fits the per-minute/hour windows, then record it.

        Raises QuotaExhausted if a daily limit would be exceeded, the model is
        blocked for today, or a single request is larger than the per-minute cap.
        """
        lim = self.limits(provider, model)
        cap_rpm, cap_tpm = self._cap(lim.rpm), self._cap(lim.tpm)
        cap_rpd, cap_tpd = self._cap(lim.rpd), self._cap(lim.tpd)
        cap_ash, cap_asd = self._cap(lim.audio_s_per_hour), self._cap(lim.audio_s_per_day)

        if cap_tpm is not None and tokens > cap_tpm:
            raise QuotaExhausted(provider, model, f"single request (~{tokens} tok) exceeds tokens/min cap {int(cap_tpm)}")
        if cap_ash is not None and audio_s > cap_ash:
            raise QuotaExhausted(provider, model, f"audio piece ({audio_s:.0f}s) exceeds audio-seconds/hour cap")

        while True:
            with self._lock:
                if self.is_blocked(provider, model):
                    raise QuotaExhausted(provider, model, "daily quota exhausted (server)", self.next_reset(provider, model))
                use = self.usage(provider, model)
                if cap_rpd is not None and use["requests_day"] + 1 > cap_rpd:
                    raise QuotaExhausted(provider, model, f"daily request cap reached ({use['requests_day']}/{int(cap_rpd)})", self.next_reset(provider, model))
                if cap_tpd is not None and use["tokens_day"] + tokens > cap_tpd:
                    raise QuotaExhausted(provider, model, f"daily token cap reached ({use['tokens_day']}+{tokens}>{int(cap_tpd)})", self.next_reset(provider, model))
                if cap_asd is not None and use["audio_s_day"] + audio_s > cap_asd:
                    raise QuotaExhausted(provider, model, "daily audio-seconds cap reached", self.next_reset(provider, model))

                now = self.clock()
                with self._conn() as c:
                    minute = self._window(c, provider, model, 60)
                    hour = self._window(c, provider, model, 3600) if cap_ash else []
                wait = self._minute_wait(minute, cap_rpm, cap_tpm, tokens, now)
                if cap_ash is not None and audio_s > 0:
                    ev = [(t, a, 0) for (t, _tok, a) in hour]
                    wait = max(wait, self._minute_wait(ev, None, cap_ash, audio_s, now, span=3600.0))
                if wait <= 0:
                    with self._conn() as c:
                        cur = c.execute(
                            "INSERT INTO events (provider, model, ts, day_key, tokens, audio_s, run_id) VALUES (?,?,?,?,?,?,?)",
                            (provider, model, now, self.day_key(provider, model, now), int(tokens), float(audio_s), run_id),
                        )
                        return int(cur.lastrowid)
            reason = f"rate limit {provider}/{model}"
            log.info("Throttling %.1fs to stay under %s free-tier limits", wait, f"{provider}/{model}")
            self._sleep(wait + 0.25, reason)

    def commit(self, event_id: int, tokens: int) -> None:
        """Replace the estimated token count with the real one from the response."""
        with self._lock, self._conn() as c:
            c.execute("UPDATE events SET tokens=? WHERE id=?", (int(tokens), event_id))

    def _sleep(self, seconds: float, reason: str) -> None:
        self.waited_s += seconds
        if self.on_wait:
            self.on_wait(seconds, reason)
        self.sleeper(seconds)

    # -------------------------------------------------------------- reports
    def report(self, pairs: list[tuple[str, str]]) -> list[dict]:
        rows = []
        for provider, model in pairs:
            lim, use, rem = self.limits(provider, model), self.usage(provider, model), self.remaining(provider, model)
            rows.append({
                "provider": provider, "model": model,
                "requests_today": use["requests_day"], "rpd_limit": lim.rpd,
                "requests_left": rem["requests_day"],
                "tokens_today": use["tokens_day"], "tpd_limit": lim.tpd, "tokens_left": rem["tokens_day"],
                "audio_s_today": round(use["audio_s_day"]), "blocked": rem["blocked"],
                "resets": dt.datetime.fromtimestamp(self.next_reset(provider, model)).strftime("%Y-%m-%d %H:%M"),
            })
        return rows
