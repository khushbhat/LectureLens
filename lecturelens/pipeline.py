"""End-to-end orchestration with per-stage caching, run logs, pause & resume.

Run folder `runs/<slug>_<hash>/` holds every intermediate artifact plus
`state.json` (status / resume info) and `run_log.json` (per-stage timings,
providers, tokens, calls, retries, fallbacks, config snapshot).
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .audio import check_ffmpeg, probe_duration
from .cache import LLMCache
from .ingest import fetch_youtube_info, ingest, is_url
from .llm import AllProvidersExhausted, LLMRouter
from .ocr import ocr_slides
from .preprocess import preprocess
from .quota import QuotaManager
from .segment import segment
from .slides import detect_slides
from .summarize import Prompts, Summarizer, estimate_calls
from .textrank import extractive_result
from .transcribe import transcribe
from .utils import (estimate_tokens, file_hash, fmt_ts, read_json, short_hash, slugify, write_json)

log = logging.getLogger(__name__)

TOKENS_PER_MIN = 200  # ~150 spoken words/min ≈ 200 tokens/min (pre-flight before transcript exists)

VARIANTS = {
    # name: (use_slides, segments file, map dir, reduce dir)
    "lecturelens": (True, "segments.json", "map", "reduce"),
    "hier_speech": (False, "segments_speech.json", "map_speech", "reduce_speech"),
}


@dataclass
class RunOptions:
    use_slides: bool = True                  # False = speech-only ablation (no frames/OCR)
    providers: list[str] | None = None       # LLM order override, e.g. ["groq"]
    transcript_order: list[str] | None = None
    verify: bool = True
    export: bool = True


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    source: dict = field(default_factory=dict)
    transcript: dict = field(default_factory=dict)
    sentences: dict = field(default_factory=dict)
    keyframes: dict = field(default_factory=dict)
    slides: dict = field(default_factory=dict)


class QuotaPaused(Exception):
    def __init__(self, run_id: str, message: str):
        self.run_id = run_id
        super().__init__(message)


class Pipeline:
    def __init__(self, cfg, progress: Callable[[str], None] | None = None,
                 on_wait: Callable[[float, str], None] | None = None,
                 sleeper: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.progress = progress or (lambda m: log.info(m))
        self.on_wait = on_wait
        self.sleeper = sleeper
        self.quota = QuotaManager(cfg, on_wait=on_wait, sleeper=sleeper)
        self.cache = LLMCache(cfg.path("cache_dir") / "llm_cache.sqlite")
        self.runs_dir = cfg.path("runs_dir")

    # -------------------------------------------------------------- helpers
    def router(self, run_id: str | None, order: list[str] | None = None) -> LLMRouter:
        return LLMRouter(self.cfg, self.quota, self.cache, order=order, on_wait=self.on_wait, run_id=run_id,
                         sleeper=self.sleeper)

    def make_run_id(self, source: str, info: dict | None = None) -> str:
        if is_url(source):
            info = info or {}
            key = info.get("id") or source
            title = info.get("title") or key
        else:
            p = Path(source).expanduser().resolve()
            key, title = file_hash(p), p.stem
        return f"{slugify(title)}_{short_hash(str(key))}"

    @contextmanager
    def stage(self, log_: dict, name: str, artifact: Path | None = None):
        cached = artifact is not None and artifact.exists()
        prev = log_["stages"].get(name, {})
        t0 = time.time()
        self.progress(f"▶ {name}{' (cached)' if cached else ''}")
        rec = {"status": "running"}
        log_["stages"][name] = rec
        try:
            yield rec
        except BaseException:
            rec.update(status="failed", seconds=round(time.time() - t0, 2))
            raise
        secs = round(time.time() - t0, 2)
        if cached and prev.get("seconds") is not None:
            rec.update({**{k: v for k, v in prev.items() if k not in ("status",)}, "status": "cached",
                        "seconds_this_run": secs})
        else:
            rec.setdefault("seconds", secs)
            rec["status"] = "done"

    # ------------------------------------------------------------- preflight
    def preflight(self, source: str, options: RunOptions | None = None, info: dict | None = None,
                  n_tokens: int | None = None) -> dict:
        """Estimate LLM calls/tokens/time and compare with today's remaining free quota."""
        options = options or RunOptions()
        duration = 0.0
        has_captions = False
        if n_tokens is None:
            if is_url(source):
                info = info or fetch_youtube_info(source, self.cfg)
                duration = float(info.get("duration") or 0)
                langs = self.cfg.get("transcript.caption_langs", ["en"])
                has_captions = any(k.split("-")[0] in [l.split("-")[0] for l in langs] for k in (info.get("subtitles") or {}))
            else:
                check_ffmpeg()
                duration = probe_duration(source)
            n_tokens = int(duration / 60 * TOKENS_PER_MIN * (1.25 if options.use_slides else 1.0))
        router = self.router(None, options.providers)
        chain = router.available_chain()
        prompts = Prompts(self.cfg.path("prompts_dir"))
        rows, plan = [], None
        for name in chain:
            est = estimate_calls(n_tokens, self.cfg, name, prompts)
            p = router.provider(name)
            rem = self.quota.remaining(p.name, p.model)
            ok = self.quota.can_afford(p.name, p.model, est["calls"], est["tokens"])
            rows.append({**est, "model": p.model, "remaining_requests": rem["requests_day"],
                         "remaining_tokens": rem["tokens_day"], "fits_today": ok})
            if ok and plan is None:
                plan = name
        est_time = 0.0
        if duration:
            asr = 0 if has_captions else (duration / 120 if self.cfg.api_key("GROQ_API_KEY") else duration * 0.15)
            ocr = (duration / 60) * 2.0 if options.use_slides else 0
            frames = duration / 120 if options.use_slides else 0
            est_time = asr + ocr + frames + 60
        if rows:
            r0 = next((r for r in rows if r["provider"] == plan), rows[0])
            rpm = self.quota.limits(r0["provider"], r0["model"]).rpm or 60
            est_time += r0["calls"] * 12 + max(0, r0["calls"] - rpm) * 60 / rpm
        msg = []
        if not chain:
            msg.append("No LLM provider available (no API keys and Ollama not running): output will be an "
                       "extractive TextRank summary + notes skeleton.")
        for r in rows:
            left = "unlimited" if r["remaining_requests"] is None else f"~{r['remaining_requests']} requests"
            msg.append(f"{r['provider']}/{r['model']}: this run needs ~{r['calls']} calls / ~{r['tokens']:,} tokens; "
                       f"you have {left} left today → {'OK' if r['fits_today'] else 'NOT enough'}")
        if chain and plan is None:
            msg.append("No provider has enough quota left today for the whole run: it will run as far as possible, "
                       "save progress and tell you to resume after the daily reset.")
        elif plan and chain and plan != chain[0]:
            msg.append(f"Primary provider short on quota → planning batches for fallback '{plan}'.")
        return {"duration": duration, "est_input_tokens": n_tokens, "chain": chain, "plan_provider": plan or (chain[0] if chain else None),
                "providers": rows, "est_time_s": round(est_time), "has_captions": has_captions, "message": "\n".join(msg)}

    # ----------------------------------------------------------------- stages
    def prepare(self, source: str, options: RunOptions, run_id: str | None = None, info: dict | None = None,
                log_: dict | None = None) -> RunContext:
        """Ingest → transcript → preprocess → (keyframes → OCR). All cached in the run folder."""
        if is_url(source) and info is None and run_id is None:
            info = fetch_youtube_info(source, self.cfg)
        run_id = run_id or self.make_run_id(source, info)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(run_id, run_dir)
        log_ = log_ if log_ is not None else {"stages": {}}
        cfg = self.cfg
        if options.transcript_order:
            cfg = cfg.with_overrides({"transcript": {"order": options.transcript_order}})
        if not options.use_slides:
            cfg = cfg.with_overrides({"ingest": {"download_video": False}})

        with self.stage(log_, "ingest", run_dir / "source.json") as rec:
            ctx.source = read_json(run_dir / "source.json") or ingest(source, run_dir, cfg, info)
            rec.update(kind=ctx.source["kind"], duration=ctx.source.get("duration"),
                       captions=ctx.source.get("subtitle_kind"))
        with self.stage(log_, "transcribe", run_dir / "transcript.json") as rec:
            ctx.transcript = transcribe(run_dir, ctx.source, cfg, self.quota, run_id, self.progress)
            rec.update(source=ctx.transcript["source"], segments=len(ctx.transcript["segments"]),
                       real_time_factor=ctx.transcript.get("real_time_factor"))
        with self.stage(log_, "preprocess", run_dir / "sentences.json") as rec:
            ctx.sentences = preprocess(run_dir, ctx.transcript, cfg)
            rec.update(sentences=len(ctx.sentences["sentences"]),
                       words=sum(len(s["text"].split()) for s in ctx.sentences["sentences"]))
        duration = float(ctx.source.get("duration") or ctx.transcript.get("duration") or 0)
        if options.use_slides:
            with self.stage(log_, "keyframes", run_dir / "keyframes.json") as rec:
                if not ctx.source.get("video_path") and ctx.source.get("kind") == "youtube" and cfg.get("ingest.download_video", True):
                    ctx.source = ingest(ctx.source["source"], run_dir, cfg)  # video skipped earlier (speech-only run)
                ctx.keyframes = detect_slides(run_dir, ctx.source.get("video_path"), duration, cfg, self.progress)
                rec.update(keyframes=len(ctx.keyframes.get("keyframes", [])),
                           frames_sampled=ctx.keyframes.get("frames_sampled"))
            with self.stage(log_, "ocr", run_dir / "slides.json") as rec:
                ctx.slides = ocr_slides(run_dir, ctx.keyframes, duration, cfg, self.progress)
                rec.update(slides=len(ctx.slides.get("slides", [])), visual_mode=ctx.slides.get("visual_mode"),
                           **ctx.slides.get("stats", {}))
        else:
            ctx.slides = {"visual_mode": "low", "slides": [], "reason": "slides disabled"}
        return ctx

    def summarize_variant(self, ctx: RunContext, variant: str, router: LLMRouter, log_: dict,
                          plan_provider: str | None = None) -> tuple[dict, dict]:
        """Run one summarization system over a prepared context. Returns (final, segments)."""
        sents = ctx.sentences["sentences"]
        slides = ctx.slides if variant == "lecturelens" else {"visual_mode": "low", "slides": []}
        use_slides, seg_file, map_dir, red_dir = VARIANTS.get(variant, VARIANTS["hier_speech"])
        chapters = ctx.source.get("chapters") or []
        with self.stage(log_, f"segment[{variant}]", ctx.run_dir / seg_file) as rec:
            segs = segment(ctx.run_dir, ctx.sentences, slides, self.cfg, chapters, use_slides, seg_file)
            rec.update(mode=segs["mode"], segments=len(segs["segments"]))
        slides_by_id = {s["slide_id"]: s for s in slides.get("slides", [])}
        title = ctx.source.get("title") or ctx.run_id
        duration = float(ctx.source.get("duration") or 0)

        chain = router.available_chain()
        if not chain:
            final = extractive_result(sents, segs["segments"], slides_by_id, title)
            log_["stages"]["summarize[" + variant + "]"] = {"status": "extractive", "reason": "no LLM provider available"}
            return final, segs

        summ = Summarizer(self.cfg, router, ctx.run_dir, self.progress, plan_provider)
        before = router.stats.calls
        with self.stage(log_, f"map[{variant}]", ctx.run_dir / red_dir / "final.json") as rec:
            items = summ.run_map(segs["segments"], sents, slides_by_id, use_slides, map_dir)
            batches = read_json(ctx.run_dir / map_dir / "batches.json") or []
            rec.update(batches=len(batches), plan_provider=summ.plan_provider, llm_calls=router.stats.calls - before)
        before = router.stats.calls
        with self.stage(log_, f"reduce[{variant}]", ctx.run_dir / red_dir / "final.json") as rec:
            final = summ.run_reduce(items, segs["segments"], title, duration, red_dir).model_dump()
            meta = read_json(ctx.run_dir / red_dir / "reduce_meta.json") or {}
            rec.update(llm_calls=router.stats.calls - before, **meta)
        if summ.notes:
            log_.setdefault("notes", []).extend(summ.notes)
        return final, segs

    # -------------------------------------------------------------------- run
    def run(self, source: str, options: RunOptions | None = None, run_id: str | None = None) -> dict:
        options = options or RunOptions()
        check_ffmpeg()
        info = fetch_youtube_info(source, self.cfg) if is_url(source) and run_id is None else None
        run_id = run_id or self.make_run_id(source, info)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_f, log_f = run_dir / "state.json", run_dir / "run_log.json"
        state = read_json(state_f) or {"run_id": run_id, "source": source, "options": asdict(options),
                                       "created": dt.datetime.now().isoformat(timespec="seconds")}
        state.update(status="running", updated=dt.datetime.now().isoformat(timespec="seconds"), message="")
        write_json(state_f, state)
        prev_log = read_json(log_f) or {}
        log_ = {"run_id": run_id, "source": source, "stages": prev_log.get("stages", {}),
                "config_snapshot": self.cfg.snapshot(), "options": asdict(options),
                "attempts": prev_log.get("attempts", 0) + 1}
        router = self.router(run_id, options.providers)
        t_start = time.time()
        waited0 = self.quota.waited_s
        try:
            ctx = self.prepare(source, options, run_id, info, log_)
            n_tok = estimate_tokens(ctx.sentences.get("text", "")) + sum(
                estimate_tokens(s.get("text", "")) for s in ctx.slides.get("slides", []))
            pf = self.preflight(source, options, n_tokens=n_tok)
            log_["preflight"] = pf
            self.progress(pf["message"].replace("\n", " | ") or "preflight ok")
            final, segs = self.summarize_variant(ctx, "lecturelens", router, log_, pf["plan_provider"])

            verification = {}
            if options.verify and self.cfg.get("verify.enabled", True):
                from .verify import verify_final

                with self.stage(log_, "verify", None) as rec:
                    verification = verify_final(final, ctx.sentences["sentences"], ctx.slides.get("slides", []),
                                                segs["segments"], self.cfg)
                    rec.update(**{k: v for k, v in verification.items() if k != "nli_model"})

            words_in = sum(len(s["text"].split()) for s in ctx.sentences["sentences"])
            words_out = len((final.get("concise_summary") or "").split())
            ev_counts = {e: sum(1 for kp in final.get("key_points", []) if kp.get("evidence") == e) for e in ("slide", "speech", "both")}
            stats = {
                "llm_calls": router.stats.calls, "cache_hits": router.stats.cache_hits,
                "input_tokens": router.stats.input_tokens, "output_tokens": router.stats.output_tokens,
                "retries": router.stats.retries, "fallbacks": router.stats.fallbacks,
                "rate_limit_wait_s": round(router.stats.wait_s + self.quota.waited_s - waited0, 1),
                "providers": sorted({f"{e['provider']}/{e['model']}" for e in router.stats.events}),
                "by_provider": router.stats.by_provider, "transcript_words": words_in,
                "summary_words": words_out, "compression_ratio": round(words_in / words_out, 1) if words_out else None,
                "evidence_split": ev_counts, "wall_clock_s": round(time.time() - t_start, 1),
            }
            result = {
                "run_id": run_id, "status": "done", "source": ctx.source, "final": final,
                "verification": verification, "slides": ctx.slides.get("slides", []),
                "visual_mode": ctx.slides.get("visual_mode"), "segments": segs["segments"],
                "segmentation_mode": segs["mode"], "transcript_source": ctx.transcript.get("source"),
                "stats": stats, "chapters_youtube": ctx.source.get("chapters"),
            }
            if options.export:
                from .export import export_all

                with self.stage(log_, "export", None) as rec:
                    result["exports"] = export_all(run_dir, result, self.cfg)
                    rec.update(files=list(result["exports"]))
                write_json(run_dir / "lecture.json", result)
            state.update(status="done", message="")
            return result
        except AllProvidersExhausted as e:
            if not e.quota_related:
                state.update(status="error", message=str(e))
                raise
            resets = []
            for name in router.order:
                try:
                    p = router.provider(name)
                    resets.append(f"{name}: {dt.datetime.fromtimestamp(self.quota.next_reset(p.name, p.model)):%Y-%m-%d %H:%M}")
                except Exception:
                    pass
            msg = (f"Free-tier quota exhausted for now ({e}). All completed work is saved. "
                   f"Resume with:  python -m lecturelens resume {run_id}   (daily resets — {'; '.join(resets)})")
            state.update(status="paused", message=msg)
            raise QuotaPaused(run_id, msg) from e
        except Exception as e:
            state.update(status="error", message=f"{type(e).__name__}: {e}")
            raise
        finally:
            state["updated"] = dt.datetime.now().isoformat(timespec="seconds")
            write_json(state_f, state)
            log_["llm"] = router.stats.as_dict()
            log_["status"] = state["status"]
            log_["wall_clock_s_this_attempt"] = round(time.time() - t_start, 1)
            write_json(log_f, log_)

    def resume(self, run_id: str) -> dict:
        state = read_json(self.runs_dir / run_id / "state.json")
        if not state:
            raise FileNotFoundError(f"No run '{run_id}' in {self.runs_dir}")
        opts = RunOptions(**state.get("options", {}))
        self.progress(f"Resuming {run_id} (last status: {state.get('status')})")
        return self.run(state["source"], opts, run_id=run_id)

    def list_runs(self) -> list[dict]:
        out = []
        for d in sorted(self.runs_dir.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            s = read_json(d)
            src = read_json(d.parent / "source.json") or {}
            out.append({"run_id": s.get("run_id"), "status": s.get("status"), "title": src.get("title"),
                        "duration": fmt_ts(src.get("duration")), "updated": s.get("updated"), "message": s.get("message", "")})
        return out
