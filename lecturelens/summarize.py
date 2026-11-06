"""Stage 9 — Quota-aware hierarchical summarization (map -> recursive reduce).

Map: one LLM call per batch of slide-aligned segments -> per-segment JSON.
Reduce: if the map outputs exceed the provider's reduce budget, they are
first condensed group-wise (recursively: this is how 2–3 h lectures fit);
then a single final call (or 2 for small-TPM providers) produces summary,
key points, notes, glossary, chapters and revision questions.
Every call is cached on disk (run folder + global LLM cache) => resumable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Type

from pydantic import BaseModel, ValidationError

from .batching import pack_batches
from .llm import AllProvidersExhausted, LLMRouter
from .schemas import (SCHEMA_HINTS, FinalNotesPart, FinalOutput, FinalSummaryPart, MapOutput, SegmentSummary)
from .segment import render_segment
from .utils import estimate_tokens, extract_json, fmt_ts, read_json, repair_truncated_json, write_json

log = logging.getLogger(__name__)


class Prompts:
    def __init__(self, prompts_dir: Path):
        self.dir = Path(prompts_dir)

    def load(self, name: str) -> str:
        return (self.dir / name).read_text(encoding="utf-8")

    def fill(self, name: str, **kw) -> str:
        t = self.load(name)
        for k, v in kw.items():
            t = t.replace("{{" + k + "}}", str(v))
        return t

    def overhead(self, name: str) -> int:
        return estimate_tokens(self.load(name)) + estimate_tokens(self.load("system_v1.txt"))


def _compact(items: list[dict]) -> str:
    def strip(d):
        return {k: v for k, v in d.items() if v not in ("", [], None)}
    return json.dumps([strip(i) for i in items], ensure_ascii=False, separators=(",", ":"))


class Summarizer:
    def __init__(self, cfg, router: LLMRouter, run_dir: Path, progress: Callable[[str], None] | None = None,
                 plan_provider: str | None = None):
        self.cfg = cfg
        self.router = router
        self.run_dir = Path(run_dir)
        self.prompts = Prompts(cfg.path("prompts_dir"))
        self.system = self.prompts.load("system_v1.txt").strip()
        self.progress = progress or (lambda m: None)
        chain = router.available_chain()
        self.plan_provider = plan_provider or (chain[0] if chain else "gemini")
        self.notes: list[str] = []
        self._usage_dir: Path | None = None

    def _log_usage(self, res, task: str) -> None:
        """Append per-call usage to <stage dir>/usage.jsonl (true cost survives caching)."""
        if self._usage_dir is None:
            return
        rec = {"task": task, "provider": res.provider, "model": res.model, "input_tokens": res.input_tokens,
               "output_tokens": res.output_tokens, "cached": res.cached}
        with open(self._usage_dir / "usage.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------- budgets
    def budget(self, key: str, default: int) -> int:
        return self.router.budget(self.plan_provider, key, default)

    def split_final(self) -> bool:
        return bool(self.cfg.get(f"llm.{self.plan_provider}.split_final", False))

    # ----------------------------------------------------------- JSON calls
    def call_json(self, prompt: str, model_cls: Type[BaseModel], task: str, schema_key: str) -> BaseModel:
        res = self.router.complete(prompt, self.system, json_mode=True, task=task)
        self._log_usage(res, task)
        try:
            return model_cls.model_validate(extract_json(res.text))
        except (ValueError, ValidationError) as err:
            first_err = err
        # 1) free local fix for truncated output, 2) one LLM repair call, 3) keep whatever is valid
        try:
            return model_cls.model_validate(repair_truncated_json(res.text))
        except (ValueError, ValidationError):
            pass
        log.warning("[%s] invalid JSON from %s (%s); one repair attempt", task, res.provider, str(first_err)[:160])
        repaired = self._repair(res.text, first_err, task, schema_key)
        if repaired is not None:
            try:
                return model_cls.model_validate(repaired)
            except ValidationError:
                salvaged = self._salvage(repaired, model_cls)
                if salvaged is not None:
                    return salvaged
        salvaged = self._salvage_text(res.text, model_cls)
        if salvaged is not None:
            return salvaged
        raise ValueError(f"[{task}] could not parse model output: {str(first_err)[:200]}")

    def _repair(self, text: str, err: Exception, task: str, schema_key: str):
        prompt = self.prompts.fill("repair_v1.txt", schema_hint=SCHEMA_HINTS[schema_key], error=str(err)[:500], broken=text)
        # cheap model first if Groq is configured; otherwise whatever chain is available
        chain = self.router.available_chain()
        try:
            if "groq" in chain and self.cfg.get("llm.groq.repair_model"):
                p = self.router.provider("groq")
                orig = p.model
                p.model = self.cfg.get("llm.groq.repair_model")
                try:
                    res = self.router.complete(prompt, self.system, True, task=f"{task}:repair", chain=["groq"])
                finally:
                    p.model = orig
            else:
                res = self.router.complete(prompt, self.system, True, task=f"{task}:repair")
            self._log_usage(res, f"{task}:repair")
            return extract_json(res.text)
        except (AllProvidersExhausted, ValueError) as e:
            log.warning("[%s] repair failed: %s", task, str(e)[:160])
            return None

    @staticmethod
    def _salvage(data, model_cls):
        """Keep whatever is valid: for map outputs validate segment-by-segment."""
        if model_cls in (MapOutput,) and isinstance(data, (dict, list)):
            raw = data.get("segments", []) if isinstance(data, dict) else data
            good = []
            for item in raw:
                try:
                    good.append(SegmentSummary.model_validate(item))
                except ValidationError:
                    continue
            return MapOutput(segments=good) if good else None
        if isinstance(data, dict):
            fields = {}
            for k, f in model_cls.model_fields.items():
                if k in data:
                    try:
                        model_cls.model_validate({k: data[k]})
                        fields[k] = data[k]
                    except ValidationError:
                        pass
            try:
                return model_cls.model_validate(fields)
            except ValidationError:
                return None
        return None

    def _salvage_text(self, text, model_cls):
        for parse in (extract_json, repair_truncated_json):
            try:
                got = self._salvage(parse(text), model_cls)
                if got is not None:
                    return got
            except ValueError:
                continue
        return None

    # ------------------------------------------------------------------ map
    def run_map(self, segments: list[dict], sents: list[dict], slides_by_id: dict, include_slides: bool = True,
                out_dir: str = "map") -> list[SegmentSummary]:
        mdir = self.run_dir / out_dir
        mdir.mkdir(parents=True, exist_ok=True)
        self._usage_dir = mdir
        rendered = {s["segment_id"]: render_segment(s, sents, slides_by_id, include_slides) for s in segments}
        budget = self.budget("batch_input_tokens", 8000) - self.prompts.overhead("map_v1.txt")
        plan_f = mdir / "batches.json"
        batches = read_json(plan_f)
        if batches is None:
            batches = pack_batches([(sid, rendered[sid]) for sid in rendered], budget)
            write_json(plan_f, batches)
        log.info("Map: %d segments -> %d batches (budget %d tok/call, plan provider %s)",
                 len(segments), len(batches), budget, self.plan_provider)

        results: dict[int, SegmentSummary] = {}
        for b in batches:
            self.progress(f"Map: batch {b['batch_id'] + 1}/{len(batches)} ({len(b['segment_ids'])} segments)")
            for item in self._map_batch(b["segment_ids"], rendered, mdir, f"batch_{b['batch_id']:03d}"):
                results[item.segment_id] = item

        # segments the model skipped: extractive stand-in so the reduce still covers them
        by_id = {s["segment_id"]: s for s in segments}
        for sid, seg in by_id.items():
            if sid not in results:
                speech = [sents[i]["text"] for i in seg["sentence_ids"]]
                self.notes.append(f"segment {sid} missing from map output; extractive stand-in used")
                results[sid] = SegmentSummary(segment_id=sid, title=f"Part {fmt_ts(seg['start'])}",
                                              summary=" ".join(speech[:3]),
                                              key_points=[{"text": t, "evidence": "speech"} for t in speech[:2]])
        return [results[k] for k in sorted(results)]

    def _map_batch(self, seg_ids: list[int], rendered: dict, mdir: Path, name: str) -> list[SegmentSummary]:
        f = mdir / f"{name}.json"
        cached = read_json(f)
        if cached is not None:
            return [SegmentSummary.model_validate(x) for x in cached]
        prompt = self.prompts.fill("map_v1.txt", segments="\n\n".join(rendered[s] for s in seg_ids))
        try:
            out = self.call_json(prompt, MapOutput, f"map:{name}", "map").segments
        except ValueError as e:  # unparseable even after repair: stand-ins fill the gap; not cached
            log.warning("%s", e)
            self.notes.append(f"{name}: unparseable model output; extractive stand-ins used")
            return []
        except AllProvidersExhausted as e:
            if e.too_large and len(seg_ids) > 1:  # fallback provider has a smaller per-call limit
                half = len(seg_ids) // 2
                log.warning("Batch %s too large for the available provider; splitting in two", name)
                return (self._map_batch(seg_ids[:half], rendered, mdir, f"{name}a")
                        + self._map_batch(seg_ids[half:], rendered, mdir, f"{name}b"))
            raise
        valid = set(seg_ids)
        out = [o for o in out if o.segment_id in valid]
        write_json(f, [o.model_dump() for o in out])
        return out

    # --------------------------------------------------------------- reduce
    def run_reduce(self, items: list[SegmentSummary], segments: list[dict], title: str, duration: float,
                   out_dir: str = "reduce") -> FinalOutput:
        rdir = self.run_dir / out_dir
        rdir.mkdir(parents=True, exist_ok=True)
        self._usage_dir = rdir
        final_f = rdir / "final.json"
        cached = read_json(final_f)
        if cached is not None:
            return FinalOutput.model_validate(cached)

        tmpl = "final_summary_v1.txt" if self.split_final() else "reduce_v1.txt"
        budget = self.budget("reduce_input_tokens", 20000) - self.prompts.overhead(tmpl)
        cur = [i.model_dump() for i in items]
        level = 0
        while estimate_tokens(_compact(cur)) > budget and len(cur) > 1:
            level += 1
            cur = self._reduce_level(cur, budget, level, rdir, segments)
            if level > 6:
                log.warning("Reduce recursion limit hit; truncating input")
                break
        payload = _compact(cur)
        kw = dict(video_title=title, duration=fmt_ts(duration), items=payload,
                  num_mcq=self.cfg.get("summarize.num_mcq", 5), num_short=self.cfg.get("summarize.num_short", 5))
        self.progress(f"Reduce: final synthesis over {len(cur)} sections")
        if self.split_final():
            a = self.call_json(self.prompts.fill("final_summary_v1.txt", **kw), FinalSummaryPart, "final_summary", "final")
            b = self.call_json(self.prompts.fill("final_notes_v1.txt", **kw), FinalNotesPart, "final_notes", "final")
            final = FinalOutput(**a.model_dump(), **b.model_dump())
        else:
            final = self.call_json(self.prompts.fill("reduce_v1.txt", **kw), FinalOutput, "reduce", "final")
        final = self.attach_times(final, segments)
        write_json(final_f, final.model_dump())
        write_json(rdir / "reduce_meta.json", {"levels": level, "final_input_items": len(cur),
                                               "split_final": self.split_final()})
        return final

    def _reduce_level(self, items: list[dict], budget: int, level: int, rdir: Path, segments: list[dict]) -> list[dict]:
        groups, cur, tok = [], [], 0
        for it in items:
            t = estimate_tokens(_compact([it]))
            if cur and tok + t > budget:
                groups.append(cur)
                cur, tok = [], 0
            cur.append(it)
            tok += t
        if cur:
            groups.append(cur)
        if len(groups) == 1 and len(groups[0]) > 1:  # make progress even if one group "fits"
            mid = len(groups[0]) // 2
            groups = [groups[0][:mid], groups[0][mid:]]
        seg_by_id = {s["segment_id"]: s for s in segments}
        out = []
        for g, grp in enumerate(groups):
            f = rdir / f"level{level}_group{g:03d}.json"
            cached = read_json(f)
            if cached is None:
                ids = [i for it in grp for i in (it.get("segment_ids") or [it["segment_id"]])]
                span = f"{fmt_ts(seg_by_id.get(min(ids), {}).get('start'))}–{fmt_ts(seg_by_id.get(max(ids), {}).get('end'))}"
                self.progress(f"Reduce level {level}: group {g + 1}/{len(groups)}")
                prompt = self.prompts.fill("reduce_group_v1.txt", items=_compact(grp), time_range=span,
                                           max_items=max(1, min(4, len(grp) // 2)))
                res = self.call_json(prompt, MapOutput, f"reduce_group:L{level}g{g}", "reduce_group")
                allowed = set(ids)
                cached = []
                for s in res.segments:
                    s.segment_ids = [i for i in s.segment_ids if i in allowed] or [min(ids)]
                    s.segment_id = s.segment_ids[0]
                    cached.append(s.model_dump())
                write_json(f, cached)
            out.extend(cached)
        log.info("Reduce level %d: %d items -> %d", level, len(items), len(out))
        return out

    @staticmethod
    def attach_times(final: FinalOutput, segments: list[dict]) -> FinalOutput:
        """Timestamps come from segment ids (never trusted from the LLM)."""
        seg = {s["segment_id"]: s for s in segments}
        for kp in final.key_points:
            kp.segment_ids = [i for i in kp.segment_ids if i in seg]
            if kp.segment_ids:
                kp.start = seg[min(kp.segment_ids)]["start"]
                kp.end = seg[max(kp.segment_ids)]["end"]
        for ch in final.chapters:
            ch.segment_ids = [i for i in ch.segment_ids if i in seg]
            if ch.segment_ids:
                ch.start = seg[min(ch.segment_ids)]["start"]
                ch.end = seg[max(ch.segment_ids)]["end"]
        final.chapters.sort(key=lambda c: (c.start is None, c.start or 0))
        return final

    # ---------------------------------------------------------- single pass
    def run_single_pass(self, segments: list[dict], sents: list[dict], title: str, duration: float,
                        out_dir: str = "single_pass") -> FinalOutput:
        """Baseline: the whole transcript (speech only) in ONE call."""
        d = self.run_dir / out_dir
        f = d / "final.json"
        cached = read_json(f)
        if cached is not None:
            return FinalOutput.model_validate(cached)
        d.mkdir(parents=True, exist_ok=True)
        self._usage_dir = d
        blocks = "\n\n".join(render_segment(s, sents, {}, include_slides=False) for s in segments)
        prompt = self.prompts.fill("single_pass_v1.txt", video_title=title, duration=fmt_ts(duration), segments=blocks,
                                   num_mcq=self.cfg.get("summarize.num_mcq", 5), num_short=self.cfg.get("summarize.num_short", 5))
        self.progress("Single-pass: whole transcript in one call")
        final = self.attach_times(self.call_json(prompt, FinalOutput, "single_pass", "final"), segments)
        write_json(f, final.model_dump())
        return final


def estimate_calls(n_tokens_input: int, cfg, provider: str, prompts: Prompts) -> dict:
    """Pre-flight estimate of LLM calls/tokens for a hierarchical run."""
    b_in = int(cfg.get(f"llm.{provider}.batch_input_tokens", 8000)) - prompts.overhead("map_v1.txt")
    r_in = int(cfg.get(f"llm.{provider}.reduce_input_tokens", 20000))
    n_map = max(1, -(-n_tokens_input // max(500, b_in)))
    map_out = int(n_tokens_input * 0.45)  # map outputs are ~40-50% of input size
    n_group = 0
    size = map_out
    while size > r_in:
        groups = -(-size // r_in)
        n_group += groups
        size = int(size * 0.5)
    n_final = 2 if cfg.get(f"llm.{provider}.split_final", False) else 1
    calls = n_map + n_group + n_final
    tokens = (n_tokens_input + n_map * prompts.overhead("map_v1.txt")) + map_out * 2 + n_group * 1500 + n_final * (min(map_out, r_in) + 5000)
    return {"provider": provider, "map_calls": n_map, "reduce_group_calls": n_group, "final_calls": n_final,
            "calls": calls, "tokens": int(tokens)}


def read_usage(*dirs: Path) -> dict:
    """Total LLM cost recorded in stage dirs (dedup by task; first record wins)."""
    seen, calls, tin, tout, providers = set(), 0, 0, 0, set()
    for d in dirs:
        f = Path(d) / "usage.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["task"] in seen:
                continue
            seen.add(r["task"])
            calls += 1
            tin += r.get("input_tokens") or 0
            tout += r.get("output_tokens") or 0
            providers.add(f"{r['provider']}/{r['model']}")
    return {"calls": calls, "input_tokens": tin, "output_tokens": tout, "providers": sorted(providers)}
