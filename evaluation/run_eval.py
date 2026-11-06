"""Quota-aware evaluation grid: videos × systems -> metrics, tables, figures, report.

    python -m lecturelens eval                       # new results/<timestamp>/
    python -m lecturelens eval --resume              # continue the latest grid (e.g. tomorrow)
    python -m lecturelens eval --max-calls 10        # cap today's LLM usage
    python -m lecturelens eval --provider groq       # same LLM for every LLM-based system

Systems: textrank (extractive, 0 calls) · single_pass (whole transcript, 1 call,
speech only) · hier_speech (our hierarchy without slides) · lecturelens (full).
All LLM systems use ONE provider (no fallback) for a fair comparison. Every
LLM output is cached, so resuming never recomputes anything.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import random
import time
from pathlib import Path

from lecturelens.config import load_config
from lecturelens.llm import AllProvidersExhausted
from lecturelens.pipeline import Pipeline, RunOptions
from lecturelens.preprocess import split_sentences
from lecturelens.segment import segment
from lecturelens.summarize import Prompts, Summarizer, estimate_calls, read_usage
from lecturelens.textrank import textrank_summary
from lecturelens.utils import estimate_tokens, read_json, write_json

from . import metrics as M

log = logging.getLogger("evaluation")

ALL_SYSTEMS = ["textrank", "single_pass", "hier_speech", "lecturelens"]
LLM_SYSTEMS = {"single_pass", "hier_speech", "lecturelens"}
SYSTEM_DIRS = {"single_pass": ["single_pass"], "hier_speech": ["map_speech", "reduce_speech"],
               "lecturelens": ["map", "reduce"]}


def load_videos(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("id") and not r["id"].startswith("#")]
    return rows


def full_text(final: dict) -> str:
    parts = [final.get("concise_summary", ""), final.get("detailed_summary", ""),
             " ".join(kp["text"] for kp in final.get("key_points", [])),
             final.get("study_notes_markdown", ""),
             " ".join(f"{g['term']}: {g.get('definition', '')}" for g in final.get("glossary", []))]
    return "\n".join(p for p in parts if p)


class EvalRunner:
    def __init__(self, cfg, out_dir: Path, provider: str | None, systems: list[str], max_calls: int | None):
        self.cfg = cfg
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "outputs").mkdir(exist_ok=True)
        self.pipe = Pipeline(cfg, progress=lambda m: print(f"   {m}", flush=True),
                             on_wait=lambda s, r: print(f"   ⏳ {s:.0f}s ({r})", flush=True))
        self.state = read_json(self.out / "eval_state.json") or {"done": {}, "timings": {}, "run_ids": {}}
        router = self.pipe.router(None)
        chain = router.available_chain()
        self.provider = provider or self.state.get("provider") or (chain[0] if chain else None)
        self.state["provider"] = self.provider
        self.systems = systems
        self.max_calls = max_calls
        self.calls_today = 0

    def save(self):
        write_json(self.out / "eval_state.json", self.state)

    def router(self, run_id):
        return self.pipe.router(run_id, [self.provider]) if self.provider else None

    # ------------------------------------------------------------ estimates
    def estimate(self, ctx, system: str) -> int:
        if system not in LLM_SYSTEMS or not self.provider:
            return 0
        final = ctx.run_dir / SYSTEM_DIRS[system][-1] / "final.json"
        if final.exists():
            return 0
        if system == "single_pass":
            return 1
        n = estimate_tokens(ctx.sentences.get("text", ""))
        if system == "lecturelens":
            n += sum(estimate_tokens(s.get("text", "")) for s in ctx.slides.get("slides", []))
        return estimate_calls(n, self.cfg, self.provider, Prompts(self.cfg.path("prompts_dir")))["calls"]

    # ------------------------------------------------------------- systems
    def run_system(self, vid: str, ctx, system: str) -> dict:
        sents = ctx.sentences["sentences"]
        if system == "textrank":
            texts = [s["text"] for s in sents]
            ratio_words = int(self.cfg.get("evaluation.textrank_words", 220))
            concise = " ".join(texts[i] for i in textrank_summary(texts, ratio_words))
            detailed = " ".join(texts[i] for i in textrank_summary(texts, ratio_words * 4))
            return {"concise_summary": concise, "detailed_summary": detailed, "key_points": []}
        router = self.router(ctx.run_id)
        log_ = {"stages": {}}
        if system in ("lecturelens", "hier_speech"):
            final, _ = self.pipe.summarize_variant(ctx, system, router, log_, self.provider)
            return final
        segs = segment(ctx.run_dir, ctx.sentences, {"visual_mode": "low", "slides": []}, self.cfg,
                       ctx.source.get("chapters") or [], False, "segments_speech.json")
        summ = Summarizer(self.cfg, router, ctx.run_dir, self.pipe.progress, self.provider)
        return summ.run_single_pass(segs["segments"], sents, ctx.source.get("title", vid),
                                    float(ctx.source.get("duration") or 0)).model_dump()

    # ------------------------------------------------------------------ run
    def run(self, videos: list[dict]) -> bool:
        """Returns True when the whole grid is complete."""
        ctxs = {}
        print(f"\n== Phase A: local preparation (download / transcript / OCR) for {len(videos)} videos ==")
        for v in videos:
            try:
                ctx = self.pipe.prepare(v["url_or_path"], RunOptions(use_slides=True),
                                        run_id=self.state["run_ids"].get(v["id"]))
            except Exception as e:
                print(f"   ✗ {v['id']}: {type(e).__name__}: {e}")
                continue
            self.state["run_ids"][v["id"]] = ctx.run_id
            ctxs[v["id"]] = ctx
            self.save()

        todo = [(vid, s) for vid in ctxs for s in self.systems if f"{vid}|{s}" not in self.state["done"]]
        est = {(vid, s): self.estimate(ctxs[vid], s) for vid, s in todo}
        total = sum(est.values())
        print(f"\n== Phase B: {len(todo)} system runs pending, ~{total} LLM calls estimated (provider: {self.provider or 'none'}) ==")
        if self.provider:
            p = self.pipe.router(None, [self.provider]).provider(self.provider)
            rem = self.pipe.quota.remaining(p.name, p.model)["requests_day"]
            print(f"   remaining today on {p.name}/{p.model}: {'unlimited' if rem is None else rem}"
                  + (f"; --max-calls {self.max_calls}" if self.max_calls else ""))
            if rem is not None and total > rem:
                print(f"   The grid needs more calls than are left today → running what fits; "
                      f"continue tomorrow with: python -m lecturelens eval --resume")
        else:
            print("   No LLM provider available: only the TextRank baseline will be evaluated.")

        complete = True
        for vid, system in todo:
            if system in LLM_SYSTEMS and not self.provider:
                complete = False
                continue
            need = est[(vid, system)]
            if need and self.max_calls is not None and self.calls_today + need > self.max_calls:
                print(f"   skip {vid}/{system}: would exceed --max-calls ({self.calls_today}+{need}>{self.max_calls})")
                complete = False
                continue
            if need and self.provider:
                p = self.pipe.router(None, [self.provider]).provider(self.provider)
                if not self.pipe.quota.can_afford(p.name, p.model, need):
                    print(f"   defer {vid}/{system}: not enough quota left today for ~{need} calls")
                    complete = False
                    continue
            print(f"-- {vid} / {system}")
            t0 = time.time()
            try:
                final = self.run_system(vid, ctxs[vid], system)
            except AllProvidersExhausted as e:
                print(f"   ⏸ quota exhausted ({e}); progress saved. Resume with: python -m lecturelens eval --resume")
                complete = False
                break
            except Exception as e:
                log.exception("system failed")
                print(f"   ✗ {vid}/{system} failed: {type(e).__name__}: {e}")
                complete = False
                continue
            secs = round(time.time() - t0, 2)
            usage = read_usage(*[ctxs[vid].run_dir / d for d in SYSTEM_DIRS.get(system, [])])
            self.calls_today += need
            write_json(self.out / "outputs" / f"{vid}__{system}.json", final)
            timing = self.state["timings"].get(f"{vid}|{system}")
            if timing is None or usage["calls"] and timing < 1:
                self.state["timings"][f"{vid}|{system}"] = secs
            self.state["done"][f"{vid}|{system}"] = {"usage": usage, "finished": dt.datetime.now().isoformat(timespec="seconds")}
            self.save()
        self.state["complete"] = complete
        self.save()
        self.ctxs = ctxs
        return complete


# ------------------------------------------------------------------ metrics
def compute_metrics(runner: EvalRunner, videos: list[dict], data_dir: Path, wer_on: bool = False) -> list[dict]:
    from lecturelens.verify import build_evidence, verify_claims

    cfg = runner.cfg
    emb_model = cfg.get("segment.embed_model")
    rows, bs_jobs = [], []
    for v in videos:
        vid = v["id"]
        ctx = runner.ctxs.get(vid)
        if ctx is None:
            continue
        ref_f = data_dir / "references" / f"{vid}.md"
        con_f = data_dir / "references" / f"{vid}_concepts.txt"
        ref = ref_f.read_text(encoding="utf-8").strip() if ref_f.exists() else None
        if ref and "PLACEHOLDER" in ref[:300]:
            placeholder = True
        else:
            placeholder = False
        concepts = [l.strip() for l in con_f.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")] if con_f.exists() else []
        sents = ctx.sentences["sentences"]
        transcript = ctx.sentences.get("text", "")
        slides = ctx.slides.get("slides", [])
        slide_terms = M.slide_only_terms(slides, transcript)
        segs = (read_json(ctx.run_dir / "segments.json") or {}).get("segments", [])
        evidence = build_evidence(sents, slides, segs, int(cfg.get("verify.passage_sentences", 2)))
        run_log = read_json(ctx.run_dir / "run_log.json") or {}
        duration_min = float(ctx.source.get("duration") or 0) / 60
        wer_val = None
        if wer_on:
            wer_val = compute_wer(ctx, cfg)

        for system in runner.systems:
            key = f"{vid}|{system}"
            out_f = runner.out / "outputs" / f"{vid}__{system}.json"
            if key not in runner.state["done"] or not out_f.exists():
                continue
            final = read_json(out_f)
            summary = final.get("concise_summary", "")
            full = full_text(final)
            usage = runner.state["done"][key].get("usage", {})
            secs = runner.state["timings"].get(key)
            if system == "lecturelens" and run_log.get("stages"):  # first-run timing if the CLI ran it
                st = run_log["stages"]
                t = sum((st.get(k) or {}).get("seconds") or 0 for k in ("map[lecturelens]", "reduce[lecturelens]"))
                if t and (secs is None or secs < 1):
                    secs = t
            row = {"video_id": vid, "system": system, "subject": v.get("subject"), "duration_min": round(duration_min, 1),
                   "has_slides": v.get("has_slides"), "visual_mode": ctx.slides.get("visual_mode"),
                   "has_reference": bool(ref) and not placeholder, "reference_is_placeholder": placeholder,
                   "summary_words": len(summary.split()),
                   "compression": M.compression_ratio(transcript, summary),
                   "llm_calls": usage.get("calls", 0), "input_tokens": usage.get("input_tokens", 0),
                   "output_tokens": usage.get("output_tokens", 0), "llm_time_s": secs,
                   "slide_only_terms": len(slide_terms), "asr_wer": wer_val}
            if ref:
                row.update(M.rouge(summary, ref, bool(cfg.get("evaluation.rouge_stemmer", True))))
                bs_jobs.append((len(rows), summary, ref))
            if concepts:
                row["concept_coverage"] = M.concept_coverage(summary, concepts, emb_model, float(cfg.get("evaluation.concept_sim_threshold", 0.7)))["coverage"]
                row["concept_coverage_full"] = M.concept_coverage(full, concepts, emb_model, float(cfg.get("evaluation.concept_sim_threshold", 0.7)))["coverage"]
            row["slide_concept_recall"] = M.slide_concept_recall(summary, slide_terms)
            row["slide_concept_recall_full"] = M.slide_concept_recall(full, slide_terms)
            claims = [{"text": s} for s in split_sentences(summary) if len(s.split()) >= 4]
            ver = verify_claims(claims, evidence, cfg, sents=sents)
            row["faithfulness"] = round(sum(r["status"] == "supported" for r in ver) / len(ver), 4) if ver else None
            if final.get("key_points"):
                kv = verify_claims([{"text": k["text"], "segment_ids": k.get("segment_ids", [])} for k in final["key_points"]], evidence, cfg, sents=sents)
                row["faithfulness_keypoints"] = round(sum(r["status"] == "supported" for r in kv) / len(kv), 4)
            p = runner.provider
            if p and system in LLM_SYSTEMS:
                pr = runner.pipe.router(None, [p]).provider(p)
                lim = runner.pipe.quota.limits(pr.name, pr.model)
                row["daily_request_limit"] = lim.rpd
                row["within_free_tier"] = (lim.rpd is None or usage.get("calls", 0) <= lim.rpd)
            rows.append(row)
    if bs_jobs:
        print(f"   BERTScore on {len(bs_jobs)} summaries ({cfg.get('evaluation.bertscore_model')}, CPU)")
        scores = M.bertscore([j[1] for j in bs_jobs], [j[2] for j in bs_jobs], cfg.get("evaluation.bertscore_model"),
                             cfg.get("evaluation.bertscore_layers"))
        for (i, _, _), s in zip(bs_jobs, scores):
            rows[i].update(s)
    return rows


def compute_wer(ctx, cfg) -> float | None:
    """WER of ASR vs human captions (only when manual captions exist)."""
    if ctx.source.get("subtitle_kind") != "manual" or not ctx.source.get("audio_path"):
        return None
    from lecturelens.transcribe import transcribe

    asr_dir = ctx.run_dir / "wer_asr"
    asr_dir.mkdir(exist_ok=True)
    src = {**ctx.source, "subtitle_path": None}
    asr = transcribe(asr_dir, src, cfg.with_overrides({"transcript": {"order": ["groq", "local"]}}))
    ref = " ".join(s["text"] for s in ctx.transcript["segments"])
    hyp = " ".join(s["text"] for s in asr["segments"])
    return round(M.wer(ref, hyp), 4)


def write_human_eval(runner: EvalRunner, out: Path, seed: int = 13) -> None:
    rng = random.Random(seed)
    items = []
    for key in runner.state["done"]:
        vid, system = key.split("|")
        f = runner.out / "outputs" / f"{vid}__{system}.json"
        if f.exists():
            items.append((vid, system, read_json(f).get("concise_summary", "")))
    rng.shuffle(items)
    with open(out / "human_eval_form.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "video_id", "summary", "relevance_1to5", "coherence_1to5",
                    "factual_consistency_1to5", "coverage_1to5", "comments"])
        for i, (vid, _s, text) in enumerate(items):
            w.writerow([f"S{i:03d}", vid, text, "", "", "", "", ""])
    with open(out / "human_eval_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "system"])
        for i, (_vid, s, _t) in enumerate(items):
            w.writerow([f"S{i:03d}", s])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lecturelens eval")
    ap.add_argument("--videos", default=None, help="CSV (default data/videos.csv)")
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--provider", default=None, help="LLM used by ALL LLM systems (default: first available)")
    ap.add_argument("--max-calls", type=int, default=None, help="cap LLM calls in this invocation")
    ap.add_argument("--resume", nargs="?", const="latest", default=None, help="continue a results dir (default: latest)")
    ap.add_argument("--wer", action="store_true", help="also run ASR on videos with human captions and report WER")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    results_root = cfg.path("results_dir")
    latest = results_root / "latest.txt"
    if args.resume:
        name = latest.read_text().strip() if args.resume == "latest" and latest.exists() else args.resume
        out = results_root / name if not Path(name).is_absolute() else Path(name)
        if not out.exists():
            print(f"No results dir to resume ({out}).")
            return 2
    else:
        out = results_root / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    latest.write_text(out.name)

    videos = load_videos(Path(args.videos) if args.videos else cfg.path("data_dir") / "videos.csv")
    systems = [s for s in args.systems.split(",") if s in ALL_SYSTEMS]
    runner = EvalRunner(cfg, out, args.provider, systems, args.max_calls)
    print(f"Evaluation → {out}\n  videos: {', '.join(v['id'] for v in videos)}\n  systems: {', '.join(systems)}\n"
          f"  LLM (all LLM systems): {runner.provider or 'none available'}")
    complete = runner.run(videos)

    print("\n== Phase C: metrics (local) ==")
    rows = compute_metrics(runner, videos, cfg.path("data_dir"), args.wer)
    from .report import write_report

    write_report(out, rows, runner, cfg, complete)
    write_human_eval(runner, out)
    print(f"\nResults: {out}")
    for f in ("results.csv", "summary_table.md", "summary_table.tex", "report.md", "human_eval_form.csv"):
        print(f"  {out / f}")
    if not complete:
        print("\n⏸ Grid not complete (quota/limits). Continue later with:  python -m lecturelens eval --resume")
    return 0


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
