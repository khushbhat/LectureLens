"""CLI:  python -m lecturelens {run, resume, quota, runs, models, preflight, eval}"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "sentence_transformers", "transformers",
                  "huggingface_hub", "faster_whisper", "filelock", "PIL", "matplotlib", "groq", "easyocr", "fontTools", "fpdf", "httpx2", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def _print_wait(seconds: float, reason: str) -> None:
    print(f"   ⏳ waiting {seconds:.0f}s ({reason}) — free-tier throttle, not frozen", flush=True)


def cmd_run(args, cfg) -> int:
    from .audio import FFmpegMissing
    from .ingest import IngestError
    from .pipeline import Pipeline, QuotaPaused, RunOptions
    from .transcribe import NoSpeechError

    pipe = Pipeline(cfg, progress=lambda m: print(f"   {m}", flush=True), on_wait=_print_wait)
    opts = RunOptions(use_slides=not args.no_slides, providers=args.providers.split(",") if args.providers else None,
                      verify=not args.no_verify)
    try:
        if args.command == "resume":
            res = pipe.resume(args.run_id)
        else:
            pf = pipe.preflight(args.source, opts)
            print("\nPre-flight estimate:")
            print("  " + pf["message"].replace("\n", "\n  "))
            print(f"  estimated time: ~{pf['est_time_s'] // 60} min {pf['est_time_s'] % 60} s\n")
            res = pipe.run(args.source, opts)
    except QuotaPaused as e:
        print(f"\n⏸  PAUSED: {e}")
        return 3
    except FFmpegMissing as e:
        print(f"\n✗ {e}")
        return 2
    except (IngestError, NoSpeechError, FileNotFoundError) as e:
        print(f"\n✗ {e}")
        return 2
    s = res["stats"]
    v = res.get("verification") or {}
    print("\n✓ Done:", res["final"].get("title"))
    print(f"  run folder : {pipe.runs_dir / res['run_id']}")
    for k, p in (res.get("exports") or {}).items():
        print(f"  {k:<10} : {p}")
    print(f"  transcript : {res['transcript_source']} | segmentation: {res['segmentation_mode']} | visual_mode: {res['visual_mode']}")
    print(f"  LLM calls  : {s['llm_calls']} (cache hits {s['cache_hits']}), tokens in/out {s['input_tokens']}/{s['output_tokens']}, "
          f"retries {s['retries']}, fallbacks {s['fallbacks']}, throttle wait {s['rate_limit_wait_s']}s")
    print(f"  providers  : {', '.join(s['providers']) or 'none (extractive)'}")
    if v.get("faithfulness") is not None:
        print(f"  faithful   : {v['faithfulness'] * 100:.0f}% of {v['n_claims']} key points supported "
              f"({v['uncertain']} uncertain, {v['unsupported']} unsupported)")
    print(f"  wall clock : {s['wall_clock_s']} s")
    return 0


def cmd_quota(args, cfg) -> int:
    from .llm import LLMRouter
    from .quota import QuotaManager

    q = QuotaManager(cfg)
    router = LLMRouter(cfg, q)
    pairs = []
    for name in router.order:
        try:
            p = router.provider(name)
            pairs.append((p.name, p.model))
        except Exception:
            continue
    if cfg.get("llm.groq.repair_model"):
        pairs.append(("groq", cfg.get("llm.groq.repair_model")))
    pairs.append(("groq", cfg.get("transcript.groq_model")))
    print("Keys found:", ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in cfg.keys_found().items()))
    print(f"\n{'provider/model':<40}{'req today':>10}{'limit':>8}{'left':>8}{'tokens today':>14}{'left':>10}  resets")
    for r in q.report(pairs):
        print(f"{r['provider'] + '/' + r['model']:<40}{r['requests_today']:>10}{str(r['rpd_limit'] or '∞'):>8}"
              f"{str(r['requests_left'] if r['requests_left'] is not None else '∞'):>8}{r['tokens_today']:>14}"
              f"{str(r['tokens_left'] if r['tokens_left'] is not None else '∞'):>10}  {r['resets']}"
              f"{'  [BLOCKED today]' if r['blocked'] else ''}")
    print("\nLimits come from config.yaml (quota.limits). Check your real numbers in Google AI Studio → "
          "'Rate limits' and console.groq.com/settings/limits, and edit them there.")
    return 0


def cmd_models(args, cfg) -> int:
    from .llm import LLMRouter

    router = LLMRouter(cfg)
    for row in router.status():
        print(f"{row['provider']:<8} {row['model']:<32} {'OK' if row['available'] else 'UNAVAILABLE'}  {'' if row['available'] else row['message']}")
    return 0


def cmd_runs(args, cfg) -> int:
    from .pipeline import Pipeline

    for r in Pipeline(cfg).list_runs():
        print(f"{r['status']:<8} {r['run_id']:<55} {r['duration']:>8}  {r['title']}")
        if r["status"] == "paused":
            print(f"         ↳ python -m lecturelens resume {r['run_id']}")
    return 0


def cmd_preflight(args, cfg) -> int:
    from .pipeline import Pipeline, RunOptions

    pf = Pipeline(cfg).preflight(args.source, RunOptions(use_slides=not args.no_slides))
    print(pf["message"])
    print(f"duration {pf['duration'] / 60:.1f} min, est. input ~{pf['est_input_tokens']:,} tokens, est. time ~{pf['est_time_s'] // 60} min")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if "eval" in argv[:2]:  # forward everything after "eval" untouched to the evaluation CLI
        i = argv.index("eval")
        _setup_logging("-v" in argv[:i] or "--verbose" in argv[:i])
        from evaluation.run_eval import main as eval_main

        return eval_main(argv[i + 1:])
    ap = argparse.ArgumentParser(prog="lecturelens",description="Multimodal (speech + slide) lecture summarization on free tiers")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--config", help="path to config.yaml")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="summarize a video file or YouTube URL")
    r.add_argument("source")
    r.add_argument("--no-slides", action="store_true", help="speech only (skip frames/OCR)")
    r.add_argument("--no-verify", action="store_true")
    r.add_argument("--providers", help="LLM order override, e.g. groq,ollama or mock")
    rs = sub.add_parser("resume", help="continue a paused run")
    rs.add_argument("run_id")
    rs.add_argument("--providers")
    sub.add_parser("quota", help="today's usage and remaining free quota")
    sub.add_parser("models", help="check configured models exist for your keys")
    sub.add_parser("runs", help="list runs")
    pf = sub.add_parser("preflight", help="estimate calls/tokens for a video without running")
    pf.add_argument("source")
    pf.add_argument("--no-slides", action="store_true")
    ev = sub.add_parser("eval", help="run the evaluation grid (see evaluation/run_eval.py)")
    ev.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "eval":
        from evaluation.run_eval import main as eval_main

        return eval_main(args.rest)

    from .config import load_config

    cfg = load_config(args.config)
    if args.command in ("run", "resume"):
        if args.command == "resume":
            args.no_slides = args.no_verify = False
            if args.providers:
                cfg = cfg.with_overrides({"llm": {"order": args.providers.split(",")}})
            args.providers = None
        return cmd_run(args, cfg)
    return {"quota": cmd_quota, "models": cmd_models, "runs": cmd_runs, "preflight": cmd_preflight}[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
