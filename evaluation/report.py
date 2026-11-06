"""Tables (Markdown + LaTeX), 300-dpi figures and an auto-written report.md."""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

from lecturelens.utils import read_json

from .metrics import mean_std

SYSTEM_LABEL = {"textrank": "TextRank (extractive)", "single_pass": "Single-pass LLM (speech)",
                "hier_speech": "Hierarchical (speech)", "lecturelens": "LectureLens (speech+slides)"}
# fixed categorical order (validated reference palette, light mode): one hue per system, never cycled
SYSTEM_COLOR = {"textrank": "#2a78d6", "single_pass": "#eb6834", "hier_speech": "#1baf7a", "lecturelens": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

TABLE_METRICS = [("rouge1", "R-1"), ("rouge2", "R-2"), ("rougeL", "R-L"), ("bertscore_f1", "BERTScore-F1"),
                 ("concept_coverage", "Concept cov."), ("slide_concept_recall", "Slide-concept rec."),
                 ("slide_concept_recall_full", "Slide-concept rec. (full)"), ("faithfulness", "Faithfulness"),
                 ("compression", "Compression"), ("llm_calls", "LLM calls"), ("llm_time_s", "LLM time (s)")]


def _fmt(m: float, s: float, pct: bool) -> str:
    if m is None or math.isnan(m):
        return "–"
    if pct:
        return f"{m * 100:.1f} ± {s * 100:.1f}"
    return f"{m:.1f} ± {s:.1f}"


def summary_table(rows: list[dict], systems: list[str]) -> tuple[str, str, dict]:
    stats = {}
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        stats[s] = {k: mean_std([r.get(k) for r in rs]) for k, _ in TABLE_METRICS}
        stats[s]["n"] = len(rs)
    pct = {k for k, _ in TABLE_METRICS if k not in ("compression", "llm_calls", "llm_time_s")}
    head = ["System", "n"] + [lbl for _, lbl in TABLE_METRICS]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = ["\\begin{table*}[t]", "\\centering", "\\small",
           "\\begin{tabular}{l" + "c" * (len(head) - 1) + "}", "\\toprule",
           " & ".join(head).replace("±", "$\\pm$") + " \\\\", "\\midrule"]
    for s in systems:
        cells = [_fmt(*stats[s][k], k in pct) for k, _ in TABLE_METRICS]
        md.append(f"| {SYSTEM_LABEL.get(s, s)} | {stats[s]['n']} | " + " | ".join(cells) + " |")
        tex.append(f"{SYSTEM_LABEL.get(s, s)} & {stats[s]['n']} & " + " & ".join(c.replace("±", "$\\pm$") for c in cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            "\\caption{Mean $\\pm$ std over videos. ROUGE/BERTScore/coverage/faithfulness in \\%. "
            "Slide-concept recall = share of slide-only terms (on slides, never spoken) present in the summary.}",
            "\\label{tab:main}", "\\end{table*}"]
    return "\n".join(md), "\n".join(tex), stats


# ----------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_groups(ax, metrics: list[tuple[str, str]], stats: dict, systems: list[str]):
    import numpy as np

    n = len(systems)
    width = 0.8 / max(1, n)
    x = np.arange(len(metrics))
    for i, s in enumerate(systems):
        means = [stats[s][k][0] * 100 if not math.isnan(stats[s][k][0]) else 0 for k, _ in metrics]
        stds = [stats[s][k][1] * 100 if not math.isnan(stats[s][k][1]) else 0 for k, _ in metrics]
        ax.bar(x + (i - (n - 1) / 2) * width, means, width * 0.92, yerr=stds, color=SYSTEM_COLOR.get(s),
               label=SYSTEM_LABEL.get(s, s), error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
    ax.set_xticks(x, [lbl for _, lbl in metrics], color=INK)
    ax.set_ylabel("score (%)", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, n), loc="upper center", bbox_to_anchor=(0.5, -0.12))


def make_figures(out: Path, rows: list[dict], stats: dict, systems: list[str], runner) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": SURFACE})
    figs = []
    fdir = out / "figures"
    fdir.mkdir(exist_ok=True)

    # 1. metric comparison by system
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _style(ax)
    _bar_groups(ax, [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore F1"),
                     ("concept_coverage", "Concept coverage")], stats, systems)
    ax.set_title("Summary quality by system (mean ± std over videos)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(fdir / "metrics_by_system.png", dpi=300)
    plt.close(fig)
    figs.append("figures/metrics_by_system.png")

    # 2. speech-only vs speech+slides (RQ1)
    pair = [s for s in ("hier_speech", "lecturelens") if s in systems]
    if len(pair) == 2:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        _style(ax)
        _bar_groups(ax, [("concept_coverage", "Concept coverage"), ("slide_concept_recall", "Slide-concept recall\n(summary)"),
                         ("slide_concept_recall_full", "Slide-concept recall\n(full notes)")], stats, pair)
        ax.set_title("RQ1 — adding slide OCR (speech-only vs speech+slides)", loc="left", color=INK)
        fig.tight_layout()
        fig.savefig(fdir / "speech_vs_slides.png", dpi=300)
        plt.close(fig)
        figs.append("figures/speech_vs_slides.png")

    # 3. faithfulness by system
    fig, ax = plt.subplots(figsize=(6, 3.2))
    _style(ax)
    _bar_groups(ax, [("faithfulness", "Summary sentences supported (NLI)")], stats, systems)
    ax.set_ylim(0, 105)
    ax.set_title("Faithfulness by system", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(fdir / "faithfulness_by_system.png", dpi=300)
    plt.close(fig)
    figs.append("figures/faithfulness_by_system.png")

    # 4. RQ3: LLM calls & time vs video length (two charts, never a dual axis)
    ll = [r for r in rows if r["system"] == "lecturelens"]
    if ll:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for ax, key, lbl in ((axes[0], "llm_calls", "LLM calls per video"), (axes[1], "llm_time_s", "LLM wall-clock (s)")):
            _style(ax)
            xs = [r["duration_min"] for r in ll]
            ys = [r.get(key) or 0 for r in ll]
            ax.scatter(xs, ys, s=42, color=SYSTEM_COLOR["lecturelens"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.set_xlabel("video length (min)", color=INK2)
            ax.set_title(lbl, loc="left", color=INK)
            if key == "llm_calls":
                ax.axhline(15, color=INK2, linestyle="--", linewidth=1)
                ax.text(max(xs + [1]), 15.4, "design budget: 15 calls / 60-min lecture", ha="right", fontsize=7.5, color=INK2)
                ax.set_ylim(0, max(ys + [16]) * 1.15)
        fig.suptitle("RQ3 — LectureLens cost vs lecture length (free tier)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
        fig.tight_layout()
        fig.savefig(fdir / "calls_time_vs_length.png", dpi=300)
        plt.close(fig)
        figs.append("figures/calls_time_vs_length.png")

    # 5. per-stage time breakdown (stacked, 2px surface gaps between segments)
    stage_keys = ["ingest", "transcribe", "preprocess", "keyframes", "ocr", "map[lecturelens]", "reduce[lecturelens]", "verify", "export"]
    stage_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#9a9a95"]
    data = {}
    for vid, ctx in (getattr(runner, "ctxs", {}) or {}).items():
        rl = read_json(ctx.run_dir / "run_log.json") or {}
        st = rl.get("stages", {})
        if st:
            data[vid] = [float((st.get(k) or {}).get("seconds") or 0) for k in stage_keys]
    if data:
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.6))
        _style(ax)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.grid(False)
        vids = list(data)
        left = [0.0] * len(vids)
        for j, (k, c) in enumerate(zip(stage_keys, stage_colors)):
            vals = [data[v][j] for v in vids]
            ax.barh(vids, vals, left=left, color=c, edgecolor=SURFACE, linewidth=2, label=k.replace("[lecturelens]", ""))
            left = [l + v for l, v in zip(left, vals)]
        ax.set_xlabel("seconds (first, uncached run)", color=INK2)
        ax.set_title("Per-stage time per video", loc="left", color=INK)
        ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        fig.tight_layout()
        fig.savefig(fdir / "stage_breakdown.png", dpi=300)
        plt.close(fig)
        figs.append("figures/stage_breakdown.png")
    return figs


# ------------------------------------------------------------------ report
def write_report(out: Path, rows: list[dict], runner, cfg, complete: bool) -> None:
    systems = runner.systems
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("video_id", "system"), k))
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md, tex, stats = summary_table(rows, systems)
    (out / "summary_table.md").write_text(md, encoding="utf-8")
    (out / "summary_table.tex").write_text(tex, encoding="utf-8")
    figs = make_figures(out, rows, stats, systems, runner) if rows else []

    provider = runner.provider
    model = runner.pipe.router(None, [provider]).provider(provider).model if provider else "none"
    n_ref = len({r["video_id"] for r in rows if r.get("has_reference")})
    n_ph = len({r["video_id"] for r in rows if r.get("reference_is_placeholder")})
    vids = sorted({r["video_id"] for r in rows})
    L = [f"# LectureLens evaluation report", "", f"*Generated {dt.datetime.now():%Y-%m-%d %H:%M} · grid "
         f"{'complete' if complete else 'INCOMPLETE (resume with `python -m lecturelens eval --resume`)'}*", "",
         "## Setup", "",
         f"- Videos: {len(vids)} ({', '.join(vids)}); human references: {n_ref}"
         + (f"; **{n_ph} placeholder reference(s) — ROUGE/BERTScore on these are NOT meaningful**" if n_ph else ""),
         f"- Systems: {', '.join(SYSTEM_LABEL.get(s, s) for s in systems)}",
         f"- LLM for every LLM-based system: `{provider}/{model}` (no fallback, temperature {cfg.get('llm.temperature')})",
         f"- Transcript: captions → Groq `{cfg.get('transcript.groq_model')}` → faster-whisper `{cfg.get('transcript.local_model')}` (local)",
         f"- OCR: {cfg.get('ocr.engine')} (CPU); embeddings: `{cfg.get('segment.embed_model')}`; NLI: `{cfg.get('verify.nli_model')}`",
         f"- BERTScore model: `{cfg.get('evaluation.bertscore_model')}` (layer {cfg.get('evaluation.bertscore_layers')}); "
         f"ROUGE with Porter stemming (ROUGE-L = summary-level ROUGE-Lsum)",
         f"- Concept coverage: stemmed exact match, else MiniLM cosine ≥ {cfg.get('evaluation.concept_sim_threshold')} against output n-grams",
         "- Slide-concept recall: terms on slide OCR never spoken in the transcript, found in the output (stemmed).",
         "- Faithfulness: share of summary sentences (≥4 words) whose best evidence passage entails them (NLI P(entail) ≥ "
         f"{cfg.get('verify.supported_threshold')}).", "",
         "## Main results", "", md, "",
         ]
    for fpath in figs:
        L += [f"![{Path(fpath).stem}]({fpath})", ""]
    # per-video cost (RQ3)
    ll = [r for r in rows if r["system"] == "lecturelens"]
    if ll:
        L += ["## Cost on the free tier (RQ3)", "", "| Video | Length (min) | Visual mode | LLM calls | Input tok | Output tok | LLM time (s) | Within daily limit |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sorted(ll, key=lambda r: r["duration_min"]):
            L.append(f"| {r['video_id']} | {r['duration_min']} | {r.get('visual_mode')} | {r['llm_calls']} | {r['input_tokens']} | "
                     f"{r['output_tokens']} | {r.get('llm_time_s') or '–'} | {'yes' if r.get('within_free_tier', True) else 'NO'} |")
        L.append("")
    # honest observations (computed, not hand-written)
    obs = []

    def m(s, k):
        return stats.get(s, {}).get(k, (float("nan"), 0))[0]

    for a, b, k, name in (("lecturelens", "hier_speech", "slide_concept_recall_full", "slide-concept recall (full notes)"),
                          ("lecturelens", "hier_speech", "concept_coverage", "concept coverage"),
                          ("lecturelens", "single_pass", "rougeL", "ROUGE-L"),
                          ("lecturelens", "single_pass", "bertscore_f1", "BERTScore F1"),
                          ("lecturelens", "textrank", "rougeL", "ROUGE-L"),
                          ("lecturelens", "single_pass", "faithfulness", "faithfulness")):
        va, vb = m(a, k), m(b, k)
        if not (math.isnan(va) or math.isnan(vb)):
            word = "higher" if va > vb else ("lower" if va < vb else "equal")
            obs.append(f"- {SYSTEM_LABEL[a]} vs {SYSTEM_LABEL[b]} on {name}: {va * 100:.1f} vs {vb * 100:.1f} ({word})"
                       + (" — **the baseline wins here**." if va < vb else "."))
    sp, ll_calls = m("single_pass", "llm_calls"), m("lecturelens", "llm_calls")
    if not (math.isnan(sp) or math.isnan(ll_calls)):
        obs.append(f"- Cost: single-pass uses {sp:.1f} call(s) per video vs {ll_calls:.1f} for LectureLens; the single-pass "
                   "baseline is only feasible because Gemini's long context fits a whole transcript — on a 12K-TPM provider it is not.")
    if len(vids) < 5:
        obs.append(f"- Only {len(vids)} video(s): differences are **not** statistically meaningful yet. Add videos to `data/videos.csv`.")
    if n_ph:
        obs.append("- Placeholder references were used: replace `data/references/<id>.md` / `<id>_concepts.txt` with human-written ones.")
    L += ["## Observations (auto-generated from the numbers above)", ""] + (obs or ["- Not enough data yet."]) + [""]
    L += ["## Files", "", "- `results.csv` — video × system × all metrics", "- `summary_table.md`, `summary_table.tex`",
          "- `figures/*.png` (300 dpi)", "- `outputs/<video>__<system>.json` — raw system outputs",
          "- `human_eval_form.csv` (shuffled, anonymised) + `human_eval_key.csv` (keep away from raters); analyse with "
          "`python -m evaluation.analyze_human_eval <this dir>`"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
