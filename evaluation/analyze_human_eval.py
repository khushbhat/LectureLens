"""Analyse human ratings: per-system means and Krippendorff's alpha (ordinal).

Each rater fills a copy of `human_eval_form.csv` named `human_eval_<rater>.csv`
in the results dir. Then:

    python -m evaluation.analyze_human_eval results/<timestamp>
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CRITERIA = ["relevance_1to5", "coherence_1to5", "factual_consistency_1to5", "coverage_1to5"]


def load(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    d = Path(argv[0]) if argv else Path("results") / (Path("results/latest.txt").read_text().strip())
    key = {r["item_id"]: r["system"] for r in load(d / "human_eval_key.csv").values()}
    raters = sorted(p for p in d.glob("human_eval_*.csv") if p.name not in ("human_eval_form.csv", "human_eval_key.csv"))
    if not raters:
        print(f"No rater files (human_eval_<name>.csv) in {d}")
        return 1
    sheets = [load(p) for p in raters]
    items = sorted(key)
    lines = [f"# Human evaluation ({len(raters)} raters: {', '.join(p.stem.replace('human_eval_', '') for p in raters)})", "",
             "| System | " + " | ".join(c.replace("_1to5", "") for c in CRITERIA) + " |", "|---|" + "---|" * len(CRITERIA)]
    per_sys = defaultdict(lambda: defaultdict(list))
    for sh in sheets:
        for it in items:
            for c in CRITERIA:
                v = (sh.get(it, {}).get(c) or "").strip()
                if v:
                    per_sys[key[it]][c].append(float(v))
    for s, crit in sorted(per_sys.items()):
        lines.append(f"| {s} | " + " | ".join(f"{np.mean(crit[c]):.2f} ± {np.std(crit[c]):.2f}" if crit[c] else "–" for c in CRITERIA) + " |")
    lines += ["", "Krippendorff's alpha (ordinal):", ""]
    if len(sheets) >= 2:
        import krippendorff

        for c in CRITERIA:
            mat = np.array([[float(sh[it][c]) if sh.get(it, {}).get(c, "").strip() else np.nan for it in items] for sh in sheets])
            try:
                a = krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal")
                lines.append(f"- {c.replace('_1to5', '')}: α = {a:.3f}")
            except Exception as e:
                lines.append(f"- {c}: n/a ({e})")
    else:
        lines.append("- needs ≥ 2 raters")
    out = d / "human_eval_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
