"""Stage 6 — Local OCR of slide keyframes + post-processing.

EasyOCR (CPU) by default; RapidOCR if installed and selected. Post-processing:
drop low-confidence tokens, merge boxes into lines in reading order, detect
the title (largest text in the top part), drop headers/footers/logos/page
numbers repeated on most slides, merge near-identical consecutive slides
(incl. progressive bullet reveals), and decide the video's `visual_mode`.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import Counter
from pathlib import Path

from .utils import read_json, write_json

log = logging.getLogger(__name__)

_READERS: dict = {}


# ------------------------------------------------------------------ engines
def _easyocr(langs: list[str]):
    key = ("easyocr", tuple(langs))
    if key not in _READERS:
        import easyocr

        log.info("Loading EasyOCR (CPU) — first run downloads ~100 MB of models")
        _READERS[key] = easyocr.Reader(langs, gpu=False, verbose=False)
    return _READERS[key]


def run_ocr(image_path: str | Path, cfg) -> list[dict]:
    """Return raw boxes [{text, conf, x0, y0, x1, y1}] in pixel coords."""
    engine = cfg.get("ocr.engine", "easyocr")
    boxes = []
    if engine == "rapidocr":
        try:
            from rapidocr_onnxruntime import RapidOCR

            if "rapid" not in _READERS:
                _READERS["rapid"] = RapidOCR()
            res, _ = _READERS["rapid"](str(image_path))
            for pts, text, conf in res or []:
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                boxes.append({"text": text, "conf": float(conf), "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
            return boxes
        except ImportError:
            log.warning("rapidocr_onnxruntime not installed; falling back to EasyOCR")
    reader = _easyocr(cfg.get("ocr.languages", ["en"]))
    for pts, text, conf in reader.readtext(str(image_path), paragraph=False):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        boxes.append({"text": text, "conf": float(conf), "x0": float(min(xs)), "y0": float(min(ys)),
                      "x1": float(max(xs)), "y1": float(max(ys))})
    return boxes


# ---------------------------------------------------------- post-processing
def boxes_to_lines(boxes: list[dict], min_conf: float) -> list[dict]:
    """Group boxes into lines (vertical overlap), left-to-right, top-to-bottom."""
    good = [b for b in boxes if b["conf"] >= min_conf and b["text"].strip()]
    good.sort(key=lambda b: (b["y0"] + b["y1"]) / 2)
    lines: list[list[dict]] = []
    for b in good:
        cy, h = (b["y0"] + b["y1"]) / 2, b["y1"] - b["y0"]
        if lines:
            last = lines[-1]
            ly = sum((x["y0"] + x["y1"]) / 2 for x in last) / len(last)
            lh = sum(x["y1"] - x["y0"] for x in last) / len(last)
            if abs(cy - ly) < 0.5 * max(h, lh):
                last.append(b)
                continue
        lines.append([b])
    out = []
    for ln in lines:
        ln.sort(key=lambda b: b["x0"])
        out.append({
            "text": " ".join(b["text"].strip() for b in ln),
            "conf": sum(b["conf"] for b in ln) / len(ln),
            "height": sum(b["y1"] - b["y0"] for b in ln) / len(ln),
            "y": min(b["y0"] for b in ln), "x": min(b["x0"] for b in ln),
        })
    return out


def pick_title(lines: list[dict], img_h: float) -> str:
    alpha = [l for l in lines if sum(c.isalpha() for c in l["text"]) >= 3]
    if not alpha:
        return ""
    top = [l for l in alpha if l["y"] < 0.35 * img_h] or alpha[:1]
    best = max(top, key=lambda l: (round(l["height"] / 4), -l["y"]))
    return best["text"]


_PAGE_NUM = re.compile(r"^\s*(?:page\s*)?\d{1,3}(?:\s*(?:/|of)\s*\d{1,3})?\s*$", re.I)


def _norm_line(t: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.lower())).strip()


def is_junk_line(t: str) -> bool:
    alnum = sum(c.isalnum() for c in t)
    return alnum < 2 or bool(_PAGE_NUM.match(t))


def remove_repeated_lines(slides: list[dict], ratio: float) -> set[str]:
    """Lines (normalized) appearing on >= ratio of slides: headers, footers, logos, URLs."""
    if len(slides) < 4:
        return set()
    counts = Counter()
    for s in slides:
        counts.update({_norm_line(l["text"]) for l in s["lines"]})
    return {k for k, c in counts.items() if c >= ratio * len(slides)}


def _tokens(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def near_duplicate(a: str, b: str, ratio: float) -> bool:
    """Same slide (minor OCR noise) or progressive reveal (one's words ⊂ other's)."""
    if not a or not b:
        return not a and not b
    if difflib.SequenceMatcher(None, a, b).ratio() >= ratio:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if min(len(ta), len(tb)) < 4:
        return False
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    bc = Counter(big)
    covered = sum(min(c, bc[w]) for w, c in Counter(small).items())
    return covered / len(small) >= ratio


def postprocess(raw_slides: list[dict], cfg) -> list[dict]:
    min_conf = float(cfg.get("ocr.min_confidence", 0.35))
    for s in raw_slides:
        s["lines"] = [l for l in boxes_to_lines(s.pop("boxes", []), min_conf) if not is_junk_line(l["text"])]
    repeated = remove_repeated_lines(raw_slides, float(cfg.get("ocr.header_footer_ratio", 0.6)))
    for s in raw_slides:
        s["lines"] = [l for l in s["lines"] if _norm_line(l["text"]) not in repeated]
        s["title"] = pick_title(s["lines"], s.get("img_h", 480))
        s["text"] = "\n".join(l["text"] for l in s["lines"])
        s["ocr_confidence"] = round(sum(l["conf"] for l in s["lines"]) / len(s["lines"]), 3) if s["lines"] else 0.0

    dup_ratio = float(cfg.get("ocr.near_duplicate_ratio", 0.9))
    merged: list[dict] = []
    for s in raw_slides:
        if merged and near_duplicate(merged[-1]["text"], s["text"], dup_ratio):
            prev = merged[-1]
            keep_new = len(s["text"]) >= len(prev["text"])  # later reveal is more complete
            base = s if keep_new else prev
            merged[-1] = {**base, "start": prev["start"], "end": s["end"],
                          "merged_from": prev.get("merged_from", [prev["slide_id"]]) + [s["slide_id"]]}
        else:
            merged.append(s)
    out = []
    for i, s in enumerate(merged):
        out.append({
            "slide_id": i, "start": s["start"], "end": s["end"], "title": s["title"], "text": s["text"],
            "ocr_confidence": s["ocr_confidence"], "thumbnail": s["thumbnail"], "frame": s["frame"],
            "lines": [{"text": l["text"], "conf": round(l["conf"], 3)} for l in s["lines"]],
            "merged_from": s.get("merged_from", [s["slide_id"]]),
        })
    return out


def decide_visual_mode(slides: list[dict], duration: float, cfg) -> tuple[str, dict]:
    min_chars = int(cfg.get("slides.min_chars_per_text_slide", 25))
    text_slides = [s for s in slides if len(s["text"]) >= min_chars]
    covered = sum(max(0.0, s["end"] - s["start"]) for s in text_slides)
    coverage = covered / duration if duration else 0.0
    stats = {"text_slides": len(text_slides), "text_coverage": round(coverage, 3)}
    low = len(text_slides) < int(cfg.get("slides.min_text_slides", 3)) or coverage < float(cfg.get("slides.min_text_coverage", 0.25))
    return ("low" if low else "high"), stats


def ocr_slides(run_dir: Path, keyframes: dict, duration: float, cfg, progress=None) -> dict:
    out_f = run_dir / "slides.json"
    cached = read_json(out_f)
    if cached is not None:
        return cached
    kfs = keyframes.get("keyframes", [])
    if not cfg.get("ocr.enabled", True) or not kfs:
        reason = "OCR disabled" if not cfg.get("ocr.enabled", True) else keyframes.get("reason", "no keyframes")
        result = {"visual_mode": "low", "slides": [], "reason": reason, "stats": {}}
        write_json(out_f, result)
        return result

    from PIL import Image

    raw_cache = run_dir / "ocr_raw.json"  # per-keyframe cache: resume mid-OCR
    raw = read_json(raw_cache, {})
    raw_slides = []
    for i, kf in enumerate(kfs):
        key = str(kf["slide_id"])
        img_p = run_dir / kf["frame"]
        if key not in raw:
            if kf.get("pixel_std", 99) < 6:   # blank / black frame
                raw[key] = {"boxes": [], "img_h": 480}
            else:
                with Image.open(img_p) as im:
                    h = im.size[1]
                raw[key] = {"boxes": run_ocr(img_p, cfg), "img_h": h}
            if i % 5 == 4 or i == len(kfs) - 1:
                write_json(raw_cache, raw)
            if progress:
                progress(f"OCR slide {i + 1}/{len(kfs)}")
        raw_slides.append({**kf, "boxes": [dict(b) for b in raw[key]["boxes"]], "img_h": raw[key]["img_h"]})
    write_json(raw_cache, raw)

    slides = postprocess(raw_slides, cfg)
    mode, stats = decide_visual_mode(slides, duration, cfg)
    result = {"visual_mode": mode, "slides": slides, "stats": stats, "engine": cfg.get("ocr.engine", "easyocr")}
    write_json(out_f, result)
    return result
