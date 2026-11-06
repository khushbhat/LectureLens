"""Stage 7 — Alignment & segmentation into multimodal segments.

visual_mode "high": one segment per slide (or merged run of slides) containing
every transcript sentence whose midpoint falls inside the slide's on-screen
window. Very short segments are merged, very long ones split on sentence
boundaries.

visual_mode "low" (talking head / whiteboard / no slides): speech-only topic
segmentation — embedding TextTiling with all-MiniLM-L6-v2: cosine similarity
between sentence windows left/right of each gap, depth scores, boundaries
where depth > mean + k*std (YouTube chapters are used when present).
"""

from __future__ import annotations

import bisect
import logging

import numpy as np

from .utils import fmt_ts, read_json, write_json

log = logging.getLogger(__name__)

_EMBEDDERS: dict = {}


def get_embedder(name: str):
    if name not in _EMBEDDERS:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s (CPU)", name)
        _EMBEDDERS[name] = SentenceTransformer(name, device="cpu")
    return _EMBEDDERS[name]


def embed(texts: list[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return np.asarray(get_embedder(model_name).encode(texts, batch_size=64, normalize_embeddings=True,
                                                      show_progress_bar=False))


def _words(sents: list[dict], ids: list[int]) -> int:
    return sum(len(sents[i]["text"].split()) for i in ids)


def _make(seg_id: int, sent_ids: list[int], sents: list[dict], slide_ids: list[int], start=None, end=None) -> dict:
    s0 = sents[sent_ids[0]]["start"] if sent_ids else start
    s1 = sents[sent_ids[-1]]["end"] if sent_ids else end
    return {"segment_id": seg_id, "start": float(start if start is not None else s0),
            "end": float(end if end is not None else s1), "slide_ids": slide_ids, "sentence_ids": sent_ids}


def align_to_slides(sents: list[dict], slides: list[dict]) -> list[dict]:
    """Assign each sentence to the slide on screen at its midpoint."""
    starts = [s["start"] for s in slides]
    buckets: list[list[int]] = [[] for _ in slides]
    for s in sents:
        mid = (s["start"] + s["end"]) / 2
        k = max(0, bisect.bisect_right(starts, mid) - 1)
        buckets[k].append(s["id"])
    return [{"segment_id": k, "start": sl["start"], "end": sl["end"], "slide_ids": [sl["slide_id"]],
             "sentence_ids": buckets[k]} for k, sl in enumerate(slides)]


def merge_short(segs: list[dict], sents: list[dict], slide_chars: dict[int, int], min_words: int) -> list[dict]:
    """Merge segments whose speech + slide text is tiny into the following (or previous) one."""
    def size(seg):
        return _words(sents, seg["sentence_ids"]) + sum(slide_chars.get(i, 0) for i in seg["slide_ids"]) // 6

    segs = [dict(s) for s in segs]
    i = 0
    while len(segs) > 1 and i < len(segs):
        if size(segs[i]) < min_words:
            j = i + 1 if i + 1 < len(segs) else i - 1
            a, b = (segs[i], segs[j]) if j > i else (segs[j], segs[i])
            m = {"segment_id": a["segment_id"], "start": a["start"], "end": b["end"],
                 "slide_ids": a["slide_ids"] + b["slide_ids"], "sentence_ids": a["sentence_ids"] + b["sentence_ids"]}
            lo = min(i, j)
            segs[lo: lo + 2] = [m]
            i = max(0, lo - 1) if size(m) < min_words else lo
            continue
        i += 1
    return segs


def split_long(segs: list[dict], sents: list[dict], max_words: int) -> list[dict]:
    out = []
    for seg in segs:
        ids = seg["sentence_ids"]
        if _words(sents, ids) <= max_words:
            out.append(seg)
            continue
        n_parts = int(np.ceil(_words(sents, ids) / max_words))
        target = _words(sents, ids) / n_parts
        part, count = [], 0
        parts = []
        for sid in ids:
            part.append(sid)
            count += len(sents[sid]["text"].split())
            if count >= target and len(parts) < n_parts - 1:
                parts.append(part)
                part, count = [], 0
        if part:
            parts.append(part)
        for p in parts:
            out.append({"segment_id": 0, "start": sents[p[0]]["start"], "end": sents[p[-1]]["end"],
                        "slide_ids": seg["slide_ids"], "sentence_ids": p})
        # keep the slide's real on-screen window on first/last part
        out[-len(parts)]["start"] = min(seg["start"], out[-len(parts)]["start"])
        out[-1]["end"] = max(seg["end"], out[-1]["end"])
    return out


def texttiling_boundaries(emb: np.ndarray, window: int, k_std: float) -> list[int]:
    """Gap indices g (boundary *before* sentence g) by embedding TextTiling depth scores."""
    n = len(emb)
    if n < 2 * window + 1:
        return []
    gaps = list(range(window, n - window + 1))
    sims = []
    for g in gaps:
        left, right = emb[g - window: g].mean(0), emb[g: g + window].mean(0)
        sims.append(float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9)))
    sims = np.array(sims)
    depths = np.zeros_like(sims)
    for i, s in enumerate(sims):
        # climb left/right while similarity keeps increasing (TextTiling peaks)
        l = i
        while l > 0 and sims[l - 1] >= sims[l]:
            l -= 1
        r = i
        while r < len(sims) - 1 and sims[r + 1] >= sims[r]:
            r += 1
        depths[i] = (sims[l] - s) + (sims[r] - s)
    thr = depths.mean() + k_std * depths.std()
    cands = [gaps[i] for i in range(len(sims)) if depths[i] > thr and depths[i] > 0]
    # non-maximum suppression within one window
    chosen: list[int] = []
    for g in sorted(cands, key=lambda g: -depths[gaps.index(g)]):
        if all(abs(g - c) >= window for c in chosen):
            chosen.append(g)
    return sorted(chosen)


def topic_segments(sents: list[dict], cfg, chapters: list[dict] | None = None) -> list[dict]:
    if not sents:
        return []
    bounds: list[int]
    if chapters and len(chapters) >= 3:
        starts = [c["start"] or 0 for c in chapters]
        bounds = sorted({next((s["id"] for s in sents if s["start"] >= t), len(sents)) for t in starts[1:]})
        bounds = [b for b in bounds if 0 < b < len(sents)]
        log.info("Speech-only segmentation from %d YouTube chapters", len(chapters))
    else:
        emb = embed([s["text"] for s in sents], cfg.get("segment.embed_model"))
        bounds = texttiling_boundaries(emb, int(cfg.get("segment.tiling_window_sentences", 6)),
                                       float(cfg.get("segment.tiling_depth_std", 0.5)))
        # too few topic shifts: fall back to ~target-length windows on top of them
        total = sents[-1]["end"] - sents[0]["start"]
        target = float(cfg.get("segment.tiling_target_minutes", 5)) * 60
        if len(bounds) + 1 < total / (2.5 * target):
            extra, t_next = [], sents[0]["start"] + target
            for s in sents:
                if s["start"] >= t_next:
                    extra.append(s["id"])
                    t_next = s["start"] + target
            bounds = sorted(set(bounds) | set(extra))
    cuts = [0] + bounds + [len(sents)]
    segs = []
    for a, b in zip(cuts, cuts[1:]):
        if a < b:
            segs.append(_make(0, list(range(a, b)), sents, []))
    return segs


def segment(run_dir, sentences: dict, slides: dict, cfg, chapters: list[dict] | None = None,
            use_slides: bool = True, out_name: str = "segments.json") -> dict:
    out_f = run_dir / out_name
    cached = read_json(out_f)
    if cached is not None:
        return cached
    sents = sentences["sentences"]
    slide_list = slides.get("slides", []) if use_slides else []
    mode = slides.get("visual_mode", "low") if use_slides else "low"
    min_w, max_w = int(cfg.get("segment.min_words", 60)), int(cfg.get("segment.max_words", 700))

    if mode == "high" and slide_list:
        segs = align_to_slides(sents, slide_list)
        slide_chars = {s["slide_id"]: len(s["text"]) for s in slide_list}
        segs = merge_short(segs, sents, slide_chars, min_w)
        segs = split_long(segs, sents, max_w)
        segs = [s for s in segs if s["sentence_ids"] or any(slide_chars.get(i) for i in s["slide_ids"])]
    else:
        segs = topic_segments(sents, cfg, chapters)
        segs = merge_short(segs, sents, {}, min_w)
        segs = split_long(segs, sents, max_w)
    for i, s in enumerate(segs):
        s["segment_id"] = i
        s["words"] = _words(sents, s["sentence_ids"])
    result = {"mode": "slide-aligned" if mode == "high" and slide_list else "speech-only", "segments": segs}
    write_json(out_f, result)
    log.info("Segmentation (%s): %d segments", result["mode"], len(segs))
    return result


def render_segment(seg: dict, sents: list[dict], slides_by_id: dict[int, dict], include_slides: bool = True) -> str:
    """Prompt block: `[SEGMENT k | 12:30–15:10] SLIDE: ... SPEECH: ...`."""
    head = f"[SEGMENT {seg['segment_id']} | {fmt_ts(seg['start'])}–{fmt_ts(seg['end'])}]"
    parts = [head]
    if include_slides:
        slide_txt = []
        for sid in seg["slide_ids"]:
            sl = slides_by_id.get(sid)
            if sl and sl["text"].strip():
                body = " | ".join(l for l in sl["text"].splitlines() if l.strip() and l.strip() != sl["title"])
                slide_txt.append(f"{sl['title']}" + (f" — {body}" if body else ""))
        parts.append("SLIDE: " + (" || ".join(slide_txt) if slide_txt else "(none)"))
    speech = " ".join(sents[i]["text"] for i in seg["sentence_ids"])
    parts.append("SPEECH: " + (speech or "(no speech)"))
    return "\n".join(parts)
