"""Stage 4 — Preprocess: filler removal, repetition-loop collapse, sentence
splitting with a sentence -> timestamp mapping (word times are interpolated
inside each transcript segment)."""

from __future__ import annotations

import logging
import re

from .utils import read_json, write_json

log = logging.getLogger(__name__)

MAX_SENT_WORDS = 45   # unpunctuated captions: force-split long "sentences"
_PUNKT_OK: bool | None = None


def _punkt_ready() -> bool:
    global _PUNKT_OK
    if _PUNKT_OK is None:
        try:
            import nltk

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                nltk.data.find("tokenizers/punkt_tab")
            _PUNKT_OK = True
        except Exception:
            log.info("nltk punkt unavailable; using regex sentence splitter")
            _PUNKT_OK = False
    return _PUNKT_OK


def split_sentences(text: str) -> list[str]:
    if _punkt_ready():
        from nltk.tokenize import sent_tokenize

        return [s for s in sent_tokenize(text) if s.strip()]
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if s.strip()]


def remove_fillers(text: str, fillers: list[str]) -> str:
    if not fillers:
        return text
    alts = "|".join(re.escape(f) for f in sorted(fillers, key=len, reverse=True))
    t = re.sub(rf"(?<![\w'-])(?:{alts})(?![\w'-])[,.]?", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def collapse_repeats(text: str, max_ngram: int = 4) -> str:
    """'the the the' -> 'the'; 'so so so' -> 'so'; Whisper phrase loops -> one copy."""
    t = text
    for n in range(max_ngram, 0, -1):
        pat = re.compile(rf"\b((?:[\w'-]+[,]?\s+){{{n - 1}}}[\w'-]+)(?:[,.]?\s+\1\b)+", re.I)
        t = pat.sub(r"\1", t)
    return t


def collapse_repeated_sentences(sents: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in sents:
        key = re.sub(r"\W+", " ", s["text"].lower()).strip()
        if out and key and key == re.sub(r"\W+", " ", out[-1]["text"].lower()).strip():
            out[-1]["end"] = s["end"]
            continue
        out.append(s)
    return out


def clean_text(text: str, cfg) -> str:
    t = remove_fillers(text, cfg.get("preprocess.fillers", []) or [])
    t = collapse_repeats(t, int(cfg.get("preprocess.max_repeat_ngram", 4)))
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?])\1+", r"\1", t)
    return t.strip()


def preprocess(run_dir, transcript: dict, cfg) -> dict:
    out_f = run_dir / "sentences.json"
    cached = read_json(out_f)
    if cached:
        return cached

    words: list[tuple[str, float, float]] = []  # (word, start, end)
    for seg in transcript["segments"]:
        text = clean_text(seg["text"], cfg)
        ws = text.split()
        if not ws:
            continue
        dur = max(0.0, seg["end"] - seg["start"])
        step = dur / len(ws)
        for i, w in enumerate(ws):
            words.append((w, seg["start"] + i * step, seg["start"] + (i + 1) * step))
    if not words:
        result = {"sentences": [], "text": ""}
        write_json(out_f, result)
        return result

    full = " ".join(w for w, _, _ in words)
    raw_sents = split_sentences(full)
    sents, idx = [], 0
    for s in raw_sents:
        n = len(s.split())
        chunk = words[idx: idx + n]
        idx += n
        if not chunk:
            continue
        # force-split very long unpunctuated runs (auto captions)
        pieces = [chunk[j: j + MAX_SENT_WORDS] for j in range(0, len(chunk), MAX_SENT_WORDS)]
        if len(pieces) > 1 and len(pieces[-1]) < 8:
            tail = pieces.pop()
            pieces[-1] = pieces[-1] + tail
        for part in pieces:
            sents.append({"text": " ".join(w for w, _, _ in part), "start": round(part[0][1], 2),
                          "end": round(part[-1][2], 2)})
    if idx < len(words):  # tokenizer dropped something: keep it
        rest = words[idx:]
        sents.append({"text": " ".join(w for w, _, _ in rest), "start": rest[0][1], "end": rest[-1][2]})

    sents = collapse_repeated_sentences(sents)
    for i, s in enumerate(sents):
        s["id"] = i
    result = {"sentences": sents, "text": " ".join(s["text"] for s in sents)}
    write_json(out_f, result)
    return result
