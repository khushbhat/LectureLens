"""Evaluation metrics — all local: ROUGE, BERTScore, concept coverage,
slide-concept recall, compression, ASR WER."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

_STOP = None


def _stopwords() -> set[str]:
    global _STOP
    if _STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _STOP = set(ENGLISH_STOP_WORDS) | {"lecture", "slide", "today", "introduction", "summary", "using", "use", "uses"}
    return _STOP


@lru_cache(maxsize=100_000)
def _stem(w: str) -> str:
    from nltk.stem import PorterStemmer

    return PorterStemmer().stem(w)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())


def stems(text: str) -> list[str]:
    return [_stem(t) for t in tokens(text)]


def contains_phrase(hay_stems: list[str], phrase: str) -> bool:
    p = stems(phrase)
    if not p:
        return False
    n = len(p)
    return any(hay_stems[i:i + n] == p for i in range(len(hay_stems) - n + 1))


# ------------------------------------------------------------------ ROUGE
def rouge(pred: str, ref: str, stemmer: bool = True) -> dict:
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=stemmer)
    # rougeLsum expects newline-separated sentences
    from lecturelens.preprocess import split_sentences

    r = sc.score("\n".join(split_sentences(ref)), "\n".join(split_sentences(pred)))
    return {"rouge1": r["rouge1"].fmeasure, "rouge2": r["rouge2"].fmeasure, "rougeL": r["rougeLsum"].fmeasure}


# -------------------------------------------------------------- BERTScore
def bertscore(preds: list[str], refs: list[str], model: str = "distilbert-base-uncased", layers: int | None = 5) -> list[dict]:
    import logging

    from bert_score import score

    logging.getLogger("transformers").setLevel(logging.ERROR)
    kw = {"model_type": model, "lang": "en", "verbose": False, "batch_size": 8}
    if layers:
        kw["num_layers"] = layers
    P, R, F = score(preds, refs, **kw)
    return [{"bertscore_p": float(p), "bertscore_r": float(r), "bertscore_f1": float(f)} for p, r, f in zip(P, R, F)]


# -------------------------------------------------------- concept coverage
def _ngrams(toks: list[str], nmax: int = 4) -> list[str]:
    out = set()
    for n in range(1, nmax + 1):
        for i in range(len(toks) - n + 1):
            out.add(" ".join(toks[i:i + n]))
    return list(out)


def concept_coverage(text: str, concepts: list[str], embed_model: str, threshold: float = 0.7) -> dict:
    """Fraction of reference concepts present in `text`: exact (stemmed phrase) match,
    else max cosine similarity between the concept and the text's n-grams >= threshold."""
    concepts = [c.strip() for c in concepts if c.strip()]
    if not concepts:
        return {"coverage": None, "matched": [], "missed": []}
    hay = stems(text)
    matched, missed, pending = [], [], []
    for c in concepts:
        (matched if contains_phrase(hay, c) else pending).append(c)
    if pending:
        from lecturelens.segment import embed

        grams = _ngrams(tokens(text), 4)
        if grams:
            ge = embed(grams, embed_model)
            ce = embed(pending, embed_model)
            sims = (ce @ ge.T).max(axis=1)
            for c, s in zip(pending, sims):
                (matched if s >= threshold else missed).append(c)
        else:
            missed = pending
    return {"coverage": len(matched) / len(concepts), "matched": matched, "missed": missed}


# ------------------------------------------------------- slide-only terms
def _technical_unigram(raw: str, first_in_line: bool) -> bool:
    """Acronyms / CamelCase (RMSProp, SGD, ReLU) or capitalised mid-line words (Adam)."""
    if len(raw) < 3 or not raw.isalpha():
        return False
    inner_caps = any(c.isupper() for c in raw[1:])
    return inner_caps or (raw[0].isupper() and not first_in_line and len(raw) >= 4)


def slide_only_terms(slides: list[dict], transcript_text: str, min_conf: float = 0.5) -> list[str]:
    """Terms printed on slides but never spoken (stemmed match against the transcript).

    Candidates from confident OCR lines: maximal 2–3-word phrases of content
    words (each >= 3 letters, no stop words) not spoken as a phrase and with
    at least one unspoken word, plus technical-looking unigrams (acronyms,
    CamelCase, capitalised mid-line). This is what speech-only systems cannot
    know about — the basis of the slide-concept recall metric (RQ1).
    """
    spoken = set(stems(transcript_text))
    spoken_seq = stems(transcript_text)
    stop = _stopwords()
    terms: dict[str, None] = {}
    for sl in slides:
        lines = sl.get("lines") or [{"text": l, "conf": 1.0} for l in sl.get("text", "").splitlines()]
        for line in lines:
            if line.get("conf", 1.0) < min_conf:
                continue
            for ci, chunk in enumerate(re.split(r"[:;,.()\[\]|/•\-–—=<>^*+]+", line["text"])):
                raw = re.findall(r"[A-Za-z0-9]+", chunk)
                toks = [t.lower() for t in raw]
                for j, (r, t) in enumerate(zip(raw, toks)):
                    if t not in stop and _stem(t) not in spoken and _technical_unigram(r, ci == 0 and j == 0):
                        terms[t] = None
                ok = [t.isalpha() and len(t) >= 3 and t not in stop for t in toks]
                for n in (3, 2):
                    for i in range(len(toks) - n + 1):
                        if not all(ok[i:i + n]):
                            continue
                        g = toks[i:i + n]
                        phrase = " ".join(g)
                        if any(phrase in k for k in terms if " " in k):
                            continue
                        if not contains_phrase(spoken_seq, phrase) and any(_stem(w) not in spoken for w in g):
                            terms[phrase] = None
    phrases = [t for t in terms if " " in t]
    return [t for t in terms if " " in t or not any(t in p.split() for p in phrases)]


def slide_concept_recall(text: str, terms: list[str]) -> float | None:
    if not terms:
        return None
    hay = stems(text)
    return sum(contains_phrase(hay, t) for t in terms) / len(terms)


# ------------------------------------------------------------------- misc
def compression_ratio(source_text: str, summary: str) -> float | None:
    s = len(summary.split())
    return round(len(source_text.split()) / s, 2) if s else None


def wer(reference: str, hypothesis: str) -> float:
    import jiwer

    norm = lambda t: " ".join(tokens(t))
    return float(jiwer.wer(norm(reference), norm(hypothesis)))


def mean_std(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
