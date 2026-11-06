"""Extractive TextRank (local, 0 LLM calls).

Used (a) as an evaluation baseline and (b) as the last-resort output when no
LLM provider is available: summary + key points + a notes skeleton built
from slide titles / topic segments.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import fmt_ts


def textrank_rank(sentences: list[str]) -> list[float]:
    if len(sentences) < 2:
        return [1.0] * len(sentences)
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(sentences)
    sim = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(sim, 0.0)
    g = nx.from_numpy_array(sim)
    try:
        pr = nx.pagerank(g, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        pr = {i: float(sim[i].sum()) for i in range(len(sentences))}
    return [pr[i] for i in range(len(sentences))]


def textrank_summary(sentences: list[str], max_words: int = 220, min_words_per_sentence: int = 6) -> list[int]:
    """Indices (in original order) of the top-ranked sentences fitting `max_words`."""
    cand = [i for i, s in enumerate(sentences) if len(s.split()) >= min_words_per_sentence]
    if not cand:
        cand = list(range(len(sentences)))
    if not cand:
        return []
    scores = textrank_rank([sentences[i] for i in cand])
    order = [cand[j] for j in np.argsort(scores)[::-1]]
    chosen, words = [], 0
    for i in order:
        w = len(sentences[i].split())
        if words + w > max_words and chosen:
            continue
        chosen.append(i)
        words += w
        if words >= max_words * 0.9:
            break
    return sorted(chosen)


def extractive_result(sents: list[dict], segments: list[dict], slides_by_id: dict, title: str,
                      max_words: int = 220) -> dict:
    """Full FinalOutput-shaped dict without any LLM (clearly labelled)."""
    texts = [s["text"] for s in sents]
    idx = textrank_summary(texts, max_words)
    detailed_idx = textrank_summary(texts, max_words * 3)
    kp_idx = textrank_summary(texts, 400)[:12]
    sent_to_seg = {sid: seg["segment_id"] for seg in segments for sid in seg["sentence_ids"]}
    notes = [f"# {title}", "", "> Extractive notes (no LLM available): sentences selected by TextRank.", ""]
    chapters = []
    for seg in segments:
        heading = ""
        for sid in seg["slide_ids"]:
            sl = slides_by_id.get(sid)
            if sl and sl.get("title"):
                heading = sl["title"]
                break
        if not heading:
            first = sents[seg["sentence_ids"][0]]["text"] if seg["sentence_ids"] else "Section"
            heading = " ".join(first.split()[:8]) + "…"
        notes.append(f"## {heading}  ({fmt_ts(seg['start'])})")
        seg_texts = [sents[i]["text"] for i in seg["sentence_ids"]]
        for j in textrank_summary(seg_texts, 80):
            notes.append(f"- {seg_texts[j]}")
        for sid in seg["slide_ids"]:
            sl = slides_by_id.get(sid)
            if sl and sl.get("text"):
                notes.append("- *Slide:* " + " | ".join(sl["text"].splitlines()[:6]))
        notes.append("")
        chapters.append({"title": heading, "segment_ids": [seg["segment_id"]]})
    return {
        "title": title,
        "concise_summary": " ".join(texts[i] for i in idx),
        "detailed_summary": " ".join(texts[i] for i in detailed_idx),
        "key_points": [{"text": texts[i], "segment_ids": [sent_to_seg.get(i, 0)], "evidence": "speech"} for i in kp_idx],
        "study_notes_markdown": "\n".join(notes),
        "glossary": [],
        "chapters": chapters[:12],
        "revision_questions": {"mcq": [], "short_answer": []},
        "extractive_only": True,
    }
