import numpy as np

from lecturelens.batching import pack_batches
from lecturelens.segment import (align_to_slides, merge_short, render_segment, split_long, texttiling_boundaries)
from lecturelens.utils import estimate_tokens


def _sents(n, words=10, dur=3.0):
    return [{"id": i, "start": i * dur, "end": (i + 1) * dur, "text": " ".join([f"w{i}"] * words) + "."} for i in range(n)]


def test_align_sentences_to_slide_windows():
    sents = _sents(10)  # 0..30 s
    slides = [{"slide_id": 0, "start": 0, "end": 9}, {"slide_id": 1, "start": 9, "end": 21}, {"slide_id": 2, "start": 21, "end": 30}]
    segs = align_to_slides(sents, slides)
    assert [s["sentence_ids"] for s in segs] == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]


def test_merge_short_and_split_long():
    sents = _sents(30)
    segs = [{"segment_id": 0, "start": 0, "end": 3, "slide_ids": [0], "sentence_ids": [0]},
            {"segment_id": 1, "start": 3, "end": 90, "slide_ids": [1], "sentence_ids": list(range(1, 30))}]
    merged = merge_short(segs, sents, {}, min_words=20)
    assert len(merged) == 1 and merged[0]["slide_ids"] == [0, 1]
    parts = split_long(merged, sents, max_words=100)
    assert len(parts) == 3
    assert sum(len(p["sentence_ids"]) for p in parts) == 30
    assert all(sum(len(sents[i]["text"].split()) for i in p["sentence_ids"]) <= 110 for p in parts)


def test_texttiling_finds_topic_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=16), rng.normal(size=16)
    emb = np.array([a + 0.1 * rng.normal(size=16) for _ in range(12)] + [b + 0.1 * rng.normal(size=16) for _ in range(12)])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert texttiling_boundaries(emb, window=4, k_std=0.5) == [12]


def test_render_segment_contains_slide_and_speech():
    sents = _sents(3)
    seg = {"segment_id": 7, "start": 750, "end": 910, "slide_ids": [4], "sentence_ids": [0, 1]}
    slides = {4: {"slide_id": 4, "title": "Update Rule", "text": "Update Rule\nw <- w - eta grad L"}}
    txt = render_segment(seg, sents, slides)
    assert txt.startswith("[SEGMENT 7 | 12:30–15:10]")
    assert "SLIDE: Update Rule — w <- w - eta grad L" in txt and "SPEECH: w0" in txt
    assert "SLIDE:" not in render_segment(seg, sents, slides, include_slides=False)


def test_batches_respect_token_budget_and_order():
    texts = [(i, "x" * (400 * (1 + i % 3))) for i in range(40)]  # 100–300 tokens each
    budget = 1500
    batches = pack_batches(texts, budget)
    assert [i for b in batches for i in b["segment_ids"]] == list(range(40))
    assert all(b["est_tokens"] <= budget for b in batches)
    total = sum(estimate_tokens(t) + 2 for _, t in texts)
    assert len(batches) <= -(-total // budget) + 1     # near-minimal number of calls
    assert len(pack_batches([(0, "short")], budget)) == 1
