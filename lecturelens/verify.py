"""Stage 10 — Grounding & fact-check (fully local, 0 API calls).

For every claim (key point / summary sentence): retrieve the top-k evidence
passages (transcript sentence windows + slide text) by MiniLM cosine
similarity — globally and inside the claim's own segments — then score
(premise=evidence, hypothesis=claim) with a small NLI cross-encoder.
supported   : best P(entailment) >= supported_threshold
unsupported : P(contradiction) >= contradiction_threshold and not supported
uncertain   : otherwise
Faithfulness = % of claims that are supported.
"""

from __future__ import annotations

import logging

import numpy as np

from .segment import embed

log = logging.getLogger(__name__)

_NLI: dict = {}


def get_nli(name: str):
    if name not in _NLI:
        from sentence_transformers import CrossEncoder

        log.info("Loading NLI cross-encoder %s (CPU)", name)
        model = CrossEncoder(name, device="cpu")
        id2label = {int(k): v.lower() for k, v in model.model.config.id2label.items()}
        _NLI[name] = (model, id2label)
    return _NLI[name]


def nli_scores(pairs: list[tuple[str, str]], model_name: str) -> list[dict]:
    if not pairs:
        return []
    model, id2label = get_nli(model_name)
    logits = np.asarray(model.predict(pairs, batch_size=32, show_progress_bar=False, apply_softmax=False))
    if logits.ndim == 1:
        logits = logits[None, :]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return [{id2label[j]: float(p[j]) for j in range(p.shape[0])} for p in probs]


def build_evidence(sents: list[dict], slides: list[dict], segments: list[dict], window: int = 2) -> list[dict]:
    """Passages: sliding windows of `window` transcript sentences + each slide's text."""
    sent_seg = {sid: seg["segment_id"] for seg in segments for sid in seg["sentence_ids"]}
    slide_seg = {}
    for seg in segments:
        for sl in seg["slide_ids"]:
            slide_seg.setdefault(sl, seg["segment_id"])
    ev = []
    for i in range(len(sents)):
        chunk = sents[i: i + window]
        ev.append({"text": " ".join(s["text"] for s in chunk), "source": "speech", "time": sents[i]["start"],
                   "segment_id": sent_seg.get(sents[i]["id"]), "sent_idx": i})
    for sl in slides:
        text = sl.get("text", "").replace("\n", ". ")
        if len(text) >= 15:
            ev.append({"text": text, "source": "slide", "time": sl["start"], "segment_id": slide_seg.get(sl["slide_id"]),
                       "slide_id": sl["slide_id"]})
    return ev


def _window(evidence: list[dict], j: int, sents: list[dict] | None, before: int = 2, after: int = 4) -> str | None:
    """Best speech passage widened to its neighbouring sentences (abstractive claims span several)."""
    i = evidence[j].get("sent_idx")
    if i is None or not sents:
        return None
    return " ".join(s["text"] for s in sents[max(0, i - before): i + after])


def verify_claims(claims: list[dict], evidence: list[dict], cfg, ev_emb: np.ndarray | None = None,
                  sents: list[dict] | None = None) -> list[dict]:
    """claims: [{text, segment_ids?}] -> [{status, support_score, contradiction, evidence_text, evidence_source, evidence_time}]."""
    if not claims:
        return []
    if not evidence:
        return [{"status": "uncertain", "support_score": 0.0, "contradiction": 0.0, "evidence_text": None,
                 "evidence_source": None, "evidence_time": None} for _ in claims]
    emb_model = cfg.get("verify.embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    nli_model = cfg.get("verify.nli_model", "cross-encoder/nli-deberta-v3-xsmall")
    k = int(cfg.get("verify.top_k", 3))
    sup_t = float(cfg.get("verify.supported_threshold", 0.5))
    con_t = float(cfg.get("verify.contradiction_threshold", 0.6))
    if ev_emb is None:
        ev_emb = embed([e["text"] for e in evidence], emb_model)
    c_emb = embed([c["text"] for c in claims], emb_model)
    sims = c_emb @ ev_emb.T
    seg_of = np.array([e.get("segment_id") if e.get("segment_id") is not None else -1 for e in evidence])

    plans, pairs = [], []
    for ci, c in enumerate(claims):
        cand = list(np.argsort(-sims[ci])[:k])
        segs = c.get("segment_ids") or []
        if segs:
            local = np.where(np.isin(seg_of, segs))[0]
            if len(local):
                cand += [int(j) for j in local[np.argsort(-sims[ci][local])[:k]] if int(j) not in cand]
        cand = [int(j) for j in cand]
        top3 = sorted(cand, key=lambda j: -sims[ci][j])[:3]
        premises = [" ".join(evidence[j]["text"] for j in sorted(top3, key=lambda j: evidence[j]["time"]))]
        best_speech = next((j for j in top3 if evidence[j]["source"] == "speech"), None)
        best_slide = next((j for j in cand if evidence[j]["source"] == "slide"), None)
        if best_speech is not None:
            w = _window(evidence, best_speech, sents)
            if w:
                premises.append(w)
            if best_slide is not None:   # slide formula/bullet + what was said about it
                premises.append(evidence[best_slide]["text"] + " " + (w or evidence[best_speech]["text"]))
        start = len(pairs)
        pairs += [(evidence[j]["text"], c["text"]) for j in cand] + [(p, c["text"]) for p in premises]
        plans.append((cand, top3, start, len(pairs)))

    scores = nli_scores(pairs, nli_model)
    out = []
    for (cand, top3, a, b) in plans:
        sc = scores[a:b]
        ent = [s.get("entailment", 0.0) for s in sc]
        con = [s.get("contradiction", 0.0) for s in sc]
        best = int(np.argmax(ent))
        best_ev = evidence[cand[best]] if best < len(cand) else evidence[top3[0]]
        e_max = ent[best]
        # contradiction judged on the most relevant single passages only
        c_max = max(con[: len(cand)]) if cand else 0.0
        # unsupported only when contradicted AND nothing entails it reasonably
        status = "supported" if e_max >= sup_t else ("unsupported" if c_max >= con_t and e_max < 0.25 else "uncertain")
        out.append({"status": status, "support_score": round(e_max, 3), "contradiction": round(c_max, 3),
                    "evidence_text": best_ev["text"][:400] if best < len(cand) else
                    " … ".join(evidence[j]["text"] for j in top3)[:400],
                    "evidence_source": best_ev["source"], "evidence_time": best_ev["time"]})
    return out


def verify_final(final: dict, sents: list[dict], slides: list[dict], segments: list[dict], cfg) -> dict:
    """Annotate final['key_points'] in place; return the verification report."""
    ev = build_evidence(sents, slides, segments, int(cfg.get("verify.passage_sentences", 2)))
    kps = final.get("key_points", [])
    res = verify_claims([{"text": kp["text"], "segment_ids": kp.get("segment_ids", [])} for kp in kps], ev, cfg, sents=sents)
    for kp, r in zip(kps, res):
        kp.update({"status": r["status"], "support_score": r["support_score"], "evidence_text": r["evidence_text"],
                   "evidence_source": r["evidence_source"], "evidence_time": r["evidence_time"]})
    n = len(res)
    counts = {s: sum(r["status"] == s for r in res) for s in ("supported", "uncertain", "unsupported")}
    return {"n_claims": n, **counts, "faithfulness": round(counts["supported"] / n, 4) if n else None,
            "nli_model": cfg.get("verify.nli_model"), "evidence_passages": len(ev)}
