"""Deterministic offline provider for tests and development (never touches the network).

It reads the `TASK:` header of each prompt and returns schema-valid JSON built
from the input, so the full pipeline can be exercised with zero quota.
Exceptions can be queued in `fail_queue` to test retry/fallback behaviour.
"""

from __future__ import annotations

import json
import re

from .base import LLMProvider, LLMResult
from ..utils import estimate_tokens

_SEG_RE = re.compile(r"\[SEGMENT (\d+) \| ([^\]]*)\]\n(.*?)(?=\n\[SEGMENT \d+ \||\n=== END INPUT ===|\Z)", re.S)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]


def _input_block(prompt: str) -> str:
    m = re.search(r"=== INPUT ===\n(.*?)\n=== END INPUT ===", prompt, re.S)
    return m.group(1) if m else ""


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, model: str = "mock-1", max_output_tokens: int = 4096, **_):
        super().__init__(model, max_output_tokens)
        self.calls: list[dict] = []
        self.fail_queue: list[Exception] = []

    def complete(self, prompt, system=None, json_schema=None, temperature=0.2, max_tokens=None) -> LLMResult:
        self.calls.append({"prompt": prompt, "system": system})
        if self.fail_queue:
            raise self.fail_queue.pop(0)
        task = (re.search(r"^TASK:\s*(\S+)", prompt, re.M) or [None, "unknown"])[1]
        handler = getattr(self, f"_task_{task}", self._task_unknown)
        out = json.dumps(handler(prompt), ensure_ascii=False)
        return LLMResult(out, self.name, self.model, estimate_tokens(prompt), estimate_tokens(out))

    # ---------------------------------------------------------------- tasks
    def _segments(self, prompt: str) -> list[dict]:
        items = []
        for sid, _span, body in _SEG_RE.findall(_input_block(prompt) or prompt):
            slide = re.search(r"SLIDE:\s*(.*?)(?=\nSPEECH:|\Z)", body, re.S)
            speech = re.search(r"SPEECH:\s*(.*)", body, re.S)
            slide_t = slide.group(1).strip() if slide else ""
            speech_t = speech.group(1).strip() if speech else body.strip()
            sents = _sentences(speech_t)
            has_slide = bool(slide_t) and slide_t.lower() != "(none)"
            title = slide_t.splitlines()[0][:60] if has_slide else " ".join(speech_t.split()[:6])
            items.append({
                "segment_id": int(sid),
                "title": title or f"Segment {sid}",
                "summary": " ".join(sents[:2]) or speech_t[:200],
                "key_points": [{"text": s, "evidence": "both" if has_slide else "speech"} for s in sents[:3]],
                "concepts": [{"term": title.split(":")[0][:40] or f"Topic {sid}", "definition": sents[0] if sents else ""}],
                "formulas_or_code": [],
                "examples": [],
            })
        return items

    def _task_map(self, prompt):
        return {"segments": self._segments(prompt)}

    def _items(self, prompt) -> list[dict]:
        try:
            data = json.loads(_input_block(prompt))
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else data.get("segments", [])

    def _task_reduce_group(self, prompt):
        items = self._items(prompt)
        ids = sorted({i for it in items for i in (it.get("segment_ids") or [it.get("segment_id")]) if i is not None})
        kps = [kp for it in items for kp in it.get("key_points", [])][:6]
        return {"segments": [{
            "segment_id": ids[0] if ids else 0, "segment_ids": ids,
            "title": " / ".join(it.get("title", "") for it in items[:2]),
            "summary": " ".join(it.get("summary", "") for it in items),
            "key_points": kps, "concepts": [c for it in items for c in it.get("concepts", [])][:8],
            "formulas_or_code": [f for it in items for f in it.get("formulas_or_code", [])],
            "examples": [e for it in items for e in it.get("examples", [])],
        }]}

    def _final_summary(self, items):
        kps, chapters = [], []
        for it in items:
            ids = it.get("segment_ids") or [it.get("segment_id", 0)]
            for kp in it.get("key_points", [])[:2]:
                kps.append({"text": kp["text"], "segment_ids": ids, "evidence": kp.get("evidence", "speech")})
            chapters.append({"title": it.get("title", "Part"), "segment_ids": ids})
        summary = " ".join(it.get("summary", "") for it in items)
        return {
            "title": items[0].get("title", "Lecture") if items else "Lecture",
            "concise_summary": summary[:1200],
            "detailed_summary": summary,
            "key_points": kps,
            "glossary": [c for it in items for c in it.get("concepts", [])],
            "chapters": chapters,
        }

    def _final_notes(self, items):
        notes = "\n\n".join(f"## {it.get('title','Topic')}\n\n" + "\n".join(f"- {kp['text']}" for kp in it.get("key_points", []))
                            for it in items)
        kps = [kp["text"] for it in items for kp in it.get("key_points", [])] or ["The lecture covers its topic."]
        mcq = [{"question": f"Which statement was made in the lecture? ({i+1})",
                "options": [kps[i % len(kps)], "None of these", "All of these", "Not discussed"],
                "answer": "A", "explanation": "Stated in the lecture."} for i in range(5)]
        short = [{"question": f"Explain: {kps[i % len(kps)][:80]}", "answer": kps[i % len(kps)]} for i in range(5)]
        return {"study_notes_markdown": notes, "revision_questions": {"mcq": mcq, "short_answer": short}}

    def _task_reduce(self, prompt):
        items = self._items(prompt)
        return {**self._final_summary(items), **self._final_notes(items)}

    def _task_final_summary(self, prompt):
        return self._final_summary(self._items(prompt))

    def _task_final_notes(self, prompt):
        return self._final_notes(self._items(prompt))

    def _task_single_pass(self, prompt):
        items = self._segments(prompt)
        return {**self._final_summary(items), **self._final_notes(items)}

    def _task_repair(self, prompt):
        block = _input_block(prompt)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            return {}

    def _task_unknown(self, prompt):
        return {"text": prompt[:200]}
