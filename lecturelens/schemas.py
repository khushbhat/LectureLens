"""Pydantic schemas for every LLM output. Validators are lenient: they coerce
common LLM deviations (strings for lists, "Slide" vs "slide", "B)" answers)
instead of rejecting the whole response."""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Evidence = Literal["slide", "speech", "both"]


def _to_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, (str, dict, int, float)):
        return [v]
    return list(v)


def norm_evidence(v: Any) -> str:
    v = str(v or "speech").lower().strip()
    if "both" in v or ("slide" in v and "speech" in v):
        return "both"
    return "slide" if "slide" in v else "speech"


def _ids(v: Any) -> list[int]:
    out = []
    for x in _to_list(v):
        try:
            out.append(int(re.sub(r"[^\d-]", "", str(x)) if not isinstance(x, int) else x))
        except ValueError:
            continue
    return out


class KeyPoint(BaseModel):
    text: str
    evidence: Evidence = "speech"

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v):
        if isinstance(v, str):
            return {"text": v}
        return v

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v):
        return norm_evidence(v)


class Concept(BaseModel):
    term: str
    definition: str = ""

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v):
        if isinstance(v, str):
            term, _, d = v.partition(":")
            return {"term": term.strip(), "definition": d.strip()}
        if isinstance(v, dict) and "term" not in v and len(v) == 1:
            (k, d), = v.items()
            return {"term": k, "definition": str(d)}
        return v

    @field_validator("definition", mode="before")
    @classmethod
    def _d(cls, v):
        return "" if v is None else str(v)


class SegmentSummary(BaseModel):
    segment_id: int
    segment_ids: list[int] = Field(default_factory=list)
    title: str = ""
    summary: str = ""
    key_points: list[KeyPoint] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    formulas_or_code: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)

    @field_validator("segment_ids", mode="before")
    @classmethod
    def _sids(cls, v):
        return _ids(v)

    @field_validator("segment_id", mode="before")
    @classmethod
    def _sid(cls, v):
        ids = _ids(v)
        if not ids:
            raise ValueError("segment_id missing")
        return ids[0]

    @field_validator("key_points", "concepts", mode="before")
    @classmethod
    def _lst(cls, v):
        return _to_list(v)

    @field_validator("formulas_or_code", "examples", mode="before")
    @classmethod
    def _strs(cls, v):
        return [x if isinstance(x, str) else str(x) for x in _to_list(v) if x]

    @model_validator(mode="after")
    def _fill_ids(self):
        if not self.segment_ids:
            self.segment_ids = [self.segment_id]
        return self


class MapOutput(BaseModel):
    segments: list[SegmentSummary]

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v):
        if isinstance(v, list):
            return {"segments": v}
        if isinstance(v, dict) and "segments" not in v and "segment_id" in v:
            return {"segments": [v]}
        return v


class FinalKeyPoint(BaseModel):
    text: str
    segment_ids: list[int] = Field(default_factory=list)
    evidence: Evidence = "speech"
    start: Optional[float] = None
    end: Optional[float] = None
    # filled by verify.py (local NLI)
    status: Optional[str] = None
    support_score: Optional[float] = None
    evidence_text: Optional[str] = None
    evidence_source: Optional[str] = None
    evidence_time: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, v):
        if isinstance(v, str):
            return {"text": v}
        return v

    @field_validator("evidence", mode="before")
    @classmethod
    def _ev(cls, v):
        return norm_evidence(v)

    @field_validator("segment_ids", mode="before")
    @classmethod
    def _sids(cls, v):
        return _ids(v)


class Chapter(BaseModel):
    title: str
    segment_ids: list[int] = Field(default_factory=list)
    start: Optional[float] = None
    end: Optional[float] = None

    @field_validator("segment_ids", mode="before")
    @classmethod
    def _sids(cls, v):
        return _ids(v)


class MCQ(BaseModel):
    question: str
    options: list[str]
    answer: str
    explanation: str = ""

    @model_validator(mode="after")
    def _norm(self):
        """Normalise answer to a letter A-D matching `options`."""
        letters = "ABCDEFGH"
        a = self.answer.strip()
        m = re.match(r"^\(?([A-Ha-h])[\).:\s]|^([A-Ha-h])$", a + " ")
        if m:
            self.answer = (m.group(1) or m.group(2)).upper()
        else:
            for i, o in enumerate(self.options):
                if a.lower() in o.lower() or o.lower().lstrip("abcdefgh).: ").startswith(a.lower()):
                    self.answer = letters[i]
                    break
        self.options = [re.sub(r"^\(?[A-Ha-h][\).:]\s*", "", o).strip() for o in self.options]
        return self


class ShortQ(BaseModel):
    question: str
    answer: str


class RevisionQuestions(BaseModel):
    mcq: list[MCQ] = Field(default_factory=list)
    short_answer: list[ShortQ] = Field(default_factory=list)


class FinalSummaryPart(BaseModel):
    title: str = ""
    concise_summary: str = ""
    detailed_summary: str = ""
    key_points: list[FinalKeyPoint] = Field(default_factory=list)
    glossary: list[Concept] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)


class FinalNotesPart(BaseModel):
    study_notes_markdown: str = ""
    revision_questions: RevisionQuestions = Field(default_factory=RevisionQuestions)


class FinalOutput(FinalSummaryPart, FinalNotesPart):
    pass


SCHEMA_HINTS = {
    "map": '{"segments": [{"segment_id": int, "title": str, "summary": str, "key_points": [{"text": str, "evidence": "slide|speech|both"}], "concepts": [{"term": str, "definition": str}], "formulas_or_code": [str], "examples": [str]}]}',
    "reduce_group": '{"segments": [{"segment_id": int, "segment_ids": [int], "title": str, "summary": str, "key_points": [{"text": str, "evidence": str}], "concepts": [{"term": str, "definition": str}], "formulas_or_code": [str], "examples": [str]}]}',
    "final": '{"title": str, "concise_summary": str, "detailed_summary": str, "key_points": [{"text": str, "segment_ids": [int], "evidence": str}], "study_notes_markdown": str, "glossary": [{"term": str, "definition": str}], "chapters": [{"title": str, "segment_ids": [int]}], "revision_questions": {"mcq": [{"question": str, "options": [str], "answer": "A|B|C|D", "explanation": str}], "short_answer": [{"question": str, "answer": str}]}}',
}
