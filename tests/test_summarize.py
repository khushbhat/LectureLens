import json

import pytest

from lecturelens.cache import LLMCache
from lecturelens.llm import LLMRouter
from lecturelens.providers.base import LLMResult
from lecturelens.providers.mock import MockProvider
from lecturelens.quota import QuotaManager
from lecturelens.schemas import MCQ, FinalOutput, MapOutput
from lecturelens.summarize import Summarizer, estimate_calls, Prompts
from lecturelens.textrank import textrank_summary


def _data(n_seg=12, words=60):
    sents, segs, sid = [], [], 0
    for k in range(n_seg):
        ids = []
        for j in range(3):
            sents.append({"id": sid, "start": sid * 5.0, "end": sid * 5.0 + 5, "text":
                          f"Topic {k} sentence {j} explains concept number {k} in detail. " * (words // 30)})
            ids.append(sid)
            sid += 1
        segs.append({"segment_id": k, "start": ids[0] * 5.0, "end": ids[-1] * 5.0 + 5, "slide_ids": [k], "sentence_ids": ids})
    slides = {k: {"slide_id": k, "title": f"Slide {k}", "text": f"Slide {k}\nformula y = {k}x + b"} for k in range(n_seg)}
    return sents, segs, slides


def _summ(cfg, tmp_path, provider, overrides=None):
    c = cfg.with_overrides(overrides or {})
    router = LLMRouter(c, QuotaManager(c, tmp_path / "q.db"), LLMCache(tmp_path / "c.db"), order=["mock"],
                       providers={"mock": provider}, sleeper=lambda s: None)
    router.check = lambda n: (True, "ok")
    return Summarizer(c, router, tmp_path / "run", plan_provider="mock"), router


def test_map_reduce_end_to_end_and_resume_without_calls(cfg, tmp_path):
    sents, segs, slides = _data()
    mock = MockProvider()
    summ, router = _summ(cfg, tmp_path, mock)
    items = summ.run_map(segs, sents, slides)
    assert [i.segment_id for i in items] == list(range(12))
    final = summ.run_reduce(items, segs, "Lecture", 600)
    assert final.concise_summary and final.key_points and final.revision_questions.mcq
    assert all(kp.start is not None for kp in final.key_points if kp.segment_ids)  # timestamps from segment ids
    n_calls = len(mock.calls)

    # a fresh summarizer over the same run dir = resume: zero new calls
    mock2 = MockProvider()
    summ2, _ = _summ(cfg, tmp_path, mock2)
    summ2.run_reduce(summ2.run_map(segs, sents, slides), segs, "Lecture", 600)
    assert mock2.calls == [] and n_calls >= 2


def test_recursive_reduce_for_long_lectures(cfg, tmp_path):
    sents, segs, slides = _data(n_seg=40)
    mock = MockProvider()
    summ, _ = _summ(cfg, tmp_path, mock, {"llm": {"mock": {"batch_input_tokens": 2500, "reduce_input_tokens": 2200}}})
    items = summ.run_map(segs, sents, slides)
    final = summ.run_reduce(items, segs, "Long lecture", 3 * 3600)
    tasks = [c["prompt"].splitlines()[0] for c in mock.calls]
    assert tasks.count("TASK: reduce_group") >= 2 and tasks[-1] == "TASK: reduce"
    meta = json.loads((tmp_path / "run" / "reduce" / "reduce_meta.json").read_text())
    assert meta["levels"] >= 1
    covered = {i for ch in final.chapters for i in ch.segment_ids}
    assert covered and covered <= set(range(40))


def test_split_final_uses_two_calls(cfg, tmp_path):
    sents, segs, slides = _data(4)
    mock = MockProvider()
    summ, _ = _summ(cfg, tmp_path, mock, {"llm": {"mock": {"split_final": True}}})
    final = summ.run_reduce(summ.run_map(segs, sents, slides), segs, "L", 100)
    tasks = [c["prompt"].splitlines()[0] for c in mock.calls]
    assert "TASK: final_summary" in tasks and "TASK: final_notes" in tasks
    assert final.study_notes_markdown and final.concise_summary


class BrokenJSONProvider(MockProvider):
    """First map answer is damaged (`mode`): truncated, or corrupted mid-way."""

    def __init__(self, mode: str, fixed_by_repair: bool):
        super().__init__()
        self.mode, self.fixed_by_repair, self.good = mode, fixed_by_repair, None

    def complete(self, prompt, system=None, json_schema=None, temperature=0.2, max_tokens=None):
        if prompt.startswith("TASK: repair"):
            self.calls.append({"prompt": prompt})
            return LLMResult(self.good if self.fixed_by_repair else "{}", self.name, self.model, 10, 10)
        res = super().complete(prompt, system, json_schema, temperature, max_tokens)
        if prompt.startswith("TASK: map") and self.good is None:
            self.good = res.text
            bad = res.text[:-25] if self.mode == "truncated" else res.text.replace('"summary":', '"summary" ', 1)
            return LLMResult(bad, self.name, self.model, 10, 10)
        return res


@pytest.mark.parametrize("mode,fixed,expect_repair_call", [
    ("truncated", False, False),   # fixed locally for free
    ("corrupt", True, True),       # one LLM repair call fixes it
    ("corrupt", False, True),      # repair fails -> extractive stand-ins, no crash
])
def test_json_repair_paths(cfg, tmp_path, mode, fixed, expect_repair_call):
    sents, segs, slides = _data(3)
    mock = BrokenJSONProvider(mode, fixed)
    summ, _ = _summ(cfg, tmp_path, mock)
    items = summ.run_map(segs, sents, slides)
    assert any(c["prompt"].startswith("TASK: repair") for c in mock.calls) == expect_repair_call
    assert [i.segment_id for i in items] == [0, 1, 2]
    if not fixed and mode == "corrupt":
        assert summ.notes   # stand-ins are reported


def test_schema_coercion():
    m = MapOutput.model_validate({"segments": [{"segment_id": "4", "key_points": ["a point"], "concepts": ["SGD: stochastic"]}]})
    assert m.segments[0].segment_ids == [4] and m.segments[0].concepts[0].term == "SGD"
    q = MCQ(question="?", options=["A) one", "B) two", "C) three", "D) four"], answer="three")
    assert q.answer == "C" and q.options[0] == "one"
    f = FinalOutput.model_validate({"title": "t", "key_points": [{"text": "x", "segment_ids": "3", "evidence": "Both"}]})
    assert f.key_points[0].evidence == "both" and f.key_points[0].segment_ids == [3]


def test_estimate_calls_meets_60_min_budget(cfg):
    prompts = Prompts(cfg.path("prompts_dir"))
    est = estimate_calls(15000, cfg, "gemini", prompts)  # ~60-min lecture + slide text
    assert est["calls"] <= 15
    est_groq = estimate_calls(15000, cfg, "groq", prompts)
    assert est_groq["calls"] > est["calls"]


def test_textrank_extractive():
    sents = ["Gradient descent minimizes the loss function iteratively.", "Hello everyone and welcome.",
             "The learning rate controls the gradient descent step size.", "Stochastic gradient descent uses mini batches of data.",
             "Thanks for listening to this lecture today."]
    idx = textrank_summary(sents, max_words=20)
    assert idx == sorted(idx) and 0 < len(idx) <= 3
    assert all(len(sents[i].split()) >= 6 for i in idx)
