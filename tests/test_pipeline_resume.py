"""Pause on quota exhaustion mid-run, then resume without repeating any completed call."""

import pytest

from lecturelens.pipeline import Pipeline, QuotaPaused, RunContext, RunOptions
from lecturelens.providers.mock import MockProvider
from lecturelens.utils import read_json


def _fake_prepare(n_seg):
    def prepare(self, source, options, run_id=None, info=None, log_=None):
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        sents, slides = [], []
        for k in range(n_seg):
            for j in range(3):
                i = len(sents)
                sents.append({"id": i, "start": i * 10.0, "end": i * 10.0 + 10,
                              "text": f"In part {k} we explain idea {k} step {j} with enough words to matter here. " * 3})
            slides.append({"slide_id": k, "start": k * 30.0, "end": k * 30.0 + 30, "title": f"Topic {k}",
                           "text": f"Topic {k}\nkey formula f{k}(x) = {k}x", "thumbnail": "", "lines": []})
        ctx = RunContext(run_id, run_dir, source={"kind": "file", "title": "Fake lecture", "duration": n_seg * 30.0},
                         transcript={"source": "test"}, sentences={"sentences": sents, "text": " ".join(s["text"] for s in sents)},
                         slides={"visual_mode": "high", "slides": slides})
        return ctx
    return prepare


def test_pause_then_resume_reuses_completed_calls(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr("lecturelens.pipeline.check_ffmpeg", lambda: None)
    monkeypatch.setattr(Pipeline, "prepare", _fake_prepare(20))
    small = cfg.with_overrides({"llm": {"mock": {"batch_input_tokens": 1500}}, "verify": {"enabled": False},
                                "export": {"pdf": False},
                                "quota": {"limits": {"mock/mock-1": {"rpd": 3}}}})
    calls_seen = []
    real_build = __import__("lecturelens.providers", fromlist=["build_provider"]).build_provider

    def build(name, c):
        p = real_build(name, c)
        if isinstance(p, MockProvider):
            calls_seen.append(p)
        return p
    monkeypatch.setattr("lecturelens.llm.build_provider", build)

    pipe = Pipeline(small, sleeper=lambda s: None)
    with pytest.raises(QuotaPaused) as ei:
        pipe.run("fake.mp4", RunOptions(providers=["mock"]), run_id="fake_run")
    run_dir = small.path("runs_dir") / "fake_run"
    state = read_json(run_dir / "state.json")
    assert state["status"] == "paused" and "lecturelens resume fake_run" in str(ei.value)
    first_calls = sum(len(p.calls) for p in calls_seen)
    assert first_calls == 3                      # used exactly the daily cap, then paused
    done_batches = len(list((run_dir / "map").glob("batch_*.json")))
    assert done_batches == 3

    # next day: quota available again -> resume continues where it stopped
    big = small.with_overrides({"quota": {"limits": {"mock/mock-1": {"rpd": 1000}}}})
    calls_seen.clear()
    pipe2 = Pipeline(big, sleeper=lambda s: None)
    res = pipe2.resume("fake_run")
    assert res["status"] == "done" and res["final"]["concise_summary"]
    total_batches = len(read_json(run_dir / "map" / "batches.json"))
    resumed_calls = sum(len(p.calls) for p in calls_seen)
    assert resumed_calls == (total_batches - 3) + 1   # remaining map batches + 1 reduce, nothing repeated
    assert read_json(run_dir / "state.json")["status"] == "done"
    assert (run_dir / "lecture.md").exists() and (run_dir / "run_log.json").exists()


def test_no_llm_gives_extractive_result(cfg, monkeypatch):
    monkeypatch.setattr("lecturelens.pipeline.check_ffmpeg", lambda: None)
    monkeypatch.setattr(Pipeline, "prepare", _fake_prepare(4))
    c = cfg.with_overrides({"verify": {"enabled": False}, "export": {"pdf": False}})
    res = Pipeline(c).run("fake.mp4", RunOptions(providers=["gemini"]), run_id="nollm")   # no key in tests
    assert res["final"]["extractive_only"] and res["final"]["concise_summary"]
    assert res["stats"]["llm_calls"] == 0
