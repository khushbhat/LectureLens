import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# tests must never spend real quota
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GROQ_API_KEY", None)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    from lecturelens.config import load_config

    return load_config(overrides={
        "paths": {"runs_dir": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache"),
                  "results_dir": str(tmp_path / "results")},
        "llm": {"order": ["mock"], "check_models_on_startup": False},
        "quota": {"safety_factor": 1.0},
    })


class FakeClock:
    def __init__(self, t: float = 1_700_000_000.0):
        self.t = t
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()
