import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """每个测试用独立的数据目录,避免互相污染。"""
    monkeypatch.setenv("TUTOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TUTOR_FAKE_LLM", "1")


@pytest.fixture
def sample_text():
    return (ROOT / "examples" / "gradient_descent.md").read_text(encoding="utf-8")


@pytest.fixture
def course(tmp_path, sample_text):
    from tutor.llm import FakeLLM
    from tutor.memory.store import Store
    from tutor.service import TutorService

    svc = TutorService(llm=FakeLLM(), store=Store(tmp_path / "t.db"))
    return svc.ingest_text("梯度下降与学习率", sample_text)
