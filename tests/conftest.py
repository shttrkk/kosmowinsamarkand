from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRAIN_DIR = ROOT / "data" / "train"
FAKE_TEST_DIR = ROOT / "data" / "test_fake"
BASELINE_DIR = ROOT / "baseline"


@pytest.fixture(scope="session")
def train_dir() -> Path:
    if not (TRAIN_DIR / "af" / "meta.csv").exists():
        pytest.skip("нет data/train — см. README, раздел «Подготовка данных»")
    return TRAIN_DIR


@pytest.fixture(scope="session")
def fake_test_dir() -> Path:
    if not (FAKE_TEST_DIR / "sample_submission.csv").exists():
        pytest.skip("нет data/test_fake — запустите python scripts/make_fake_test.py")
    return FAKE_TEST_DIR
