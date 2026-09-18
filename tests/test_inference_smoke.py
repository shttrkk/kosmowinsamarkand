from __future__ import annotations

import subprocess
import sys
import time

import pandas as pd

from firemon.io import read_meta
from firemon.metrics import evaluate_submission
from firemon.submission import shapes_from_meta, validate_submission, write_submission

from conftest import ROOT, TRAIN_DIR


def test_inference_cli_smoke(fake_test_dir, tmp_path):
    """inference.py на 2 AF + 2 BS чипах: код 0, валидный файл, < 30 с, метрика считается."""
    af = read_meta(fake_test_dir, "af")["chip_id"].tolist()[:2]
    bs = read_meta(fake_test_dir, "bs")["chip_id"].tolist()[:2]
    sample = tmp_path / "sample_submission.csv"
    write_submission([(c, 1, "") for c in af] + [(c, k, "") for c in bs for k in (1, 2, 3)], sample)
    out = tmp_path / "nested" / "submission.csv"

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), "--data-dir", str(fake_test_dir),
         "--output", str(out), "--sample-submission", str(sample)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=120)
    elapsed = time.perf_counter() - t0

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert elapsed < 30
    assert "Полное время инференса" in proc.stdout
    assert validate_submission(out, sample, shapes_from_meta(fake_test_dir)) == []
    assert len(pd.read_csv(out, keep_default_na=False)) == 8

    result = evaluate_submission(out, TRAIN_DIR, sample)
    assert result["n_af"] == 2 and result["n_bs"] == 2
    assert 0.0 <= result["score"] <= 1.0


def test_inference_is_deterministic(fake_test_dir, tmp_path):
    outs = []
    for name in ("a.csv", "b.csv"):
        out = tmp_path / name
        proc = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--data-dir", str(fake_test_dir),
                               "--output", str(out)], capture_output=True, stdin=subprocess.DEVNULL, timeout=300)
        assert proc.returncode == 0
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
