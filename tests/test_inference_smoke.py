from __future__ import annotations

import os
import subprocess
import sys
import time

import pandas as pd

from firemon.io import read_meta
from firemon.metrics import evaluate_submission
from firemon.submission import read_submission, shapes_from_meta, validate_submission, write_submission

from conftest import ROOT, TRAIN_DIR

# вывод дочернего процесса — всегда UTF-8, независимо от кодовой страницы консоли Windows
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


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
        capture_output=True, text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, env=ENV, timeout=120)
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
                               "--output", str(out)], capture_output=True, stdin=subprocess.DEVNULL, env=ENV, timeout=300)
        assert proc.returncode == 0
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


def test_inference_on_real_test(test_dir, tmp_path):
    """Полный прогон на реальном тесте: шаблон берётся из <data-dir>/sample_submission.csv, 447 строк, < 30 с."""
    out = tmp_path / "submission.csv"
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--data-dir", str(test_dir), "--output", str(out)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, env=ENV, timeout=300)
    elapsed = time.perf_counter() - t0
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert elapsed < 30, f"инференс {elapsed:.1f} с — порог 8 баллов за скорость 30 с"
    sample = test_dir / "sample_submission.csv"
    assert validate_submission(out, sample, shapes_from_meta(test_dir)) == []
    sub, tpl = read_submission(out), read_submission(sample)
    assert list(zip(sub["chip_id"], sub["class_id"])) == list(zip(tpl["chip_id"], tpl["class_id"]))
    assert len(sub["chip_id"]) == 447
    lines = out.read_text(encoding="utf-8").splitlines()[1:]
    assert all(line.count('"') == 2 and line.endswith('"') for line in lines)  # rle всегда в кавычках, пустая → ""
