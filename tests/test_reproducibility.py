"""Детерминизм инференса: повторные запуски дают побайтно одинаковый submission.csv.

Критерий организаторов — метрика при перезапуске в пределах ±0.005; наша цель — 0.000, т.е. идентичные файлы,
независимо от числа потоков чтения/признаков и потоков LightGBM.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

from conftest import ROOT

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def _infer(data_dir, out, *extra: str) -> str:
    proc = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--data-dir", str(data_dir), "--output", str(out),
                           *extra], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, env=ENV, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return hashlib.sha256(out.read_bytes()).hexdigest()


def test_two_runs_are_byte_identical(fake_test_dir, tmp_path):
    assert _infer(fake_test_dir, tmp_path / "a.csv") == _infer(fake_test_dir, tmp_path / "b.csv")


def test_result_does_not_depend_on_workers(fake_test_dir, tmp_path):
    hashes = {w: _infer(fake_test_dir, tmp_path / f"w{w}.csv", "--workers", str(w)) for w in (1, 3, 8)}
    assert len(set(hashes.values())) == 1, hashes


def test_result_does_not_depend_on_lgbm_threads(fake_test_dir, tmp_path):
    """LightGBM predict: строки независимы → число потоков не меняет вероятности и маску."""
    import yaml

    base = yaml.safe_load((ROOT / "configs" / "inference.yaml").read_text(encoding="utf-8"))
    hashes = []
    for n in (1, 8):
        cfg = {**base, "af": {**base["af"], "num_threads": n, "weights_dir": str(ROOT / "weights")}}
        path = tmp_path / f"cfg{n}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        hashes.append(_infer(fake_test_dir, tmp_path / f"t{n}.csv", "--config", str(path)))
    assert hashes[0] == hashes[1]


def test_real_test_two_runs_are_byte_identical(test_dir, tmp_path):
    assert _infer(test_dir, tmp_path / "a.csv") == _infer(test_dir, tmp_path / "b.csv", "--workers", "2")
