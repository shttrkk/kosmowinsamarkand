"""Готовность к чужой машине (Linux + H100, reports/tasklog/h100_readiness.md): потоки, откат с CUDA, ASCII-консоль, сеть."""
from __future__ import annotations

import io
import os
import re
import sys

import pytest

from firemon import utils
from firemon.pipeline import RunContext, default_workers
from firemon.utils import auto_threads, cuda_usable

from conftest import ROOT


@pytest.mark.parametrize("cores,expected", [(2, 2), (8, 8), (64, 16), (None, 1)])
def test_auto_threads_follows_cpu_count(monkeypatch, cores, expected):
    monkeypatch.setattr(os, "cpu_count", lambda: cores)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(utils, "_cgroup_cpu_quota", lambda *a, **k: None)
    assert auto_threads(None) == auto_threads("auto") == auto_threads(0) == default_workers() == expected
    assert auto_threads(3) == 3 and auto_threads("5") == 5  # явное число сильнее


def test_cgroup_quota_limits_threads(monkeypatch, tmp_path):
    """Docker `--cpus 2`: os.cpu_count() = ядра хоста, но квота cgroup = 2 → 2 потока; вне Linux файла нет → без изменений."""
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    read_quota = utils._cgroup_cpu_quota
    (tmp_path / "cpu.max").write_text("200000 100000")
    assert read_quota(tmp_path) == 2.0
    monkeypatch.setattr(utils, "_cgroup_cpu_quota", lambda *a, **k: 2.0)
    assert utils.available_cpus() == 2 and auto_threads("auto") == 2
    (tmp_path / "cpu.max").write_text("max 100000")
    assert read_quota(tmp_path) is None
    assert read_quota(tmp_path / "нет такого каталога") is None   # Windows / нет cgroup → no-op
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("150000")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert read_quota(v1) == 1.5


def test_configs_use_auto_threads():
    from firemon.utils import load_config

    cfg = load_config(ROOT / "configs" / "inference.yaml")
    assert cfg["workers"] == "auto" and cfg["af"]["num_threads"] == "auto"
    assert load_config(ROOT / "configs" / "bs_inference.yaml")["bs"]["num_threads"] == "auto"


def test_cuda_failure_falls_back_to_cpu():
    """is_available() = True, но первый CUDA-вызов падает (старый драйвер) → CPU, без исключения."""

    class _Cuda:
        @staticmethod
        def is_available():
            return True

    class _Torch:
        cuda = _Cuda()

        @staticmethod
        def zeros(*a, **k):
            raise RuntimeError("CUDA driver version is insufficient for CUDA runtime version")

    assert cuda_usable(_Torch()) is False
    _Torch.cuda.is_available = staticmethod(lambda: False)
    assert cuda_usable(_Torch()) is False
    assert RunContext(device="cpu").torch_device() == "cpu"


def test_cyrillic_log_survives_ascii_console(monkeypatch):
    """LANG=C: stdout в ASCII со strict-ошибками; после get_logger кириллица не роняет процесс."""
    raw = io.BytesIO()
    fake = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    monkeypatch.setattr(sys, "stdout", fake)
    utils.get_logger("firemon.test_ascii").info("проверка: кириллица в логе, 269 чипов")
    fake.flush()
    assert b"269" in raw.getvalue()


def test_inference_path_has_no_network_or_pretrained_downloads():
    """На пути инференса нет сетевых вызовов; UNet создаётся без скачивания весов энкодера (encoder_weights=None)."""
    src = ROOT / "src" / "firemon"
    files = [ROOT / "inference.py", src / "pipeline.py", src / "io.py", src / "submission.py", src / "rle.py", src / "utils.py",
             src / "models" / "bs" / "predictor.py", src / "models" / "bs" / "infer.py", src / "models" / "bs" / "postprocess.py",
             src / "models" / "af" / "predictor.py", src / "features" / "af_features.py", src / "features" / "bs_features.py"]
    banned = re.compile(r"torch\.hub|urllib|requests\.|http://|https://|urlretrieve|subprocess|socket")
    for f in files:
        hits = [line for line in f.read_text(encoding="utf-8").splitlines() if banned.search(line) and not line.lstrip().startswith("#")]
        assert not hits, (f.name, hits)
    predictor = (src / "models" / "bs" / "predictor.py").read_text(encoding="utf-8")
    assert "pretrained=False" in predictor and "weights_only=True" in predictor
    assert "allow_tf32 = False" in predictor and "CUBLAS_WORKSPACE_CONFIG" in predictor
    unet = (src / "models" / "bs" / "unet.py").read_text(encoding="utf-8")
    assert 'encoder_weights="imagenet" if pretrained else None' in unet


def test_inference_imports_stay_lean():
    """Путь инференса не тянет то, что ему не нужно (сервис/EDA/графики): geopandas, shapely, pyproj, matplotlib, albumentations;
    `--help` не импортирует torch и LightGBM вовсе. Проверяется в отдельном процессе, чтобы не мешали импорты других тестов."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'src'); sys.argv = ['inference.py', '--help']\n"
        "import runpy\n"
        "try:\n"
        "    runpy.run_path('inference.py', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "assert 'torch' not in sys.modules and 'lightgbm' not in sys.modules, 'help path imports heavy ML stack'\n"
        "import firemon.models.af.predictor, firemon.models.bs.predictor, firemon.models.bs.infer, firemon.features.bs_features\n"
        "bad = [m for m in ('geopandas', 'shapely', 'pyproj', 'matplotlib', 'albumentations', 'fastapi') if m in sys.modules]\n"
        "assert not bad, bad\n"
        "scipy_mods = sorted(m for m in sys.modules if m.startswith('scipy.') and m.count('.') == 1)\n"
        "print('OK', len(sys.modules))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-800:]


def test_prewarm_is_optional_and_harmless():
    """Фоновый прогрев torch/CUDA: для бэкенда rules не запускается; для unet — поток, который завершается без исключений."""
    from firemon.pipeline import prewarm_torch

    assert prewarm_torch({"bs": {"model": "rules"}}, RunContext(device="cpu")) is None
    assert prewarm_torch({}, RunContext(device="cpu")) is None
    thread = prewarm_torch({"bs": {"model": "unet"}}, RunContext(device="cpu"))
    assert thread is not None and thread.daemon
    thread.join(timeout=120)
    assert not thread.is_alive()
