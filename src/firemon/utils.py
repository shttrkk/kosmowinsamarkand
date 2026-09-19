"""Общие утилиты: детерминизм, таймеры, логирование, конфиги."""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int = 42, with_torch: bool = True) -> None:
    """Фиксирует все источники случайности (random, numpy, torch, cudnn)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # torch импортируется долго (секунды) — сидируем его, только если он установлен;
    # with_torch=False позволяет быстрому инференсу без нейросетей не платить за импорт
    if not with_torch:
        return
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


MAX_AUTO_THREADS = 16


def auto_threads(value: Any = None) -> int:
    """Число потоков: явное положительное число — как есть; None / 0 / "auto" → min(доступные ядра, 16).

    Константа из конфига плоха на чужой машине: 8 потоков на 2 ядрах — лишние переключения, на 64 — простой.
    На результат инференса число потоков не влияет (tests/test_reproducibility.py)."""
    if value not in (None, "", 0, "0", "auto"):
        return max(1, int(value))
    return max(1, min(available_cpus(), MAX_AUTO_THREADS))


def available_cpus() -> int:
    """Ядра, реально доступные процессу: os.cpu_count(), урезанный affinity и квотой cgroup (Linux, Docker `--cpus N`).

    В контейнере с `--cpus 2` os.cpu_count() возвращает все ядра хоста: 16 потоков на 2 ядрах замедляют инференс в разы.
    На Windows/macOS affinity и cgroup нет — остаётся os.cpu_count()."""
    n = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            n = min(n, len(os.sched_getaffinity(0)) or n)
        except OSError:
            pass
    quota = _cgroup_cpu_quota()
    if quota:
        n = min(n, max(1, int(quota + 0.5)))
    return max(1, n)


def _cgroup_cpu_quota(root: str | Path = "/sys/fs/cgroup") -> float | None:
    """Квота CPU из cgroup v2 (`cpu.max`) или v1 (`cpu.cfs_quota_us` / `cpu.cfs_period_us`); None — квоты нет / не Linux."""
    root = Path(root)
    try:
        v2 = root / "cpu.max"
        if v2.is_file():
            quota, period = v2.read_text().split()[:2]
            return None if quota == "max" else float(quota) / float(period)
        q, per = root / "cpu" / "cpu.cfs_quota_us", root / "cpu" / "cpu.cfs_period_us"
        if q.is_file() and per.is_file():
            quota = float(q.read_text())
            return None if quota <= 0 else quota / float(per.read_text())
    except (OSError, ValueError):
        pass
    return None


def cuda_usable(torch_module: Any = None) -> bool:
    """True, если CUDA реально работает: is_available() и пробное выделение памяти.

    Старый драйвер / несовместимая сборка torch иногда проходят is_available() и падают на первом вызове —
    тогда инференс обязан уйти на CPU, а не упасть."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return False
    try:
        if not torch_module.cuda.is_available():
            return False
        torch_module.zeros(1, device="cuda")
        return True
    except Exception as e:  # noqa: BLE001 — любая ошибка инициализации CUDA = «GPU нет»
        get_logger("firemon").warning("CUDA не инициализировалась (%s) — CPU", str(e).splitlines()[0][:200])
        return False


class Timer:
    """Контекст-менеджер: `with Timer("чтение") as t: ...; t.elapsed` (секунды)."""

    def __init__(self, name: str = "", logger: logging.Logger | None = None):
        self.name = name
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.logger is not None:
            self.logger.info("%s: %.3f с", self.name, self.elapsed)


def _safe_stdout() -> None:
    """Кириллица в логах не должна ронять процесс при LANG=C / ASCII-консоли: непредставимые символы заменяются."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def get_logger(name: str = "firemon", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        _safe_stdout()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def load_config(path: str | Path) -> dict[str, Any]:
    """Читает yaml-конфиг; пустой файл → {}."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
