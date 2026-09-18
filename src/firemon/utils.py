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


def get_logger(name: str = "firemon", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
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
