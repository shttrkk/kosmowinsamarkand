"""Конвейер инференса: реестр бэкендов AF/BS, общий контекст запуска, профилировщик этапов.

`inference.py` — только CLI; вся работа с моделями идёт через реестры `AF_BACKENDS` / `BS_BACKENDS`.
Бэкенд — функция `(data_dir, chip_ids, cfg, ctx) -> (preds, n_failed)`:

    preds     dict chip_id → маска uint8 (H, W): AF 0/1, BS 0..3 — ДЛЯ КАЖДОГО chip_id из списка
              (битый/отсутствующий чип → пустая маска, см. `empty_mask`, а не исключение);
    n_failed  сколько чипов не удалось прочитать/обработать.

Какой бэкенд работает, задаёт `configs/inference.yaml`: `af.model`, `bs.model`.

КАК ПОДКЛЮЧИТЬ НОВУЮ МОДЕЛЬ (например, BSPredictor из ветки task/bs) — одно место, см. конец файла:
    1) написать функцию-бэкенд (или обернуть готовую) и добавить её в `BS_BACKENDS` под новым именем;
    2) положить веса в `weights/` и указать `bs.model: <имя>` в `configs/inference.yaml`.
`inference.py`, Dockerfile и README при этом не меняются. Если бэкенд не смог загрузиться (нет весов,
нет torch) — `run_backend` откатывается на `fallback` из конфига (по умолчанию `rules`) с ошибкой в логе:
валидный submission важнее падения.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .io import LAYERS, chip_path, read_af_chip, read_bs_chip
from .submission import DEFAULT_SHAPES
from .utils import get_logger

log = get_logger("inference")


class Profiler:
    """Накопитель времени по этапам. Потокобезопасен: для этапов внутри пула потоков копится сумма по потокам
    (она может превышать реальное время — в таблице это колонка «сумма по потокам»)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stages: dict[str, list[float]] = {}  # имя → [секунды, число вызовов]

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            rec = self.stages.setdefault(name, [0.0, 0])
            rec[0] += seconds
            rec[1] += 1

    def seconds(self, name: str) -> float:
        return self.stages.get(name, [0.0, 0])[0]

    def table(self, total: float) -> str:
        rows = [f"{'этап':<34}{'с':>9}{'вызовов':>9}{'% от полного':>14}"]
        for name, (sec, n) in self.stages.items():
            rows.append(f"{name:<34}{sec:>9.2f}{n:>9}{100 * sec / total:>13.1f}%")
        rows.append(f"{'ПОЛНОЕ ВРЕМЯ (от старта процесса)':<34}{total:>9.2f}")
        return "\n".join(rows)


@dataclass
class RunContext:
    """Общие параметры запуска, доступные каждому бэкенду."""
    shapes: dict[str, tuple[int, int]] = field(default_factory=dict)  # chip_id → (H, W) из meta.csv
    workers: int = 4                                                   # потоки чтения/признаков
    device: str = "auto"                                               # auto | cpu | cuda (для torch-бэкендов)
    profiler: Profiler = field(default_factory=Profiler)

    def torch_device(self) -> str:
        """'cuda' | 'cpu'. torch импортируется только здесь — бэкенды без нейросетей за импорт не платят."""
        if self.device == "cpu":
            return "cpu"
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if self.device == "cuda":
            log.warning("--device cuda запрошен, но CUDA недоступна — работаем на CPU")
        return "cpu"


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def empty_mask(kind: str, chip_id: str, ctx: RunContext) -> np.ndarray:
    return np.zeros(ctx.shapes.get(chip_id, DEFAULT_SHAPES[kind]), np.uint8)


def missing_layers(data_dir: Path, kind: str, chip_id: str, layers: tuple[str, ...] | None = None) -> list[str]:
    """Каких файлов чипа нет на диске (пустой список = всё на месте)."""
    keys = layers or tuple(LAYERS[kind])
    return [k for k in keys if not chip_path(data_dir, kind, LAYERS[kind][k], chip_id).exists()]


def map_chips(kind: str, data_dir: Path, chip_ids: list[str], ctx: RunContext, work: Callable,
              layers: tuple[str, ...] | None = None, pool: ThreadPoolExecutor | None = None) -> list:
    """Параллельно (потоки: rasterio и numpy отпускают GIL) читает чипы и применяет `work(chip)`.

    Возвращает список результатов в порядке `chip_ids`; для отсутствующего или битого чипа — None
    (warning/error в лог, исключение наружу не выходит). Порядок и результат не зависят от числа потоков.
    """
    reader = read_af_chip if kind == "af" else read_bs_chip
    prof = ctx.profiler

    def one(cid: str):
        absent = missing_layers(data_dir, kind, cid, layers)
        if absent:
            log.warning("%s %s: нет файлов слоёв %s — пустая маска", kind.upper(), cid, ", ".join(absent))
            return None
        try:
            with prof.stage(f"{kind}: чтение чипов"):
                chip = reader(data_dir, cid, with_mask=False, layers=layers)
            with prof.stage(f"{kind}: признаки/правила"):
                return work(chip)
        except Exception as e:  # один битый чип не должен ронять весь submission
            log.error("%s %s: %s: %s — пустая маска", kind.upper(), cid, type(e).__name__, e)
            return None

    if pool is not None:
        return list(pool.map(one, chip_ids))
    with ThreadPoolExecutor(max(1, ctx.workers)) as own:
        return list(own.map(one, chip_ids))


# ───────────────────────────── AF ─────────────────────────────

def af_rules_backend(data_dir: Path, chip_ids: list[str], cfg: dict, ctx: RunContext):
    """Контекстное правило шага 0 (OOF F1 0.8965) — аварийный вариант без весов."""
    from .models.af.rules import predict_af_context

    parts = map_chips("af", data_dir, chip_ids, ctx, predict_af_context, layers=("viirs",))
    preds = {c: (p if p is not None else empty_mask("af", c, ctx)) for c, p in zip(chip_ids, parts)}
    return preds, sum(p is None for p in parts)


def af_lgbm_backend(data_dir: Path, chip_ids: list[str], cfg: dict, ctx: RunContext):
    """LightGBM по пикселям-кандидатам: признаки считаются в потоках, пока в фоне грузятся модели,
    затем ОДИН вызов predict на строки всех чипов (накладные расходы вызова ≫ стоимости строки)."""
    from .models.af.predictor import AFPredictor

    prof = ctx.profiler
    predictor = AFPredictor(cfg).load_meta()  # ошибки весов всплывают здесь → run_backend откатится на fallback

    def load():
        with prof.stage("af: загрузка весов (фон)"):
            predictor.load_boosters()

    with ThreadPoolExecutor(max(1, ctx.workers) + 1) as pool:
        loading = pool.submit(load)
        parts = map_chips("af", data_dir, chip_ids, ctx, predictor.candidates_and_rows, pool=pool)
        loading.result()
    ok = [i for i, p in enumerate(parts) if p is not None]
    with prof.stage("af: predict LightGBM"):
        masks = dict(zip(ok, predictor.masks_from_parts([parts[i] for i in ok])))
    preds = {c: masks.get(i, empty_mask("af", c, ctx)) for i, c in enumerate(chip_ids)}
    return preds, len(chip_ids) - len(ok)


# ───────────────────────────── BS ─────────────────────────────

def bs_rules_backend(data_dir: Path, chip_ids: list[str], cfg: dict, ctx: RunContext):
    """Заглушка/аварийный вариант: глобальные пороги dNBR (наш baseline, строка 0 experiments.md)."""
    from .models.bs.rules import predict_bs_rules

    thresholds = tuple(cfg.get("dnbr_thresholds", (0.10, 0.27, 0.44)))
    parts = map_chips("bs", data_dir, chip_ids, ctx, lambda chip: predict_bs_rules(chip, thresholds),
                      layers=("s2_pre", "s2_post"))
    preds = {c: (p if p is not None else empty_mask("bs", c, ctx)) for c, p in zip(chip_ids, parts)}
    return preds, sum(p is None for p in parts)


def bs_unet_backend(data_dir: Path, chip_ids: list[str], cfg: dict, ctx: RunContext):
    """BSPredictor: 2×UNet → стек-LGBM → декодирование → постобработка (reports/bs_module.md).
    Параметры — configs/bs_inference.yaml, ключи секции `bs` общего конфига их перекрывают; устройство и потоки — из ctx.
    Нет весов → исключение → `run_backend` откатывается на fallback."""
    from .models.bs.infer import load_bs_config, run_bs_inference

    bs_cfg = load_bs_config({k: v for k, v in cfg.items() if k != "fallback"})
    bs_cfg["device"] = ctx.torch_device()
    bs_cfg["num_threads"] = max(1, ctx.workers)
    preds, n_failed, _ = run_bs_inference(data_dir, chip_ids, bs_cfg, ctx.shapes, profiler=ctx.profiler,
                                          require_models=True)
    for c in chip_ids:
        if c not in preds:
            preds[c] = empty_mask("bs", c, ctx)
    return preds, n_failed


# ───────────────────── реестры: ЕДИНСТВЕННОЕ место подключения моделей ─────────────────────

Backend = Callable[[Path, list, dict, RunContext], tuple]

AF_BACKENDS: dict[str, Backend] = {
    "lgbm": af_lgbm_backend,
    "rules": af_rules_backend,
}

BS_BACKENDS: dict[str, Backend] = {
    "unet": bs_unet_backend,   # BSPredictor (task/bs): 2×UNet + стек-LGBM
    "rules": bs_rules_backend,
}

BACKENDS = {"af": AF_BACKENDS, "bs": BS_BACKENDS}


def run_backend(kind: str, data_dir: Path, chip_ids: list[str], cfg: dict, ctx: RunContext):
    """Запускает бэкенд `cfg['model']`; если он не поднялся целиком (нет весов и т.п.) — `cfg['fallback']` (rules).

    Возвращает (preds, n_failed, имя фактически отработавшего бэкенда).
    """
    registry = BACKENDS[kind]
    name = str(cfg.get("model", "rules"))
    if name not in registry:
        raise ValueError(f"{kind}.model = {name!r}: нет такого бэкенда, доступны {sorted(registry)}")
    if not chip_ids:
        return {}, 0, name
    try:
        preds, n_failed = registry[name](data_dir, chip_ids, cfg, ctx)
        return preds, n_failed, name
    except Exception as e:
        fallback = str(cfg.get("fallback", "rules"))
        if fallback == name or fallback not in registry:
            raise
        log.error("%s: бэкенд %r не запустился (%s: %s) — используется %r", kind.upper(), name, type(e).__name__, e,
                  fallback)
        preds, n_failed = registry[fallback](data_dir, chip_ids, cfg, ctx)
        return preds, n_failed, fallback
