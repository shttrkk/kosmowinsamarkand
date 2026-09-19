"""BS-часть inference.py: чтение чипов в потоках (deflate отпускает GIL) + батчевое предсказание."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ...io import read_bs_chip
from ...utils import REPO_ROOT, get_logger, load_config
from .predictor import BSPredictor

log = get_logger("bs.infer")
DEFAULT_SHAPE = (512, 512)


def load_bs_config(inference_cfg: dict | None = None) -> dict:
    """Секция `bs` из configs/bs_inference.yaml; ключи из секции `bs` общего inference.yaml её перекрывают."""
    path = REPO_ROOT / "configs" / "bs_inference.yaml"
    cfg = dict(load_config(path).get("bs", {})) if path.exists() else {}
    variants = cfg.pop("variants", None) or {}
    over = {k: v for k, v in (inference_cfg or {}).items() if k not in ("model", "dnbr_thresholds")}
    variant = str(over.pop("variant", cfg.pop("variant", "v4")) or "v4")
    if variant != "v4":  # пресет варианта (например ms — многомасштабный v5) поверх базовых настроек v4
        if variant not in variants:
            raise ValueError(f"bs.variant = {variant!r}: нет такого варианта, доступны {['v4'] + sorted(variants)}")
        cfg.update(variants[variant])
    cfg.update(over)  # явные ключи секции bs общего конфига сильнее пресета
    cfg["variant"] = variant
    return cfg


def run_bs_inference(data_dir: str | Path, chip_ids: list[str], cfg: dict | None = None,
                     shapes: dict | None = None, profiler=None,
                     require_models: bool = False) -> tuple[dict[str, np.ndarray], int, dict]:
    """→ (маски по chip_id, число сбойных чипов, тайминги). Битый чип → пустая маска, инференс не падает.
    profiler — `firemon.pipeline.Profiler` (этапы bs: …); require_models — исключение, если веса не загрузились
    (тогда `run_backend` откатится на fallback из конфига)."""
    shapes = shapes or {}

    def stage(name: str):
        return profiler.stage(name) if profiler is not None else nullcontext()

    timing = {"load": 0.0, "predict": 0.0}
    preds: dict[str, np.ndarray] = {}
    if not chip_ids:
        return preds, 0, timing
    t0 = time.perf_counter()
    with stage("bs: загрузка моделей"):
        predictor = BSPredictor(cfg).load()
    timing["load"] = time.perf_counter() - t0
    if require_models and not predictor.models and predictor.booster is None:
        raise FileNotFoundError("BS: веса моделей не найдены (weights/bs_unet_*.pt)")

    def read(cid: str):
        """Чтение + признаки в фоновом потоке (deflate, numpy и cv2 отпускают GIL)."""
        try:
            with stage("bs: чтение чипов"):
                chip = read_bs_chip(data_dir, cid, with_mask=False)
            chip["transform"] = chip["crs"] = None  # геопривязка модели не нужна и не используется (в тесте её нет)
            with stage("bs: признаки"):
                return cid, chip, predictor.prepare(chip)
        except Exception as e:  # noqa: BLE001
            log.error("BS %s: ошибка чтения: %s — пустая маска", cid, e)
            return cid, None, None

    n_failed = 0
    bs = int(predictor.cfg["batch_size"])
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max(1, int(predictor.cfg["num_threads"]))) as ex:
        def prefetch(window: int):
            """Упорядоченный prefetch с ограниченным окном (признаки одного чипа ≈ 110 МБ)."""
            from collections import deque

            pending: deque = deque()
            it = iter(chip_ids)
            for cid in it:
                pending.append(ex.submit(read, cid))
                if len(pending) >= window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

        chips_iter = prefetch(2 * bs)
        batch: list[tuple[str, dict, object]] = []

        stage2 = ThreadPoolExecutor(2)  # этап 2 (LGBM + постобработка) батча k идёт параллельно с UNet батча k+1
        ex2 = ThreadPoolExecutor(4)     # свой пул этапа 2: в общем ex его задачи стояли бы в очереди за prefetch
        running: list = []

        def one_by_one(items) -> list[np.ndarray]:
            nonlocal n_failed
            masks = []
            for cid, c, _ in items:
                try:
                    masks.append(predictor.predict(c))
                except Exception as e2:  # noqa: BLE001
                    n_failed += 1
                    log.error("BS %s: %s — пустая маска", cid, e2)
                    masks.append(np.zeros(shapes.get(cid, DEFAULT_SHAPE), np.uint8))
            return masks

        def finish(items, probs) -> None:
            try:
                with stage("bs: стек-LGBM + постобработка"):
                    masks = predictor.second_stage([c for _, c, _ in items], [f for _, _, f in items], probs, ex2)
            except Exception as e:  # noqa: BLE001
                log.error("BS батч (этап 2): %s — повтор по одному чипу", e)
                masks = one_by_one(items)
            for (cid, _, _), m in zip(items, masks):
                preds[cid] = m

        def flush():
            if not batch:
                return
            items = list(batch)
            batch.clear()
            if items[0][2] is None:  # модели не загружены — правила
                for (cid, _, _), m in zip(items, predictor.predict_batch([c for _, c, _ in items])):
                    preds[cid] = m
                return
            try:
                with stage("bs: UNet"):
                    probs = predictor.first_stage([f for _, _, f in items])
            except Exception as e:  # noqa: BLE001 — батч упал: пробуем по одному
                log.error("BS батч (этап 1): %s — повтор по одному чипу", e)
                for (cid, _, _), m in zip(items, one_by_one(items)):
                    preds[cid] = m
                return
            while len(running) >= 2:  # не копим признаки в памяти
                running.pop(0).result()
            running.append(stage2.submit(finish, items, probs))

        for cid, chip, feats_ in chips_iter:
            if chip is None:
                n_failed += 1
                preds[cid] = np.zeros(shapes.get(cid, DEFAULT_SHAPE), np.uint8)
                continue
            batch.append((cid, chip, feats_))
            if len(batch) >= bs:
                flush()
        flush()
        for fut in running:
            fut.result()
        stage2.shutdown()
        ex2.shutdown()
    timing["predict"] = time.perf_counter() - t0
    return preds, n_failed, timing
