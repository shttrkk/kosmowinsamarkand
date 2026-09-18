"""BS-часть inference.py: чтение чипов в потоках (deflate отпускает GIL) + батчевое предсказание."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
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
    cfg.update({k: v for k, v in (inference_cfg or {}).items() if k not in ("model", "dnbr_thresholds")})
    return cfg


def run_bs_inference(data_dir: str | Path, chip_ids: list[str], cfg: dict | None = None,
                     shapes: dict | None = None) -> tuple[dict[str, np.ndarray], int, dict]:
    """→ (маски по chip_id, число сбойных чипов, тайминги). Битый чип → пустая маска, инференс не падает."""
    shapes = shapes or {}
    timing = {"load": 0.0, "predict": 0.0}
    preds: dict[str, np.ndarray] = {}
    if not chip_ids:
        return preds, 0, timing
    t0 = time.perf_counter()
    predictor = BSPredictor(cfg).load()
    timing["load"] = time.perf_counter() - t0

    def read(cid: str):
        try:
            return cid, read_bs_chip(data_dir, cid, with_mask=False)
        except Exception as e:  # noqa: BLE001
            log.error("BS %s: ошибка чтения: %s — пустая маска", cid, e)
            return cid, None

    n_failed = 0
    bs = int(predictor.cfg["batch_size"])
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max(1, int(predictor.cfg["num_threads"]))) as ex:
        chips_iter = ex.map(read, chip_ids)  # упорядоченный prefetch: чтение идёт, пока считается батч
        batch: list[tuple[str, dict]] = []

        def flush():
            nonlocal n_failed
            if not batch:
                return
            try:
                masks = predictor.predict_batch([c for _, c in batch])
            except Exception as e:  # noqa: BLE001 — батч упал: пробуем по одному
                log.error("BS батч: %s — повтор по одному чипу", e)
                masks = []
                for cid, c in batch:
                    try:
                        masks.append(predictor.predict(c))
                    except Exception as e2:  # noqa: BLE001
                        n_failed += 1
                        log.error("BS %s: %s — пустая маска", cid, e2)
                        masks.append(np.zeros(shapes.get(cid, DEFAULT_SHAPE), np.uint8))
            for (cid, _), m in zip(batch, masks):
                preds[cid] = m
            batch.clear()

        for cid, chip in chips_iter:
            if chip is None:
                n_failed += 1
                preds[cid] = np.zeros(shapes.get(cid, DEFAULT_SHAPE), np.uint8)
                continue
            batch.append((cid, chip))
            if len(batch) >= bs:
                flush()
        flush()
    timing["predict"] = time.perf_counter() - t0
    return preds, n_failed, timing
