"""Пиксельный датасет AF для LightGBM (шаг 2): огонь + кандидаты + кольцо r=2 вокруг огня + 2 % фона.

Кэш: data/cache/af_train.npz (npz вместо parquet — в окружении нет pyarrow, лишняя зависимость не нужна).
Флаги источника строки позволяют делать абляции без пересчёта признаков.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ...features.af_features import FEATURE_NAMES, af_feature_rows, candidate_mask
from ...io import read_af_chip

RING_RADIUS = 2
RANDOM_FRAC = 0.02


def chip_rows(data_dir: str | Path, chip_id: str, chip_idx: int, seed: int = 42) -> dict:
    chip = read_af_chip(data_dir, chip_id, with_mask=True)
    valid = chip["valid"]
    fire = (chip["mask"] == 1)
    cand = candidate_mask(chip["viirs"], valid)
    ring = ndi.binary_dilation(fire, structure=np.ones((2 * RING_RADIUS + 1,) * 2, bool)) & ~fire & valid
    rng = np.random.default_rng(seed + chip_idx)
    rand = (rng.random(fire.shape) < RANDOM_FRAC) & valid
    sel = (fire & valid) | cand | ring | rand
    ys, xs = np.nonzero(sel)
    return {
        "X": af_feature_rows(chip, sel), "y": fire[ys, xs].astype(np.uint8),
        "chip_idx": np.full(len(ys), chip_idx, np.int32), "yy": ys.astype(np.int16), "xx": xs.astype(np.int16),
        "is_cand": cand[ys, xs], "is_ring": ring[ys, xs], "is_rand": rand[ys, xs],
        "n_fire": int(fire.sum()), "n_fire_invalid": int((fire & ~valid).sum()),
    }


def build_dataset(data_dir: str | Path, chip_ids: list[str], seed: int = 42, workers: int = 8) -> dict:
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda a: chip_rows(data_dir, a[1], a[0], seed), enumerate(chip_ids)))
    ds = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "chip_idx", "yy", "xx",
                                                              "is_cand", "is_ring", "is_rand")}
    ds["n_fire"] = np.array([p["n_fire"] for p in parts], np.int64)       # весь огонь чипа (для FN вне выборки)
    ds["chip_ids"] = np.array(chip_ids)
    ds["feature_names"] = np.array(FEATURE_NAMES)
    return ds


def load_or_build(data_dir: str | Path, chip_ids: list[str], cache: str | Path | None, seed: int = 42,
                  rebuild: bool = False) -> dict:
    if cache is not None and Path(cache).exists() and not rebuild:
        ds = dict(np.load(cache, allow_pickle=False))
        if list(ds["feature_names"]) == FEATURE_NAMES and list(ds["chip_ids"]) == list(chip_ids):
            return ds
    ds = build_dataset(data_dir, chip_ids, seed)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **ds)
    return ds
