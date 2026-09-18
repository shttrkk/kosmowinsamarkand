"""Загрузка всего AF-train в память (для исследовательских скриптов scripts/af/*)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


from firemon.io import read_af_chip, read_meta
from firemon.splits import load_folds

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "train"


def load_all(data_dir: Path = TRAIN):
    meta = read_meta(data_dir, "af").merge(load_folds("af"), on="chip_id")
    with ThreadPoolExecutor(8) as ex:
        chips = list(ex.map(lambda c: read_af_chip(data_dir, c, with_mask=True), meta["chip_id"]))
    return meta, chips
