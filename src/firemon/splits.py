"""Групповые фолды и временные сплиты (CLAUDE.md §4: тест разделён по территории И по времени).

Все агенты моделей обязаны брать фолды из configs/folds_af.csv и configs/folds_bs.csv.
Пересоздать: python -m firemon.splits --data-dir data/train
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BS_BLOCK_M = 50_000  # размер пространственного блока BS, м
VAL_YEAR = 2024


def _balanced_group_folds(groups: pd.Series, weights: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Жадное распределение групп по фолдам: детерминированно (seed), без утечек групп,
    с балансом по нескольким весам (столбцы `weights`, например число чипов и пикселей класса)."""
    weights = np.asarray(weights, dtype=float).reshape(len(groups), -1)
    weights = weights / np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
    codes, uniques = pd.factorize(groups, sort=True)
    gw = np.zeros((len(uniques), weights.shape[1]))
    np.add.at(gw, codes, weights)

    order = np.random.RandomState(seed).permutation(len(uniques))
    order = order[np.argsort(-gw[order].sum(axis=1), kind="stable")]  # крупные группы первыми
    load = np.zeros((n_folds, weights.shape[1]))
    fold_of_group = np.empty(len(uniques), dtype=int)
    for g in order:
        f = int(np.argmin((load + gw[g]).max(axis=1) + 1e-9 * np.arange(n_folds)))
        fold_of_group[g] = f
        load[f] += gw[g]
    return fold_of_group[codes]


def af_groups(meta: pd.DataFrame) -> pd.Series:
    """Группа AF = тайл (epsg, x_min, y_min)."""
    return (meta["epsg"].astype(int).astype(str) + "_" + meta["x_min"].round().astype(int).astype(str)
            + "_" + meta["y_min"].round().astype(int).astype(str))


def bs_groups(meta: pd.DataFrame, block_m: int = BS_BLOCK_M) -> pd.Series:
    """Группа BS = пространственный блок block_m × block_m внутри зоны UTM."""
    bx = np.floor(meta["x_min"] / block_m).astype(int).astype(str)
    by = np.floor(meta["y_min"] / block_m).astype(int).astype(str)
    return meta["epsg"].astype(int).astype(str) + "_" + bx + "_" + by


def make_af_folds(meta: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    w = np.c_[np.ones(len(meta)), meta["n_fire_px"].fillna(0).to_numpy()]
    return _balanced_group_folds(af_groups(meta), w, n_folds, seed)


def make_bs_folds(meta: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    burn_px = meta[["sev1_px", "sev2_px", "sev3_px"]].fillna(0).sum(axis=1).to_numpy()
    w = np.c_[np.ones(len(meta)), burn_px]
    return _balanced_group_folds(bs_groups(meta), w, n_folds, seed)


def _temporal(years: pd.Series) -> np.ndarray:
    """val = VAL_YEAR, train = более ранние годы, holdout = более поздние (не использовать в train
    при временной валидации, иначе обучение «заглядывает в будущее»)."""
    return np.where(years == VAL_YEAR, "val", np.where(years < VAL_YEAR, "train", "holdout"))


def make_af_temporal_split(meta: pd.DataFrame) -> np.ndarray:
    return _temporal(pd.to_datetime(meta["acq_datetime"]).dt.year)


def make_bs_temporal_split(meta: pd.DataFrame) -> np.ndarray:
    """BS: val = 2024, train = всё остальное (по ТЗ задачи; пожары независимы, лет мало)."""
    years = pd.to_datetime(meta["date_post"]).dt.year
    return np.where(years == VAL_YEAR, "val", "train")


def build_fold_tables(data_dir: str | Path, n_folds: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .io import read_meta

    af, bs = read_meta(data_dir, "af"), read_meta(data_dir, "bs")
    af_t = pd.DataFrame({"chip_id": af["chip_id"], "fold": make_af_folds(af, n_folds, seed),
                         "temporal_split": make_af_temporal_split(af), "group": af_groups(af)})
    bs_t = pd.DataFrame({"chip_id": bs["chip_id"], "fold": make_bs_folds(bs, n_folds, seed),
                         "temporal_split": make_bs_temporal_split(bs), "group": bs_groups(bs)})
    return af_t, bs_t


def load_folds(kind: str, configs_dir: str | Path | None = None) -> pd.DataFrame:
    """Зафиксированные фолды: колонки chip_id, fold, temporal_split, group."""
    from .utils import REPO_ROOT

    return pd.read_csv(Path(configs_dir or REPO_ROOT / "configs") / f"folds_{kind}.csv")


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Генерация configs/folds_{af,bs}.csv")
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--out-dir", default="configs")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from .io import read_meta

    tables = build_fold_tables(args.data_dir, args.n_folds, args.seed)
    for kind, table in zip(("af", "bs"), tables):
        out = Path(args.out_dir) / f"folds_{kind}.csv"
        table.to_csv(out, index=False, lineterminator="\n")
        meta = read_meta(args.data_dir, kind).merge(table, on="chip_id")
        print(f"\n== {kind.upper()}: {len(table)} чипов, {table['group'].nunique()} групп → {out}")
        if kind == "af":
            meta["night"] = pd.to_datetime(meta["acq_datetime"]).dt.hour.isin([20, 21, 22, 23, 0, 1])
            meta["pos"] = meta["n_fire_px"] > 0
            agg = dict(chips=("chip_id", "size"), groups=("group", "nunique"), pos=("pos", "sum"),
                       night=("night", "sum"), fire_px=("n_fire_px", "sum"))
        else:
            meta["burn_px"] = meta[["sev1_px", "sev2_px", "sev3_px"]].sum(axis=1)
            agg = dict(chips=("chip_id", "size"), groups=("group", "nunique"), events=("fire_event_id", "nunique"),
                       sev1=("sev1_px", "sum"), sev2=("sev2_px", "sum"), sev3=("sev3_px", "sum"))
        print(meta.groupby("fold").agg(**agg).to_string())
        print(meta.groupby("temporal_split").agg(**agg).to_string())


if __name__ == "__main__":
    _main()
