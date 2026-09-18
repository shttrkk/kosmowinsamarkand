from __future__ import annotations

import numpy as np
import pandas as pd

from firemon.io import read_meta
from firemon.splits import (af_groups, bs_groups, build_fold_tables, load_folds, make_af_folds,
                            make_af_temporal_split, make_bs_folds, make_bs_temporal_split)


def _no_leak(groups, folds):
    return pd.DataFrame({"g": np.asarray(groups), "f": folds}).groupby("g")["f"].nunique().max() == 1


def test_synthetic_no_leak_and_deterministic():
    rng = np.random.RandomState(0)
    meta = pd.DataFrame({
        "epsg": rng.choice([32637, 32638], 300),
        "x_min": rng.choice(np.arange(10) * 96_000.0, 300),
        "y_min": rng.choice(np.arange(4) * 96_000.0 + 5_000_000, 300),
        "n_fire_px": rng.randint(0, 100, 300),
    })
    folds = make_af_folds(meta, 5, 42)
    assert _no_leak(af_groups(meta), folds)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert np.array_equal(folds, make_af_folds(meta, 5, 42))
    counts = np.bincount(folds)
    assert counts.max() - counts.min() <= 0.25 * counts.mean()


def test_af_folds_real(train_dir):
    meta = read_meta(train_dir, "af")
    folds = make_af_folds(meta)
    groups = af_groups(meta)
    assert groups.nunique() == 70
    assert _no_leak(groups, folds)
    pos = (meta["n_fire_px"] > 0).to_numpy()
    for f in range(5):
        assert 60 <= (folds == f).sum() <= 110
        assert pos[folds == f].any() and (~pos[folds == f]).any()


def test_af_temporal_real(train_dir):
    meta = read_meta(train_dir, "af")
    split = make_af_temporal_split(meta)
    val = meta[split == "val"]
    assert len(val) > 0 and (split == "train").sum() > 0
    assert (val["n_fire_px"] > 0).any() and (val["n_fire_px"] == 0).any()
    hours = pd.to_datetime(val["acq_datetime"]).dt.hour
    assert hours.isin([20, 21, 22, 23, 0, 1]).any(), "в val нет ночных пролётов"
    years = pd.to_datetime(meta["acq_datetime"]).dt.year
    assert years[split == "train"].max() < 2024 == years[split == "val"].min()


def test_bs_folds_real(train_dir):
    meta = read_meta(train_dir, "bs")
    folds = make_bs_folds(meta)
    assert _no_leak(bs_groups(meta), folds)
    assert _no_leak(meta["fire_event_id"], folds)  # один пожар — один фолд
    assert meta["fire_event_id"].nunique() == 224
    for f in range(5):
        assert 30 <= (folds == f).sum() <= 60
    split = make_bs_temporal_split(meta)
    assert (split == "val").sum() > 0 and (split == "train").sum() > 0


def test_saved_folds_match_code(train_dir):
    """configs/folds_*.csv — источник истины для всех агентов; должен воспроизводиться кодом."""
    for kind, table in zip(("af", "bs"), build_fold_tables(train_dir)):
        saved = load_folds(kind)
        assert list(saved.columns) == ["chip_id", "fold", "temporal_split", "group"]
        pd.testing.assert_frame_equal(saved, table.reset_index(drop=True), check_dtype=False)
