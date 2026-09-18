from __future__ import annotations

import numpy as np
import pytest

from firemon.features.af_features import (FEATURE_NAMES, af_feature_rows, candidate_mask, compute_af_features)
from firemon.io import read_af_chip, read_meta
from firemon.models.af.predictor import META_NAME, AFPredictor
from firemon.models.af.rules import predict_af_context
from firemon.postprocess import drop_isolated, hysteresis
from firemon.utils import REPO_ROOT


def synthetic_chip(night: bool = False, seed: int = 0) -> dict:
    """Чип 256×256 с фоном, очагом 3×3, дырой NaN и невалидной полосой."""
    rng = np.random.default_rng(seed)
    h = w = 256
    viirs = np.zeros((8, h, w), np.float32)
    viirs[0:3] = np.nan if night else rng.uniform(0.05, 0.3, (3, h, w))
    viirs[3] = (285 if night else 305) + rng.normal(0, 1.5, (h, w))
    viirs[4] = (283 if night else 300) + rng.normal(0, 1.0, (h, w))
    viirs[3, 100:103, 50:53] += 45
    viirs[4, 100:103, 50:53] += 8
    viirs[5] = 120 if night else 40
    viirs[6] = 30
    viirs[7] = 1
    viirs[:5, 10:40, 10:40] = np.nan           # NaN-дыра
    viirs[7, :, 250:] = 0                      # невалидная полоса
    viirs[3, 120, 252] = 360                   # «огонь» в невалидной зоне — предсказывать нельзя
    aux = np.stack([np.full((h, w), 40.0), rng.uniform(0, 300, (h, w)), np.full((h, w), 20.0),
                    np.full((h, w), 40.0), np.full((h, w), 3.0)]).astype(np.float32)
    aux[:, 200:210, 200:210] = np.nan
    valid = (viirs[7] > 0.5) & np.isfinite(viirs[3]) & np.isfinite(viirs[4])
    return {"chip_id": "synthetic", "viirs": viirs, "aux": aux, "valid": valid}


@pytest.fixture(scope="module")
def predictor() -> AFPredictor:
    if not (REPO_ROOT / "weights" / META_NAME).exists():
        pytest.skip("нет весов AF — запустите python train.py --module af")
    return AFPredictor({"num_threads": 2}).load()


@pytest.mark.parametrize("night", [False, True])
def test_features_shape_no_nan(night):
    chip = synthetic_chip(night)
    feats = compute_af_features(chip)
    assert feats.shape == (len(FEATURE_NAMES), 256, 256) and feats.dtype == np.float32
    assert np.isfinite(feats).all()
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    was_nan = feats[FEATURE_NAMES.index("was_nan")]
    assert was_nan[20, 20] == 1 and was_nan[205, 205] == 1 and was_nan[128, 128] == 0
    # очаг: большой z-score относительно робастного фона (без горячих соседей)
    assert feats[FEATURE_NAMES.index("i4_rz7"), 101, 51] > 3


def test_rows_match_full_map():
    """Путь инференса (только кандидаты, интегральные изображения по вырезке) == полная карта."""
    chip = synthetic_chip()
    cand = candidate_mask(chip["viirs"], chip["valid"])
    assert cand[100:103, 50:53].all() and not cand[:, 250:].any()
    rows = af_feature_rows(chip, cand)
    full = compute_af_features(chip)
    np.testing.assert_allclose(rows, full[:, cand].T, rtol=1e-5, atol=1e-4)


def test_all_nan_chip_does_not_crash():
    chip = synthetic_chip()
    chip["viirs"][:5] = np.nan
    chip["valid"] = np.zeros((256, 256), bool)
    assert np.isfinite(compute_af_features(chip)).all()
    assert predict_af_context(chip).sum() == 0


def test_context_rule_synthetic():
    chip = synthetic_chip()
    pred = predict_af_context(chip)
    assert pred.dtype == np.uint8 and set(np.unique(pred)) <= {0, 1}
    assert pred[100:103, 50:53].all() and pred[:, 250:].sum() == 0


def test_hysteresis_and_isolated():
    p = np.zeros((8, 8), np.float32)
    p[1, 1], p[1, 2], p[2, 3] = 0.9, 0.3, 0.3      # цепочка от сильного seed
    p[6, 6] = 0.4                                   # слабый одиночный
    m = hysteresis(p, 0.8, 0.2)
    assert m[1, 1] and m[1, 2] and m[2, 3] and not m[6, 6]
    assert not drop_isolated(p > 0.35, p, 0.95).any() and drop_isolated(p > 0.35, p, 0.5)[1, 1]


def test_predictor_synthetic(predictor):
    chip = synthetic_chip()
    a, b = predictor.predict(chip), predictor.predict(chip)
    assert a.shape == (256, 256) and a.dtype == np.uint8 and set(np.unique(a)) <= {0, 1}
    assert np.array_equal(a, b)
    assert a[~chip["valid"]].sum() == 0
    assert np.array_equal(predictor.predict_many([chip, chip])[1], a)


def test_predictor_on_nan_chip(predictor, train_dir):
    meta = read_meta(train_dir, "af")
    cid = meta.sort_values("valid_frac").iloc[0]["chip_id"]
    chip = read_af_chip(train_dir, cid, with_mask=False)
    assert not chip["valid"].all()
    feats = compute_af_features(chip)
    assert feats.shape == (len(FEATURE_NAMES), 256, 256) and np.isfinite(feats).all()
    pred = predictor.predict(chip)
    assert pred.shape == (256, 256) and set(np.unique(pred)) <= {0, 1}
    assert pred[~chip["valid"]].sum() == 0


def test_predictor_few_positives_without_fire(predictor, train_dir):
    meta = read_meta(train_dir, "af")
    for cid in meta[meta["n_fire_px"] == 0]["chip_id"].head(8):
        assert predictor.predict(read_af_chip(train_dir, cid, with_mask=False)).sum() < 50, cid


def test_predictor_deterministic(train_dir):
    if not (REPO_ROOT / "weights" / META_NAME).exists():
        pytest.skip("нет весов AF")
    meta = read_meta(train_dir, "af")
    cid = meta.sort_values("n_fire_px").iloc[-1]["chip_id"]
    chip = read_af_chip(train_dir, cid, with_mask=False)
    p1 = AFPredictor({"num_threads": 2}).load().predict_proba(chip)
    p2 = AFPredictor({"num_threads": 2}).load().predict_proba(chip)
    assert np.array_equal(p1, p2) and p1.max() > 0.5


def test_make_fake_test_does_not_write_to_baseline(train_dir, tmp_path):
    """Скрипт пишет только в --out-dir; фейковый шаблон по train больше не создаётся нигде (задача 1b),
    baseline/ (каталог организаторов) скрипт не трогает."""
    import os
    import subprocess
    import sys

    baseline = REPO_ROOT / "baseline"
    before = sorted(os.listdir(baseline)) if baseline.exists() else None
    out = tmp_path / "fake"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_fake_test.py"), "--train-dir", str(train_dir),
                    "--out-dir", str(out), "--n-af", "5", "--n-bs", "5"], check=True, capture_output=True)
    assert (out / "sample_submission.csv").exists() and not (out / "sample_submission_FAKE.csv").exists()
    assert (sorted(os.listdir(baseline)) if baseline.exists() else None) == before
    assert not (baseline / "sample_submission_FAKE.csv").exists()
