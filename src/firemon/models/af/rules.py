"""Правила AF: старая заглушка (фиксированные пороги) и контекстное baseline-правило по мотивам VNP14 (шаг 0)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ...features.af_features import night_mask

# Пороги подобраны grid-поиском по фолдам (scripts/af/step0_rule_grid.py), см. reports/experiments.md
CONTEXT_RULE = {"t_abs_day": 330.0, "t_abs_night": 305.0, "k_day": 8.0, "k_night": 3.0,
                "di_min_day": 35.0, "di_min_night": 15.0, "window": 7}


def predict_af_rules(chip: dict, i4_min: float = 320.0, di45_min: float = 15.0) -> np.ndarray:
    """chip — результат read_af_chip. Возвращает маску uint8 (H, W) 0/1. NaN/невалидные → 0."""
    viirs = chip["viirs"]
    i4, i5 = viirs[3], viirs[4]
    with np.errstate(invalid="ignore"):
        fire = (i4 > i4_min) & ((i4 - i5) > di45_min)
    return (fire & chip["valid"]).astype(np.uint8)


def background_z(x: np.ndarray, valid: np.ndarray, window: int = 7) -> np.ndarray:
    """z-score относительно фона окна без центра, только по valid-пикселям (NaN не портят окно)."""
    v = valid.astype(np.float64)
    xc = np.where(valid, x, 0).astype(np.float64)
    xc = (xc - (xc.sum() / max(v.sum(), 1.0))) * v               # центрирование: точность суммы квадратов
    n_win = float(window * window)
    n = ndi.uniform_filter(v, window, mode="constant") * n_win - v
    s1 = ndi.uniform_filter(xc, window, mode="constant") * n_win - xc
    s2 = ndi.uniform_filter(xc * xc, window, mode="constant") * n_win - xc * xc
    n = np.maximum(n, 1.0)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    return ((xc - mean) / (std + 1e-3)).astype(np.float32)


def context_rule_maps(chip: dict, window: int = 7) -> dict:
    viirs, valid = chip["viirs"], chip["valid"]
    i4 = np.where(valid, viirs[3], 0).astype(np.float32)
    di = np.where(valid, viirs[3] - viirs[4], 0).astype(np.float32)
    zmin = np.minimum(background_z(i4, valid, window), background_z(di, valid, window))
    return {"i4": i4, "di": di, "zmin": zmin, "night": night_mask(viirs), "valid": valid}


def predict_af_context(chip: dict, params: dict | None = None) -> np.ndarray:
    """Шаг 0: огонь = I4 > T_abs ИЛИ (z(I4) > k И z(I4−I5) > k И I4−I5 > di_min); пороги раздельно день/ночь."""
    p = {**CONTEXT_RULE, **(params or {})}
    m = context_rule_maps(chip, int(p["window"]))
    night = m["night"]
    t_abs = np.where(night, p["t_abs_night"], p["t_abs_day"])
    k = np.where(night, p["k_night"], p["k_day"])
    di_min = np.where(night, p["di_min_night"], p["di_min_day"])
    fire = (m["i4"] > t_abs) | ((m["zmin"] > k) & (m["di"] > di_min))
    return (fire & m["valid"]).astype(np.uint8)
