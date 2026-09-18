"""ВРЕМЕННАЯ заглушка AF: фиксированные пороги I4 и I4−I5 (заменяется агентом AF-модуля)."""
from __future__ import annotations

import numpy as np


def predict_af_rules(chip: dict, i4_min: float = 320.0, di45_min: float = 15.0) -> np.ndarray:
    """chip — результат read_af_chip. Возвращает маску uint8 (H, W) 0/1. NaN/невалидные → 0."""
    viirs = chip["viirs"]
    i4, i5 = viirs[3], viirs[4]
    with np.errstate(invalid="ignore"):
        fire = (i4 > i4_min) & ((i4 - i5) > di45_min)
    return (fire & chip["valid"]).astype(np.uint8)
