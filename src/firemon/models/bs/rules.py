"""ВРЕМЕННАЯ заглушка BS: пороги dNBR по шкале USGS (заменяется агентом BS-модуля)."""
from __future__ import annotations

import numpy as np

B8A, B12 = 6, 8  # индексы каналов в Sentinel-2 чипе


def nbr(s2: np.ndarray) -> np.ndarray:
    nir = s2[B8A].astype(np.float32)
    swir = s2[B12].astype(np.float32)
    den = nir + swir
    return np.divide(nir - swir, den, out=np.zeros_like(den), where=den > 0)


def predict_bs_rules(chip: dict, thresholds: tuple[float, float, float] = (0.10, 0.27, 0.44)) -> np.ndarray:
    """chip — результат read_bs_chip. Возвращает маску uint8 (H, W) 0..3. Пиксели без данных S2 → 0."""
    dnbr = nbr(chip["s2_pre"]) - nbr(chip["s2_post"])
    sev = np.digitize(dnbr, np.asarray(thresholds, dtype=np.float32)).astype(np.uint8)
    sev[~chip["valid"]] = 0
    return sev
