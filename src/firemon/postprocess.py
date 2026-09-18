"""Постобработка масок. AF: гистерезис по вероятностям (seed p>t_high, рост до p>t_low в 8-связности)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_S8 = np.ones((3, 3), bool)


def hysteresis(prob: np.ndarray, t_high: float, t_low: float) -> np.ndarray:
    """Компоненты {p > t_low} (8-связность), содержащие хотя бы один пиксель p > t_high. → bool (H, W)."""
    low = prob > t_low
    if t_low >= t_high or not low.any():
        return prob > t_high
    labels, n = ndi.label(low, structure=_S8)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(labels[prob > t_high])] = True
    keep[0] = False
    return keep[labels]


def drop_isolated(mask: np.ndarray, prob: np.ndarray, t_iso: float) -> np.ndarray:
    """Убрать одиночные (без соседей в 8-связности) положительные пиксели с вероятностью ниже t_iso."""
    nb = ndi.convolve(mask.astype(np.uint8), _S8.astype(np.uint8), mode="constant") - mask
    return mask & ~((nb == 0) & (prob < t_iso))
