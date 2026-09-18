"""Baseline-правила BS: пороги dNBR — глобальные (USGS) и подобранные по типу покрова (landcover).

Используются как точка отсчёта в отчёте (шаг 0) и как аварийный режим `model: rules` в inference.py.
"""
from __future__ import annotations

import numpy as np

B8A, B12 = 6, 8  # индексы каналов в Sentinel-2 чипе
USGS = (0.10, 0.27, 0.44)
LC_CODES = (10, 20, 30, 40, 50, 60, 80, 90)
_EDGES = np.round(np.arange(-0.2, 1.2001, 0.01), 4)  # сетка порогов / бины гистограмм


def nbr(s2: np.ndarray) -> np.ndarray:
    nir = s2[B8A].astype(np.float32)
    swir = s2[B12].astype(np.float32)
    den = nir + swir
    return np.divide(nir - swir, den, out=np.zeros_like(den), where=den > 0)


def predict_bs_rules(chip: dict, thresholds: tuple[float, float, float] = USGS,
                     lc_thresholds: dict[int, tuple[float, float, float]] | None = None) -> np.ndarray:
    """chip — результат read_bs_chip. Возвращает маску uint8 (H, W) 0..3. Пиксели без данных S2 → 0.
    lc_thresholds: {код landcover: (t1, t2, t3)} — для остальных кодов действуют `thresholds`."""
    dnbr = nbr(chip["s2_pre"]) - nbr(chip["s2_post"])
    sev = np.digitize(dnbr, np.asarray(thresholds, dtype=np.float32)).astype(np.uint8)
    if lc_thresholds:
        lc = chip["aux"][2]
        for code, thr in lc_thresholds.items():
            sel = lc == int(code)
            if sel.any():
                sev[sel] = np.digitize(dnbr[sel], np.asarray(thr, dtype=np.float32))
    sev[~chip["valid"]] = 0
    return sev


def _histograms(chips: list[dict]) -> np.ndarray:
    """hist[lc_idx, gt, bin] — число пикселей; bin = np.digitize(dNBR, _EDGES)."""
    hist = np.zeros((len(LC_CODES), 4, len(_EDGES) + 1), np.int64)
    for chip in chips:
        dnbr = nbr(chip["s2_pre"]) - nbr(chip["s2_post"])
        ok = chip["valid"] & (chip["mask"] <= 3)
        b = np.digitize(dnbr, _EDGES)
        lc = chip["aux"][2]
        for i, code in enumerate(LC_CODES):
            sel = ok & (lc == code)
            if sel.any():
                np.add.at(hist[i], (chip["mask"][sel], b[sel]), 1)
    return hist


def fit_lc_thresholds(chips: list[dict], n_rounds: int = 3) -> dict[int, tuple[float, float, float]]:
    """Координатный спуск по порогам (t1<t2<t3 для каждого landcover), цель — глобальная
    0.35·IoU_burn + 0.30·mIoU_sev, считается по гистограммам (без прохода по пикселям)."""
    hist = _histograms(chips)
    cum = np.cumsum(hist, axis=2)  # cum[..., j] = пикселей с bin ≤ j, т.е. dNBR < _EDGES[j]
    n_e = len(_EDGES)
    thr = np.tile(np.searchsorted(_EDGES, USGS), (len(LC_CODES), 1))  # индексы в _EDGES

    def conf_lc(i: int, t: np.ndarray) -> np.ndarray:
        """conf[gt, pred] для покрова i при порогах t (индексы)."""
        c = cum[i]
        bounds = [0, c[:, t[0]], c[:, t[1]], c[:, t[2]], c[:, -1]]
        return np.stack([bounds[k + 1] - bounds[k] for k in range(4)], axis=1)

    def objective(conf: np.ndarray) -> float:
        ious = []
        for k in (1, 2, 3):
            den = conf[k].sum() + conf[:, k].sum() - conf[k, k]
            ious.append(1.0 if den == 0 else conf[k, k] / den)
        tp = conf[1:, 1:].sum()
        den = tp + conf[0, 1:].sum() + conf[1:, 0].sum()
        return 0.35 * (1.0 if den == 0 else tp / den) + 0.30 * float(np.mean(ious))

    confs = [conf_lc(i, thr[i]) for i in range(len(LC_CODES))]
    for _ in range(n_rounds):
        for i in range(len(LC_CODES)):
            if hist[i, 1:].sum() < 1000:  # мало гари на этом покрове — оставляем USGS
                continue
            rest = sum(confs[j] for j in range(len(LC_CODES)) if j != i)
            for j in range(3):
                lo = thr[i, j - 1] + 1 if j > 0 else 0
                hi = thr[i, j + 1] - 1 if j < 2 else n_e - 1
                best_v, best_s = thr[i, j], -1.0
                for v in range(lo, hi + 1):
                    t = thr[i].copy()
                    t[j] = v
                    s = objective(rest + conf_lc(i, t))
                    if s > best_s:
                        best_v, best_s = v, s
                thr[i, j] = best_v
            confs[i] = conf_lc(i, thr[i])
    return {int(code): tuple(float(_EDGES[t]) for t in thr[i]) for i, code in enumerate(LC_CODES)}
