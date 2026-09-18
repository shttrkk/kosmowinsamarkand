"""Шаг 0: grid-поиск порогов контекстного правила. Пороги подбираются на 4 фолдах, оцениваются на 5-м (OOF),
плюс временной сплит (подбор на ≤2023, оценка на 2024). Запуск: python scripts/af/step0_rule_grid.py"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_all  # noqa: E402

from firemon.models.af.rules import context_rule_maps  # noqa: E402

GRID = {
    "day": {"t_abs": [325, 328, 330, 332, 335, 340, 345, 350, 355, 400], "k": [2, 2.5, 3, 3.5, 4, 5, 6, 8], "di_min": [17, 20, 25, 30, 35]},
    "night": {"t_abs": [298, 300, 302, 305, 310, 315, 320, 330, 400], "k": [2, 2.5, 3, 3.5, 4, 5, 6, 8], "di_min": [9.5, 12, 15]},
}


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def main() -> None:
    meta, chips = load_all()
    n = len(chips)
    is_night = np.array([bool(np.nanmean(c["viirs"][5]) > 90) for c in chips])
    n_fire = np.array([(c["mask"] == 1).sum() for c in chips])
    combos = {p: list(itertools.product(*GRID[p].values())) for p in GRID}
    tp = {p: np.zeros((n, len(combos[p]))) for p in GRID}
    fp = {p: np.zeros((n, len(combos[p]))) for p in GRID}
    for ci, c in enumerate(chips):
        m = context_rule_maps(c)
        p = "night" if is_night[ci] else "day"
        keep = m["valid"] & (m["di"] > min(GRID[p]["di_min"]) - 1e-6)
        i4, di, z, gt = m["i4"][keep], m["di"][keep], m["zmin"][keep], (c["mask"] == 1)[keep]
        for j, (t_abs, k, di_min) in enumerate(combos[p]):
            pred = (i4 > t_abs) | ((z > k) & (di > di_min))
            tp[p][ci, j] = (pred & gt).sum()
            fp[p][ci, j] = (pred & ~gt).sum()

    def best(train_idx):
        """Совместный максимум микро-F1 по парам (день, ночь) на train_idx."""
        tpd, fpd = tp["day"][train_idx].sum(0), fp["day"][train_idx].sum(0)
        tpn, fpn = tp["night"][train_idx].sum(0), fp["night"][train_idx].sum(0)
        tot = n_fire[train_idx].sum()
        TP, FP = tpd[:, None] + tpn[None, :], fpd[:, None] + fpn[None, :]
        F = f1(TP, FP, tot - TP)
        jd, jn = np.unravel_index(F.argmax(), F.shape)
        return jd, jn, F.max()

    def evaluate(idx, jd, jn):
        TP = tp["day"][idx, jd].sum() + tp["night"][idx, jn].sum()
        FP = fp["day"][idx, jd].sum() + fp["night"][idx, jn].sum()
        return TP, FP, n_fire[idx].sum() - TP

    folds = meta["fold"].to_numpy()
    acc = np.zeros(3)
    for f in range(5):
        jd, jn, ftr = best(np.where(folds != f)[0])
        r = evaluate(np.where(folds == f)[0], jd, jn)
        acc += r
        print(f"fold {f}: day {combos['day'][jd]} night {combos['night'][jn]} train F1 {ftr:.4f} val F1 {f1(*r):.4f}")
    TP, FP, FN = acc
    print(f"OOF (GroupKFold): F1 {f1(TP, FP, FN):.4f} P {TP / (TP + FP):.4f} R {TP / (TP + FN):.4f}")

    ts = meta["temporal_split"].to_numpy()
    jd, jn, ftr = best(np.where(ts == "train")[0])
    TP, FP, FN = evaluate(np.where(ts == "val")[0], jd, jn)
    print(f"temporal: day {combos['day'][jd]} night {combos['night'][jn]} train F1 {ftr:.4f} "
          f"val-2024 F1 {f1(TP, FP, FN):.4f} P {TP / (TP + FP):.4f} R {TP / (TP + FN):.4f}")
    jd, jn, fall = best(np.arange(n))
    print(f"весь train: day (t_abs,k,di_min) {combos['day'][jd]} night {combos['night'][jn]} F1 {fall:.4f}")
    for p, sel in (("day", ~is_night), ("night", is_night)):
        j = jd if p == "day" else jn
        idx = np.where(sel)[0]
        t, fpos = tp[p][idx, j].sum(), fp[p][idx, j].sum()
        print(f"  {p}: F1 {f1(t, fpos, n_fire[idx].sum() - t):.4f} P {t / max(t + fpos, 1):.4f} R {t / n_fire[idx].sum():.4f}")


if __name__ == "__main__":
    main()
