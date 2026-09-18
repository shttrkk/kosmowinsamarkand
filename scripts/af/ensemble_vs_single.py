"""Шаг 6: «среднее 5 фолд-моделей» против «одна модель на всём train с тем же числом деревьев».
Честно сравнить можно только на отложенных данных → временной сплит: обе схемы строятся внутри ≤2023, оценка на 2024.
Запуск: python scripts/af/ensemble_vs_single.py"""
from __future__ import annotations

import numpy as np

from firemon.io import read_meta
from firemon.models.af.dataset import load_or_build
from firemon.models.af.train_lgbm import DEFAULT_PARAMS, fit, run_split, score_rows
from firemon.splits import load_folds
from firemon.utils import REPO_ROOT, seed_everything


def main() -> None:
    seed_everything(42, with_torch=False)
    data = REPO_ROOT / "data"
    meta = read_meta(data / "train", "af").merge(load_folds("af"), on="chip_id")
    ds = load_or_build(data / "train", meta["chip_id"].tolist(), data / "cache" / "af_train.npz")
    folds, ts = meta["fold"].to_numpy(), meta["temporal_split"].to_numpy()
    cand, cols = ds["is_cand"], np.arange(ds["X"].shape[1])
    tr, val = ts == "train", ts == "val"
    va_rows = val[ds["chip_idx"]] & cand
    Xv = ds["X"][va_rows]
    probs, iters = [], []
    for f in range(5):
        b, _, _ = run_split(ds, tr & (folds != f), tr & (folds == f), DEFAULT_PARAMS, cols, cand)
        probs.append(b.predict(Xv, num_iteration=b.best_iteration))
        iters.append(b.best_iteration)
    rows = tr[ds["chip_idx"]] & cand
    n_single = int(round(np.mean(iters) * 1.1))
    single = fit(ds["X"][rows], ds["y"][rows], ds["feature_names"], DEFAULT_PARAMS, n_single)

    def show(name, p):
        prob = np.full(len(cand), np.nan, np.float32)
        prob[va_rows] = p
        best = max((score_rows(ds, prob, t, chip_sel=val)["f1"], t) for t in np.arange(0.3, 0.71, 0.02))
        r = score_rows(ds, prob, 0.52, chip_sel=val)
        print(f"{name:34s} val-2024 F1@0.52 {r['f1']:.4f} (P {r['precision']:.4f} R {r['recall']:.4f}) | лучший порог {best[1]:.2f}: {best[0]:.4f}")

    show(f"среднее 5 фолд-моделей {iters}", np.mean(probs, axis=0))
    show(f"одна модель, {n_single} деревьев", single.predict(Xv))
    for f in range(5):
        show(f"  фолд-модель {f}", probs[f])


if __name__ == "__main__":
    main()
