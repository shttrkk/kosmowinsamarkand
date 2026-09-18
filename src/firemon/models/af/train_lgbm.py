"""Обучение LightGBM для AF: CV по фиксированным фолдам, подбор порога по OOF микро-F1, временной сплит."""
from __future__ import annotations

import numpy as np

from ...features.af_features import CATEGORICAL

DEFAULT_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
    "feature_fraction": 0.8, "bagging_fraction": 1.0, "lambda_l2": 1.0, "max_bin": 255,
    "scale_pos_weight": 1.0, "deterministic": True, "force_row_wise": True,
    "num_threads": 8, "seed": 42, "verbosity": -1,
}
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def f1_counts(tp: float, fp: float, fn: float) -> dict:
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def fit(X, y, names, params, n_rounds, valid=None, early_stopping=100):
    import lightgbm as lgb

    cats = [n for n in CATEGORICAL if n in names]
    dtrain = lgb.Dataset(X, y, feature_name=list(names), categorical_feature=cats, free_raw_data=False)
    if valid is None:
        return lgb.train(params, dtrain, num_boost_round=n_rounds)
    dval = lgb.Dataset(valid[0], valid[1], reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=n_rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(early_stopping, verbose=False)])


def run_split(ds: dict, train_chip: np.ndarray, val_chip: np.ndarray, params: dict, cols: np.ndarray,
              train_rows: np.ndarray, n_rounds: int = 2000):
    """train_chip/val_chip — булевы маски по чипам; train_rows — доп. фильтр строк обучения.
    Валидация (и ранняя остановка) — всегда на кандидатах: только к ним модель применяется на инференсе."""
    tr = train_chip[ds["chip_idx"]] & train_rows
    va = val_chip[ds["chip_idx"]] & ds["is_cand"]
    names = ds["feature_names"][cols]
    booster = fit(ds["X"][tr][:, cols], ds["y"][tr], names, params, n_rounds,
                  valid=(ds["X"][va][:, cols], ds["y"][va]))
    return booster, va, booster.predict(ds["X"][va][:, cols], num_iteration=booster.best_iteration)


def run_cv(ds: dict, folds: np.ndarray, params: dict | None = None, cols: np.ndarray | None = None,
           train_rows: np.ndarray | None = None, n_rounds: int = 2000):
    """5-fold CV → (oof-вероятности по строкам [NaN вне кандидатов], бустеры, best_iterations)."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    cols = np.arange(ds["X"].shape[1]) if cols is None else cols
    train_rows = np.ones(len(ds["y"]), bool) if train_rows is None else train_rows
    oof = np.full(len(ds["y"]), np.nan, np.float32)
    boosters = []
    for f in sorted(np.unique(folds)):
        booster, va, p = run_split(ds, folds != f, folds == f, params, cols, train_rows, n_rounds)
        oof[va] = p
        boosters.append(booster)
    return oof, boosters, [b.best_iteration for b in boosters]


def score_rows(ds: dict, prob: np.ndarray, thr, chip_sel: np.ndarray | None = None, row_sel: np.ndarray | None = None,
               n_fire_total: int | None = None) -> dict:
    """Микро-F1: строки вне кандидатов (prob = NaN) считаются предсказанными 0; FN = весь огонь − TP.
    `thr` — скаляр или массив порогов по строкам."""
    sel = np.isfinite(prob)
    if chip_sel is not None:
        sel &= chip_sel[ds["chip_idx"]]
    if row_sel is not None:
        sel &= row_sel
    pred = np.zeros(len(prob), bool)
    pred[sel] = prob[sel] > (thr[sel] if isinstance(thr, np.ndarray) else thr)
    y = ds["y"].astype(bool)
    tp = (pred & y).sum()
    if n_fire_total is None:
        n_fire_total = ds["n_fire"].sum() if chip_sel is None else ds["n_fire"][chip_sel].sum()
    return f1_counts(tp, (pred & ~y).sum(), n_fire_total - tp)


def best_threshold(ds: dict, prob: np.ndarray, chip_sel: np.ndarray | None = None,
                   row_sel: np.ndarray | None = None, n_fire_total: int | None = None) -> tuple[float, dict]:
    res = [(score_rows(ds, prob, float(t), chip_sel, row_sel, n_fire_total), float(t)) for t in THRESHOLDS]
    r, t = max(res, key=lambda a: a[0]["f1"])
    return t, r
