"""Шаг 1–2: классификатор компонент (LightGBM, групповые фолды по чипам, seed 42) и подбор порога keep.

Варианты: A — классификатор ВМЕСТО геометрического правила; B — классификатор ПОВЕРХ правила event_zero_r
(компоненты берутся из маски, уже обрезанной правилом). Порог t подбирается по BS-части на групповом OOF
(для temporal — на групповом OOF чипов не из 2024 и применяется к val-2024 без изменений).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bs_components import candidate, evaluate_sets, finish, load_setup, split_post
from firemon.models.bs.common import fmt
from firemon.models.bs.components import FEATURE_NAMES, component_features
from firemon.models.bs.postprocess import postprocess_bs
from firemon.models.bs.common import decode_probs

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, deterministic=True, force_row_wise=True,
              num_threads=8, seed=42, verbosity=-1)
T_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def build(chips, probs, idx, dec, pre):
    """Для каждого чипа: кандидат, разметка компонент, признаки, метки (доля эталона > 0.5), площади."""
    def work(i):
        p = np.asarray(probs[i], np.float32)
        cand = candidate(chips[i], p, dec, pre)
        lab, n, X = component_features(cand, p, chips[i])
        gt = chips[i]["mask"] > 0
        frac = np.zeros(n)
        if n:
            size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            inside = np.bincount(lab.ravel(), weights=gt.ravel().astype(float), minlength=n + 1)[1:]
            frac = inside / np.maximum(size, 1)
        return i, cand, lab, X, frac

    with ThreadPoolExecutor(8) as ex:
        return {r[0]: r[1:] for r in ex.map(work, idx)}


def fit_predict(data_tr: dict, tr_idx, data_va: dict, va_idx, n_rounds=300):
    import lightgbm as lgb

    Xtr = np.concatenate([data_tr[i][2] for i in tr_idx])
    ytr = np.concatenate([data_tr[i][3] > 0.5 for i in tr_idx]).astype(int)
    wtr = Xtr[:, FEATURE_NAMES.index("area")] ** float(__import__("os").environ.get("BS_COMP_WPOW", "0.5"))  # метрика пиксельная → крупные компоненты важнее
    booster = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=n_rounds)
    return {i: (booster.predict(data_va[i][2]) if len(data_va[i][2]) else np.zeros(0)) for i in va_idx}, booster


def apply(data, keep_p, idx, chips, probs, dec, rest, t):
    preds = {}
    for i in idx:
        cand, lab, X, _ = data[i]
        if len(X):
            keep = np.concatenate([[False], keep_p[i] >= t])
            cand = np.where(keep[lab], cand, 0).astype(np.uint8)
        p = np.asarray(probs[i], np.float32)
        preds[i] = finish(cand, chips[i], p, dec, rest)
    return preds


def run_variant(name, pre, chips, ids, splits, probs, decs, rest) -> dict:
    t0 = time.time()
    n = len(ids)
    all_idx = np.arange(n)
    t_tr, t_va = splits["temporal"][0]
    dg = build(chips, probs["group"], all_idx, decs["group"], pre)
    dt = build(chips, probs["temporal"], t_va, decs["temporal"], pre)
    n_comp = sum(len(dg[i][2]) for i in all_idx)
    pos = sum(int((dg[i][3] > 0.5).sum()) for i in all_idx)
    print(f"[{name}] компонент {n_comp}, «своих» {pos} ({time.time() - t0:.0f} с)")
    # групповой OOF
    keep_g = {}
    for tr, va in splits["group"]:
        pr, _ = fit_predict(dg, tr, dg, va)
        keep_g.update(pr)
    # temporal: обучение на компонентах чипов ≤2023 (из группового OOF стека), применение к val-2024
    keep_t, booster_t = fit_predict(dg, t_tr, dt, t_va)
    from sklearn.metrics import roc_auc_score

    yg = np.concatenate([dg[i][3] > 0.5 for i in all_idx])
    auc = float(roc_auc_score(yg, np.concatenate([keep_g[i] for i in all_idx])))
    res = {"n_components": n_comp, "n_positive": pos, "auc_group_oof": auc, "thresholds": {}}

    def bs_at(t, idx, data, keep, split):
        from firemon.models.bs.common import evaluate_masks
        preds = apply(data, keep, idx, chips, probs[split], decs[split], rest, t)
        return evaluate_masks(preds, {i: chips[i]["mask"] for i in idx})["bs_part"]

    grid_all = {float(t): bs_at(t, all_idx, dg, keep_g, "group") for t in T_GRID}
    grid_tr = {float(t): bs_at(t, t_tr, dg, keep_g, "group") for t in T_GRID}
    t_group = max(grid_all, key=grid_all.get)
    t_temp = max(grid_tr, key=grid_tr.get)
    res["thresholds"] = {"group": t_group, "temporal": t_temp, "grid_group": grid_all}
    preds = {"group": apply(dg, keep_g, all_idx, chips, probs["group"], decs["group"], rest, t_group),
             "temporal": apply(dt, keep_t, t_va, chips, probs["temporal"], decs["temporal"], rest, t_temp)}
    res["metrics"] = evaluate_sets(preds, chips, {"group": all_idx, "temporal": t_va})
    imp = sorted(zip(booster_t.feature_importance("gain"), FEATURE_NAMES), reverse=True)[:10]
    res["top_features_gain"] = [(n_, float(g)) for g, n_ in imp]
    print(f"[{name}] AUC {auc:.4f}, t_group {t_group}, t_temporal {t_temp}")
    for s in ("group", "temporal", "cloudy"):
        print(f"[{name}] {s:9s} {fmt(res['metrics'][s])}")
    return res


def step1(args) -> dict:
    chips, ids, splits, probs, decs, post = load_setup(args.data_dir, args.probs, args.ensemble)
    pre, rest = split_post(post)
    idx = {"group": np.arange(len(ids)), "temporal": splits["temporal"][0][1]}
    cur = {}
    for split in ("group", "temporal"):
        cur[split] = {}
        for i in idx[split]:
            p = np.asarray(probs[split][i], np.float32)
            d = decs[split]
            cur[split][i] = postprocess_bs(decode_probs(p, d["t_burn"], d["w"]), chips[i], post, probs=p, decode=d)
    out = {"current_rule": evaluate_sets(cur, chips, idx)}
    for s in ("group", "temporal", "cloudy"):
        print(f"[текущее правило] {s:9s} {fmt(out['current_rule'][s])}")
    pre_b = {**pre, **{k: v for k, v in post.items() if k.startswith("event_")}}
    for name, cfg in (("A: классификатор вместо правила", pre), ("B: классификатор поверх правила", pre_b)):
        r = run_variant(name, cfg, chips, ids, splits, probs, decs, rest)
        r["gain"] = {s: r["metrics"][s]["bs_part"] - out["current_rule"][s]["bs_part"] for s in ("group", "temporal", "cloudy")}
        r["accepted"] = bool(r["gain"]["group"] >= 0.005 and r["gain"]["temporal"] >= 0.005)
        print(f"[{name}] прирост:", {k: round(v, 4) for k, v in r["gain"].items()}, "→", "ПРИНЯТО" if r["accepted"] else "не принято")
        out[name] = r
    return out
