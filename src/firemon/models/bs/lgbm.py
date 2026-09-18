"""LightGBM (multiclass, по пикселям) для BS: выборка пикселей, CV/OOF, абляции, финальная модель.

    python -m firemon.models.bs.lgbm --stage sample          # data/cache/bs_train_*.npy
    python -m firemon.models.bs.lgbm --stage cv              # OOF group + temporal → data/cache/bs_oof/lgbm_*.npy
    python -m firemon.models.bs.lgbm --stage ablate          # абляции (облегчённый режим, сетка пикселей с шагом 2)
    python -m firemon.models.bs.lgbm --stage full            # weights/bs_lgbm.txt + bs_lc_medians.json

Выборка неравномерная (гарь 25 %, граница 5 px, «ложные кандидаты» dNBR>0.1, 2 % фона), поэтому каждому
пикселю даётся вес 1/доля_отбора — вероятности остаются откалиброванными к реальным частотам классов.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ...features.bs_features import FEATURE_GROUPS, LC_CODES, compute_bs_features, feature_names
from ...utils import REPO_ROOT, get_logger, seed_everything
from .common import decode_probs, evaluate_masks, fit_decode, fmt, get_splits, load_chips

log = get_logger("bs.lgbm")
CACHE = REPO_ROOT / "data" / "cache"
BASE_GROUPS = [g for g in FEATURE_GROUPS if g != "lcrel"]  # dNBR_rel добавляется по таблице фолда
FRAC = {"burn": 0.25, "near": 0.30, "cand": 0.10, "rest": 0.02}
MINIMAL = ["dNBR", "RdNBR", "landcover"]

PARAMS = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=255,
              deterministic=True, force_row_wise=True, num_threads=16, seed=42, verbosity=-1)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    from scipy import ndimage as ndi

    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return ndi.binary_dilation(mask, structure=(yy * yy + xx * xx) <= r * r)


def sample_chip(chip: dict, i: int, seed: int = 42):
    """→ (X float32 (n, F_base), y, w, chip_idx). Стратегия отбора — см. докстринг модуля."""
    names = feature_names(BASE_GROUPS)
    feats = compute_bs_features(chip, groups=BASE_GROUPS)
    gt = chip["mask"]
    burn = (gt >= 1) & (gt <= 3)
    near = _dilate(burn, 5) & ~burn & (gt == 0)
    cand = (feats[names.index("dNBR")] > 0.1) & (gt == 0) & ~near
    rest = (gt == 0) & ~near & ~cand
    rng = np.random.RandomState(seed + i)
    sel = np.zeros(gt.shape, bool)
    w = np.zeros(gt.shape, np.float32)
    for key, m in (("burn", burn), ("near", near), ("cand", cand), ("rest", rest)):
        pick = m & (rng.random_sample(gt.shape) < FRAC[key])
        sel |= pick
        w[pick] = 1.0 / FRAC[key]
    return feats[:, sel].T.copy(), gt[sel], w[sel], np.full(int(sel.sum()), i, np.int32)


def build_sample(data_dir: str | Path, force: bool = False):
    paths = {k: CACHE / f"bs_train_{k}.npy" for k in ("X", "y", "w", "chip")}
    if not force and all(p.exists() for p in paths.values()):
        return tuple(np.load(paths[k]) for k in ("X", "y", "w", "chip"))
    CACHE.mkdir(parents=True, exist_ok=True)
    chips = load_chips(data_dir)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda a: sample_chip(a[1], a[0]), enumerate(chips)))
    out = tuple(np.concatenate([p[j] for p in parts]) for j in range(4))
    for k, arr in zip(("X", "y", "w", "chip"), out):
        np.save(paths[k], arr)
    log.info("выборка: %s, классы %s, %.0f с", out[0].shape, np.bincount(out[1]).tolist(), time.time() - t0)
    return out


def lc_medians_from_sample(X: np.ndarray, y: np.ndarray, names: list[str]) -> dict[int, float]:
    """Медиана dNBR внутри гари по каждому landcover (только по обучающим пикселям)."""
    dn, lc = X[:, names.index("dNBR")], X[:, names.index("landcover")]
    out = {}
    for c in LC_CODES:
        sel = (lc == c) & (y > 0)
        out[c] = float(np.median(dn[sel])) if sel.sum() >= 500 else 0.25
    return out


def add_lcrel(X: np.ndarray, names: list[str], med: dict[int, float]) -> np.ndarray:
    lut = np.full(256, 0.25, np.float32)
    for c, v in med.items():
        lut[int(c)] = max(float(v), 0.02)
    lc = np.clip(X[:, names.index("landcover")].astype(np.int64), 0, 255)
    return np.concatenate([X, (X[:, names.index("dNBR")] / lut[lc])[:, None]], axis=1)


def select_columns(all_names: list[str], drop_groups: list[str] | None, only: list[str] | None) -> list[int]:
    if only:
        return [all_names.index(n) for n in only]
    drop = {n for g in (drop_groups or []) for n in FEATURE_GROUPS[g]}
    if drop_groups and "lc" in drop_groups:
        drop |= set(FEATURE_GROUPS["lcrel"])  # dNBR_rel — тоже информация о покрове
    return [i for i, n in enumerate(all_names) if n not in drop]


def fit(X, y, w, params: dict, n_rounds: int, valid=None, regression: bool = False):
    import lightgbm as lgb

    p = dict(params)
    if regression:
        p.update(objective="regression")
        p.pop("num_class")
    dtr = lgb.Dataset(X, y.astype(np.float32 if regression else np.int32), weight=w, free_raw_data=False)
    cbs, vsets = [], None
    if valid is not None:
        vsets = [lgb.Dataset(valid[0], valid[1].astype(np.float32 if regression else np.int32), weight=valid[2],
                             reference=dtr)]
        cbs = [lgb.early_stopping(30, verbose=False)]
    return lgb.train(p, dtr, num_boost_round=n_rounds, valid_sets=vsets, callbacks=cbs)


def predict_chip(booster, chip: dict, med: dict[int, float], cols: list[int], stride: int = 1,
                 regression: bool = False) -> np.ndarray:
    """→ вероятности (4, H/stride, W/stride) float16 (для регрессии — (1, h, w): непрерывная степень)."""
    feats = compute_bs_features(chip, lc_medians=med)[:, ::stride, ::stride]
    h, w = feats.shape[1:]
    X = feats.reshape(feats.shape[0], -1).T[:, cols]
    p = booster.predict(X, num_threads=PARAMS["num_threads"])
    p = p.reshape(h, w, -1) if p.ndim == 2 else p.reshape(h, w, 1)
    return np.moveaxis(p, 2, 0).astype(np.float16)


def fit_cuts(vals: np.ndarray, gts: np.ndarray) -> list[float]:
    """Регрессионный вариант: 3 порога по непрерывной степени, координатный спуск по BS-части."""
    v, g = vals.reshape(-1), gts.reshape(-1)
    keep = (v > 0.1) | (g > 0)
    v, g = v[keep].astype(np.float32), g[keep]
    cuts = [0.5, 1.5, 2.5]

    def obj(c):
        pred = np.digitize(v, c)
        ious = [np.count_nonzero((pred == k) & (g == k)) / max(np.count_nonzero((pred == k) | (g == k)), 1)
                for k in (1, 2, 3)]
        burn = np.count_nonzero((pred > 0) & (g > 0)) / max(np.count_nonzero((pred > 0) | (g > 0)), 1)
        return 0.35 * burn + 0.30 * float(np.mean(ious))

    best = obj(cuts)
    for _ in range(2):
        for j in range(3):
            for cand in np.arange(cuts[j] - 0.3, cuts[j] + 0.31, 0.05):
                c = list(cuts)
                c[j] = float(cand)
                if c == sorted(c) and (s := obj(c)) > best:
                    best, cuts = s, c
    return cuts


def run_cv(chips, sample, tag: str, params: dict, n_rounds: int, drop=None, only=None, stride: int = 1,
           regression: bool = False, max_rows: int | None = None, save: bool = True,
           splits=("group", "temporal")) -> dict:
    """OOF по обоим сплитам. Возвращает метрики (argmax и подобранное декодирование) + лучшие итерации."""
    X0, y, w, cidx = sample
    base_names = feature_names(BASE_GROUPS)
    all_names = base_names + FEATURE_GROUPS["lcrel"]
    cols = select_columns(all_names, drop, only)
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"][::stride, ::stride] for c in chips])
    res: dict = {"tag": tag, "features": len(cols), "best_iters": {}}
    for split in splits:
        folds = get_splits(ids)[split]
        oof = np.zeros((len(ids), 1 if regression else 4) + gts.shape[1:], np.float16)
        done = np.zeros(len(ids), bool)
        iters = []
        for k, (tr, va) in enumerate(folds):
            t0 = time.time()
            mtr, mva = np.isin(cidx, tr), np.isin(cidx, va)
            if max_rows and mtr.sum() > max_rows:
                rng = np.random.RandomState(42 + k)
                mtr &= rng.random_sample(len(mtr)) < max_rows / mtr.sum()
            med = lc_medians_from_sample(X0[mtr], y[mtr], base_names)
            Xtr = add_lcrel(X0[mtr], base_names, med)[:, cols]
            Xva = add_lcrel(X0[mva], base_names, med)[:, cols]
            booster = fit(Xtr, y[mtr], w[mtr], params, n_rounds, valid=(Xva, y[mva], w[mva]), regression=regression)
            iters.append(int(booster.best_iteration or n_rounds))
            for i in va:
                oof[i] = predict_chip(booster, chips[i], med, cols, stride, regression)
            done[va] = True
            log.info("%s %s%d: %d строк, best_iter %d, %.0f с", tag, split, k, mtr.sum(), iters[-1], time.time() - t0)
        val = np.nonzero(done)[0]
        gt_d = {i: gts[i] for i in val}
        if regression:
            fit_idx = val if split == "group" else np.array([], int)
            cuts = fit_cuts(oof[val, 0], gts[val]) if split == "group" else res["group"]["cuts"]
            m = evaluate_masks({i: np.digitize(oof[i, 0].astype(np.float32), cuts).astype(np.uint8) for i in val}, gt_d)
            res[split] = {"tuned": m, "cuts": cuts}
        else:
            m_arg = evaluate_masks({i: decode_probs(oof[i]) for i in val}, gt_d)
            # временной сплит: декодирование подбирается на групповом OOF чипов НЕ из 2024 (честно)
            if split == "group":
                dec = fit_decode(oof[val], gts[val])
                res["_group_oof"] = oof if "temporal" in splits else None
            else:
                tr_idx = folds[0][0]
                dec = fit_decode(res["_group_oof"][tr_idx], gts[tr_idx]) if res.get("_group_oof") is not None \
                    else fit_decode(oof[val], gts[val])
            m_tun = evaluate_masks({i: decode_probs(oof[i], dec["t_burn"], dec["w"]) for i in val}, gt_d)
            res[split] = {"argmax": m_arg, "tuned": m_tun, "decode": dec}
            log.info("%s %s argmax: %s", tag, split, fmt(m_arg))
        log.info("%s %s tuned:  %s", tag, split, fmt(res[split]["tuned"]))
        res["best_iters"][split] = iters
        if save and stride == 1:
            (CACHE / "bs_oof").mkdir(parents=True, exist_ok=True)
            np.save(CACHE / "bs_oof" / f"lgbm_{tag}_{split}.npy", oof)
    res.pop("_group_oof", None)
    (CACHE / "bs_oof").mkdir(parents=True, exist_ok=True)
    (CACHE / "bs_oof" / f"lgbm_{tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


ABLATIONS = {
    "abl_full": {},
    "abl_no_sar": {"drop": ["sar"]},
    "abl_no_scl": {"drop": ["scl"]},
    "abl_no_lc": {"drop": ["lc"]},
    "abl_no_ctx": {"drop": ["ctx"]},
    "abl_minimal": {"only": MINIMAL},
    "abl_regression": {"regression": True},
}


def train_full(sample, n_rounds: int, params: dict, weights_dir: Path) -> None:
    X0, y, w, _ = sample
    base_names = feature_names(BASE_GROUPS)
    med = lc_medians_from_sample(X0, y, base_names)
    booster = fit(add_lcrel(X0, base_names, med), y, w, params, n_rounds)
    booster.save_model(str(weights_dir / "bs_lgbm.txt"))
    (weights_dir / "bs_lc_medians.json").write_text(json.dumps({str(k): v for k, v in med.items()}, indent=1),
                                                    encoding="utf-8")
    log.info("финальная LGBM: %d деревьев → %s", booster.num_trees(), weights_dir / "bs_lgbm.txt")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "train"))
    ap.add_argument("--stage", choices=["sample", "cv", "ablate", "full"], required=True)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--balanced", action="store_true", help="веса классов по обратной частоте вместо 1/доля_отбора")
    ap.add_argument("--only", nargs="*", default=None, help="абляции: только эти варианты")
    args = ap.parse_args(argv)
    seed_everything(42, with_torch=False)

    sample = build_sample(args.data_dir, force=args.stage == "sample")
    if args.stage == "sample":
        return None
    params = dict(PARAMS, num_leaves=args.leaves)
    if args.balanced:
        X0, y, w, c = sample
        freq = np.bincount(y, weights=w, minlength=4)
        sample = (X0, y, (w * (freq.sum() / (4 * freq))[y]).astype(np.float32), c)
    if args.stage == "full":
        train_full(sample, args.rounds, params, REPO_ROOT / "weights")
        return None
    chips = load_chips(args.data_dir)
    if args.stage == "cv":
        return run_cv(chips, sample, args.tag, params, args.rounds)
    out = {}
    light = dict(params, learning_rate=0.1, num_leaves=63)
    for name, kw in ABLATIONS.items():
        if args.only and name not in args.only:
            continue
        out[name] = run_cv(chips, sample, name, light, 150, stride=2, max_rows=1_000_000, save=False, **kw)
    return out


if __name__ == "__main__":
    main()
