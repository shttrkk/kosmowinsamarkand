"""Стекинг (основная модель BS): UNet выбирает РЕГИОН гари, LightGBM 2-го уровня уточняет пиксели.

EDA показал, что эталон попиксельно почти детерминирован (пороги по медианному dNBR 3×3, свои для каждого
landcover, + маски SCL), а неопределённость — в том, какие «пятна» dNBR относятся к пожару. UNet хорошо
решает второе (контекст, текстура), но размывает границы и путает соседние степени; бустинг по пиксельным
признакам + вероятностям UNet восстанавливает точные границы и степени.

    python -m firemon.models.bs.stack --stage cv   --unet r34_cedice     # OOF group + temporal
    python -m firemon.models.bs.stack --stage full --unet r34_cedice     # weights/bs_stack.txt

LGBM-2 обучается и применяется только на «воротах» UNet: p_burn > GATE (остальное = фон) — это ~10–20 %
пикселей, поэтому инференс быстрый. Для обучения берутся OOF-вероятности UNet (без утечки).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ...features.bs_features import FEATURE_GROUPS, _box_mean, _gauss, compute_bs_features, feature_names
from ...utils import REPO_ROOT, get_logger, seed_everything
from .common import cache_dir, weights_dir, decode_probs, evaluate_masks, fit_decode, fmt, get_splits, load_chips
from .lgbm import PARAMS, fit

log = get_logger("bs.stack")
OOF = cache_dir() / "bs_oof"
GATE = 0.03
GATE_RULE = [False]  # эксперимент: ворота = p_burn > gate ИЛИ пиксельное правило эталона даёт гарь (rule_sev > 0)
OPTIONAL_GROUPS = ("med2", "scene")  # подключаются только флагом --extra-groups (эксперименты), в финальной модели их нет
STACK_GROUPS = [g for g in FEATURE_GROUPS if g != "lcrel" and g not in OPTIONAL_GROUPS]
UNET_FEATS = ["unet_p0", "unet_p1", "unet_p2", "unet_p3", "unet_pburn_m5", "unet_pburn_m11", "unet_sev_m5",
              "unet_pburn_g20", "unet_pburn_g40", "dist_event"]
N_UNET_FEATS = len(UNET_FEATS)
STACK_PARAMS = dict(PARAMS, num_leaves=63, learning_rate=0.05, min_data_in_leaf=300)


AUX_FEATS = ["aux_p", "aux_m5", "aux_g10", "aux_g30"]  # на каждый доп. канал вероятностей (напр. «зона события»)


def stack_feature_names(n_aux: int = 0) -> list[str]:
    return feature_names(STACK_GROUPS) + UNET_FEATS + [f"{n}_{k}" for k in range(n_aux) for n in AUX_FEATS]


def aux_features(aux: np.ndarray) -> np.ndarray:
    """aux (K,H,W) — дополнительные каналы вероятностей (5-й и далее каналы файла OOF) → (4K,H,W)."""
    out = []
    for a in aux:
        out += [a, _box_mean(a, 5), _gauss(a, 10.0), _gauss(a, 30.0)]
    return np.stack(out)


def unet_features(p: np.ndarray) -> np.ndarray:
    """p (4,H,W) float32 → (10,H,W): вероятности, сглаженные p_burn / ожидаемая степень на разных масштабах и
    расстояние до оценённого центра события (пик ожидаемой степени, сглаженной гауссианом σ=40 px ≈ 0.8 км):
    эталон размечает только «свой» пожар внутри буфера события, поэтому удалённые пятна — чаще фон."""
    pb = 1.0 - p[0]
    sev = p[1] + 2.0 * p[2] + 3.0 * p[3]
    g40 = _gauss(sev, 40.0)
    cy, cx = np.unravel_index(int(g40.argmax()), g40.shape)
    yy, xx = np.ogrid[:p.shape[1], :p.shape[2]]
    dist = (np.hypot(yy - cy, xx - cx) / 256.0).astype(np.float32)
    return np.stack([p[0], p[1], p[2], p[3], _box_mean(pb, 5), _box_mean(pb, 11), _box_mean(sev, 5),
                     _gauss(pb, 20.0), _gauss(pb, 40.0), dist])


def stack_matrix(chip: dict, p_unet: np.ndarray, gate: float = GATE):
    """→ (X (n,F) float32, sel (H,W) bool) — строки только для пикселей-«ворот»."""
    p = np.asarray(p_unet, np.float32)
    p, aux = p[:4], p[4:]
    sel = (1.0 - p[0]) > gate
    feats = None
    if GATE_RULE[0]:
        feats = compute_bs_features(chip, groups=STACK_GROUPS)
        sel = sel | (feats[feature_names(STACK_GROUPS).index("rule_sev")] > 0)
    if not sel.any():
        return np.zeros((0, len(stack_feature_names(len(aux)))), np.float32), sel
    if feats is None:
        feats = compute_bs_features(chip, groups=STACK_GROUPS)
    parts = [feats[:, sel], unet_features(p)[:, sel]]
    if len(aux):
        parts.append(aux_features(aux)[:, sel])
    X = np.concatenate(parts, axis=0).T
    return np.ascontiguousarray(X, dtype=np.float32), sel


def predict_stack(booster, chip: dict, p_unet: np.ndarray, gate: float = GATE, num_threads: int = 16) -> np.ndarray:
    """→ вероятности (4,H,W) float32; вне ворот — фон."""
    X, sel = stack_matrix(chip, p_unet, gate)
    out = np.zeros((4,) + sel.shape, np.float32)
    out[0] = 1.0
    if len(X):
        out[:, sel] = booster.predict(X, num_threads=num_threads).T
    return out


def rows_per_chip(chips, probs, idx, frac: float, seed: int = 42) -> list:
    """Обучающие строки 2-го уровня: доля `frac` пикселей-ворот каждого чипа → [(X, y), ...]."""
    def work(i):
        X, sel = stack_matrix(chips[i], probs[i])
        y = chips[i]["mask"][sel]
        keep = np.random.RandomState(seed + int(i)).random_sample(len(y)) < frac
        return X[keep], y[keep]

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(work, idx))


def build_rows(chips, probs, idx, frac: float, seed: int = 42):
    parts = rows_per_chip(chips, probs, idx, frac, seed)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _fit_any(X, y, n_rounds, valid, ordinal: bool):
    """multiclass-бустер или (ordinal) три бинарных P(sev ≥ k), k = 1..3."""
    if not ordinal:
        return fit(X, y, None, STACK_PARAMS, n_rounds, valid=valid)
    import lightgbm as lgb

    params = {k: v for k, v in STACK_PARAMS.items() if k != "num_class"}
    params["objective"] = "binary"
    out = []
    for k in (1, 2, 3):
        dtr = lgb.Dataset(X, (y >= k).astype(np.int32), free_raw_data=False)
        dva = lgb.Dataset(valid[0], (valid[1] >= k).astype(np.int32), reference=dtr)
        out.append(lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                             callbacks=[lgb.early_stopping(30, verbose=False)]))
    return out


def _predict_any(model, X) -> np.ndarray:
    """→ (4, n): для ordinal — вероятности классов из монотонизированных P(≥k)."""
    if not isinstance(model, list):
        return model.predict(X, num_threads=PARAMS["num_threads"]).T
    c = np.stack([b.predict(X, num_threads=PARAMS["num_threads"]) for b in model])
    c = np.minimum.accumulate(c, axis=0)
    return np.stack([1 - c[0], c[0] - c[1], c[1] - c[2], c[2]])


def run_cv(chips, pu: dict, tag: str, n_rounds: int, frac: float, drop_unet: bool = False,
           drop_feats: list[str] | None = None, ordinal: bool = False) -> dict:
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    splits = get_splits(ids)
    names = stack_feature_names(pu["group"].shape[1] - 4)
    drop = set(drop_feats or []) | (set(UNET_FEATS) if drop_unet else set())
    cols = [i for i, n in enumerate(names) if n not in drop]
    res: dict = {"tag": tag, "best_iters": {}, "gate": GATE, "dropped": sorted(drop)}
    t0 = time.time()
    parts = rows_per_chip(chips, pu["group"], range(len(chips)), frac)  # по групповому OOF, один раз
    Xall = np.concatenate([p[0] for p in parts])[:, cols]
    yall = np.concatenate([p[1] for p in parts])
    call = np.concatenate([np.full(len(p[1]), i, np.int32) for i, p in enumerate(parts)])
    del parts
    log.info("строки 2-го уровня: %s, классы %s (%.0f с)", Xall.shape, np.bincount(yall, minlength=4).tolist(),
             time.time() - t0)
    group_oof = None
    for split in ("group", "temporal"):
        oof = np.zeros((len(ids), 4, 512, 512), np.float16)
        oof[:, 0] = 1.0
        iters = []
        for k, (tr, va) in enumerate(splits[split]):
            t0 = time.time()
            mtr = np.isin(call, tr)
            if split == "group":
                mva = np.isin(call, va)
                Xva, yva = Xall[mva], yall[mva]
            else:  # val-2024: вероятности UNet, обученного на ≤2023
                Xva, yva = build_rows(chips, pu["temporal"], va, frac)
                Xva = Xva[:, cols]
            booster = _fit_any(Xall[mtr], yall[mtr], n_rounds, (Xva, yva, None), ordinal)
            iters.append([int(b.best_iteration or n_rounds) for b in booster] if ordinal
                         else int(booster.best_iteration or n_rounds))
            for i in va:
                X, sel = stack_matrix(chips[i], pu[split][i])
                if len(X):
                    oof[i][:, sel] = _predict_any(booster, X[:, cols]).astype(np.float16)
            log.info("%s %s%d: %d строк, best_iter %s, %.0f с", tag, split, k, mtr.sum(), iters[-1], time.time() - t0)
        val = np.concatenate([va for _, va in splits[split]])
        gt_d = {i: gts[i] for i in val}
        m_arg = evaluate_masks({i: decode_probs(oof[i]) for i in val}, gt_d)
        if split == "group":
            group_oof = oof
            dec = fit_decode(oof[val], gts[val])
        else:
            tr_idx = splits["temporal"][0][0]
            dec = fit_decode(group_oof[tr_idx], gts[tr_idx])
        m_tun = evaluate_masks({i: decode_probs(oof[i], dec["t_burn"], dec["w"]) for i in val}, gt_d)
        res[split] = {"argmax": m_arg, "tuned": m_tun, "decode": dec}
        res["best_iters"][split] = iters
        log.info("%s %s argmax: %s", tag, split, fmt(m_arg))
        log.info("%s %s tuned:  %s | %s", tag, split, fmt(m_tun), dec)
        np.save(OOF / f"stack_{tag}_{split}.npy", oof)
    (OOF / f"stack_{tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def train_full(chips, pu_group, n_rounds: int, frac: float, out_path) -> None:
    X, y = build_rows(chips, pu_group, range(len(chips)), frac)
    booster = fit(X, y, None, STACK_PARAMS, n_rounds)
    booster.save_model(str(out_path))
    log.info("стек-LGBM: %d строк, %d деревьев → %s", len(y), booster.num_trees(), out_path)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "train"))
    ap.add_argument("--stage", choices=["cv", "full"], required=True)
    ap.add_argument("--unet", default="r34_cedice")
    ap.add_argument("--unet-suffix", default="_tta")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--drop-unet", action="store_true", help="абляция: без признаков UNet (только ворота)")
    ap.add_argument("--drop-feats", nargs="*", default=None, help="абляция: исключить признаки по именам")
    ap.add_argument("--gate", type=float, default=None, help="порог ворот p_burn UNet (по умолчанию GATE)")
    ap.add_argument("--leaves", type=int, default=None)
    ap.add_argument("--gate-rule", action="store_true", help="ворота: p_burn > gate ИЛИ rule_sev > 0")
    ap.add_argument("--ordinal", action="store_true", help="три бинарных стекера P(sev ≥ k) вместо multiclass")
    ap.add_argument("--extra-groups", nargs="*", default=None, help=f"доп. группы признаков из {OPTIONAL_GROUPS}")
    args = ap.parse_args(argv)
    seed_everything(42, with_torch=False)
    if args.gate is not None:  # эксперименты: функции берут порог из значений по умолчанию
        for fn in (stack_matrix, predict_stack):
            fn.__defaults__ = tuple(args.gate if isinstance(d, float) else d for d in fn.__defaults__)
    GATE_RULE[0] = bool(args.gate_rule)
    if args.leaves:
        STACK_PARAMS["num_leaves"] = args.leaves
    if args.extra_groups:
        STACK_GROUPS[:] = [g for g in FEATURE_GROUPS if g != "lcrel" and (g not in OPTIONAL_GROUPS or g in args.extra_groups)]
    chips = load_chips(args.data_dir)
    pu = {s: np.load(OOF / f"{args.unet}_{s}{args.unet_suffix}.npy", mmap_mode="r") for s in ("group", "temporal")}
    if args.stage == "cv":
        return run_cv(chips, pu, args.tag or args.unet, args.rounds, args.frac, args.drop_unet, args.drop_feats, args.ordinal)
    train_full(chips, pu["group"], args.rounds, args.frac, weights_dir() / "bs_stack.txt")
    return None


if __name__ == "__main__":
    main()
