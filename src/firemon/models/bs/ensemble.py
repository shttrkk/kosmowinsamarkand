"""Ансамбль UNet+LGBM, подбор декодирования и постобработки по OOF (групповой и временной сплиты).

    python -m firemon.models.bs.ensemble --unet r34_cedice --lgbm main --stack r34_cedice

Честность временного сплита: параметры декодирования/веса ансамбля для него подбираются на групповом OOF
чипов НЕ из 2024 года и без изменений применяются к val-2024. Шаг постобработки принимается, только если
даёт прирост BS-части на ОБОИХ сплитах. Результат → data/cache/bs_oof/ensemble_<…>.json и
weights/bs_decode.json (декодирование + постобработка финальной модели; читает BSPredictor).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ...utils import REPO_ROOT, get_logger
from .common import decode_probs, evaluate_masks, fit_decode, fmt, get_splits, load_chips
from .postprocess import postprocess_bs

log = get_logger("bs.ensemble")
OOF = REPO_ROOT / "data" / "cache" / "bs_oof"

POST_STEPS = [("scl_zero", True), ("water_lc_zero", True), ("builtup_zero", True),
              ("cloud_policy", "fill"), ("cloud_policy", "zero"),  # (а) «как есть» — текущее состояние
              ("min_component_px", 4), ("min_component_px", 8), ("min_component_px", 16), ("min_component_px", 32),
              ("min_component_px", 64), ("fill_holes_px", 8), ("fill_holes_px", 32), ("fill_holes_px", 128),
              ("median_size", 3), ("median_size", 5)]


def blend(pa, pb, w_a: float, idx) -> np.ndarray:
    if pb is None or w_a >= 1.0:
        return np.asarray(pa[idx], np.float32)
    if w_a <= 0.0:
        return np.asarray(pb[idx], np.float32)
    return w_a * np.asarray(pa[idx], np.float32) + (1.0 - w_a) * np.asarray(pb[idx], np.float32)


def evaluate_variant(pa: dict, pb: dict | None, w_a: float, gts, splits, chips=None, post=None) -> dict:
    """pa/pb — {'group': oof, 'temporal': oof}. Метрики обоих сплитов + параметры декодирования."""
    out = {}
    all_idx = np.arange(len(gts))
    t_tr, t_va = splits["temporal"][0]
    for split, fit_idx, val_idx in (("group", all_idx, all_idx), ("temporal", t_tr, t_va)):
        dec = fit_decode(blend(pa["group"], pb and pb["group"], w_a, fit_idx), gts[fit_idx])
        pv = blend(pa[split], pb and pb[split], w_a, val_idx)
        preds = {}
        for j, i in enumerate(val_idx):
            m = decode_probs(pv[j], dec["t_burn"], dec["w"])
            preds[i] = postprocess_bs(m, chips[i], post) if post else m
        out[split] = evaluate_masks(preds, {i: gts[i] for i in val_idx})
        out[f"decode_{split}"] = dec
    return out


def _load_pair(prefix: str) -> dict | None:
    if not (OOF / f"{prefix}_group.npy").exists():
        return None
    return {s: np.load(OOF / f"{prefix}_{s}.npy", mmap_mode="r") for s in ("group", "temporal")}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "train"))
    ap.add_argument("--unet", default="r34_cedice")
    ap.add_argument("--lgbm", default="main")
    ap.add_argument("--stack", default=None, help="тег стека: постобработка подбирается для него (финальная модель)")
    ap.add_argument("--no-write", action="store_true", help="не перезаписывать weights/bs_decode.json")
    args = ap.parse_args(argv)

    chips = load_chips(args.data_dir)
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    splits = get_splits(ids)
    pu_plain = _load_pair(args.unet)
    pu_tta = {s: np.load(OOF / f"{args.unet}_{s}_tta.npy", mmap_mode="r") for s in ("group", "temporal")}
    pl = _load_pair(f"lgbm_{args.lgbm}")
    ps = _load_pair(f"stack_{args.stack}") if args.stack else None
    res: dict = {"unet": args.unet, "lgbm": args.lgbm, "stack": args.stack, "weights": {}, "post": {}}

    def show(name, r):
        log.info("%-26s group: %s", name, fmt(r["group"]))
        log.info("%-26s temp.: %s", name, fmt(r["temporal"]))

    res["unet_no_tta"] = evaluate_variant(pu_plain, None, 1.0, gts, splits)
    show("UNet без TTA", res["unet_no_tta"])
    if pl is not None:
        res["lgbm_only"] = evaluate_variant(pl, None, 1.0, gts, splits)
        show("LGBM", res["lgbm_only"])
    # 1) веса ансамбля UNet(TTA)+LGBM
    grid = [1.0] if pl is None else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
    for w in grid:
        res["weights"][str(w)] = r = evaluate_variant(pu_tta, pl, w, gts, splits)
        show(f"UNet+TTA w_unet={w}", r)
    ref = res["weights"]["1.0"]
    best_w = max(grid, key=lambda w: min(res["weights"][str(w)][s]["bs_part"] - ref[s]["bs_part"]
                                         for s in ("group", "temporal")))
    res["best_w_unet"] = best_w
    log.info("лучший w_unet = %s", best_w)

    # 2) постобработка финальной модели: жадно, шаг принимается при приросте на обоих сплитах
    pa, pb, w = (ps, None, 1.0) if ps is not None else (pu_tta, pl, best_w)
    cur = evaluate_variant(pa, pb, w, gts, splits)
    res["final_raw"] = cur
    show("финальная модель (raw)", cur)
    post: dict = {}
    for key, val in POST_STEPS:
        trial = {**post, key: val}
        r = evaluate_variant(pa, pb, w, gts, splits, chips, trial)
        dg = r["group"]["bs_part"] - cur["group"]["bs_part"]
        dt = r["temporal"]["bs_part"] - cur["temporal"]["bs_part"]
        ok = dg > 0 and dt > 0
        res["post"][f"{key}={val}"] = {"group": r["group"], "temporal": r["temporal"], "d_group": dg,
                                       "d_temporal": dt, "accepted": ok}
        log.info("post %-22s Δgroup %+.4f Δtemporal %+.4f → %s", f"{key}={val}", dg, dt, "принят" if ok else "нет")
        if ok:
            post, cur = trial, r
    res["postprocess"], res["final"], res["final_decode"] = post, cur, cur["decode_group"]
    show("ИТОГ", cur)
    name = f"ensemble_{args.unet}_{args.lgbm}_{args.stack}.json"
    (OOF / name).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if not args.no_write:
        dec = {"decode": {"t_burn": cur["decode_group"]["t_burn"], "w": cur["decode_group"]["w"]},
               "postprocess": post, "ens_weights": {"unet": best_w, "lgbm": round(1.0 - best_w, 3)}}
        (REPO_ROOT / "weights" / "bs_decode.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
    log.info("→ %s | decode %s | post %s", name, res["final_decode"], post)
    return res


if __name__ == "__main__":
    main()
