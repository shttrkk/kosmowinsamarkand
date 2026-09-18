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
from .common import cache_dir, weights_dir, decode_probs, evaluate_masks, fit_decode, fmt, get_splits, load_chips
from .postprocess import postprocess_bs

log = get_logger("bs.ensemble")
OOF = cache_dir() / "bs_oof"

POST_STEPS = [("scl_post_zero", True), ("scl_zero", True), ("water_lc_zero", True), ("builtup_zero", True),
              ("cloud_policy", "fill"), ("cloud_policy", "zero"),  # (а) «как есть» — текущее состояние
              # буфер события: компоненты, пересекающие круг R вокруг оценённого центра / обнуление дальше R
              ("event_keep_r", 150), ("event_keep_r", 200), ("event_keep_r", 250),
              ("event_zero_r", 350), ("event_zero_r", 300), ("event_zero_r", 250),
              # EDA B4c: компоненты относительно крупнейшей
              ("largest_keep_r", 280), ("largest_keep_r", 200), ("rel_area_pct", 5), ("rel_area_pct", 10),
              ("rel_area_pct", 20), ("largest_circle_r", 280),
              ("min_component_px", 25), ("min_component_px", 50), ("min_component_px", 100),  # MMU эталона = 25 px
              ("fill_holes_px", 8), ("fill_holes_px", 32), ("median_size", 3)]


def blend(pa, pb, w_a: float, idx) -> np.ndarray:
    if pb is None or w_a >= 1.0:
        return np.asarray(pa[idx], np.float32)
    if w_a <= 0.0:
        return np.asarray(pb[idx], np.float32)
    return w_a * np.asarray(pa[idx], np.float32) + (1.0 - w_a) * np.asarray(pb[idx], np.float32)


def evaluate_variant(pa: dict, pb: dict | None, w_a: float, gts, splits, chips=None, post=None,
                     decs: dict | None = None) -> dict:
    """pa/pb — {'group': oof, 'temporal': oof}. Метрики обоих сплитов (+ подмножество облачных чипов
    cloud_frac > 0.25 по групповому OOF, если переданы chips) + параметры декодирования.
    decs — уже подобранное декодирование {'group': …, 'temporal': …} (постобработка его не меняет)."""
    out = {}
    all_idx = np.arange(len(gts))
    t_tr, t_va = splits["temporal"][0]
    for split, fit_idx, val_idx in (("group", all_idx, all_idx), ("temporal", t_tr, t_va)):
        dec = decs[split] if decs else fit_decode(blend(pa["group"], pb and pb["group"], w_a, fit_idx), gts[fit_idx])
        pv = blend(pa[split], pb and pb[split], w_a, val_idx)
        preds = {}
        for j, i in enumerate(val_idx):
            m = decode_probs(pv[j], dec["t_burn"], dec["w"])
            preds[i] = postprocess_bs(m, chips[i], post, probs=pv[j], decode=dec) if post else m
        out[split] = evaluate_masks(preds, {i: gts[i] for i in val_idx})
        out[f"decode_{split}"] = dec
        if split == "group" and chips is not None:
            cl = [i for i in val_idx if float(chips[i]["meta"].get("cloud_frac") or 0) > 0.25]
            out["cloudy"] = evaluate_masks({i: preds[i] for i in cl}, {i: gts[i] for i in cl})
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
    ap.add_argument("--skip-weights", action="store_true", help="пропустить перебор весов ансамбля UNet+LGBM")
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
    grid = [1.0] if pl is None or args.skip_weights else [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
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
    cur = evaluate_variant(pa, pb, w, gts, splits, chips)
    decs = {"group": cur["decode_group"], "temporal": cur["decode_temporal"]}
    res["final_raw"] = cur
    show("финальная модель (raw)", cur)
    post: dict = {}
    raw = cur
    for key, val in POST_STEPS:  # каждый шаг отдельно от «сырого» выхода — для таблицы вкладов в отчёте
        r = evaluate_variant(pa, pb, w, gts, splits, chips, {key: val}, decs)
        res.setdefault("post_alone", {})[f"{key}={val}"] = {
            s: r[s]["bs_part"] - raw[s]["bs_part"] for s in ("group", "temporal", "cloudy")}
        res["post_alone"][f"{key}={val}"].update({f"{s}_metrics": r[s] for s in ("group", "temporal", "cloudy")})
    for key, val in POST_STEPS:
        trial = {**post, key: val}
        r = evaluate_variant(pa, pb, w, gts, splits, chips, trial, decs)
        dg = r["group"]["bs_part"] - cur["group"]["bs_part"]
        dt = r["temporal"]["bs_part"] - cur["temporal"]["bs_part"]
        ok = dg > 0 and dt > 0
        dc = r["cloudy"]["bs_part"] - cur["cloudy"]["bs_part"]
        res["post"][f"{key}={val}"] = {"group": r["group"], "temporal": r["temporal"], "cloudy": r["cloudy"],
                                       "d_group": dg, "d_temporal": dt, "d_cloudy": dc, "accepted": ok}
        log.info("post %-22s Δgroup %+.4f Δtemporal %+.4f Δоблачные(>0.25) %+.4f → %s", f"{key}={val}", dg, dt, dc,
                 "принят" if ok else "нет")
        if ok:
            post, cur = trial, r
    res["postprocess"], res["final"], res["final_decode"] = post, cur, cur["decode_group"]
    show("ИТОГ", cur)
    name = f"ensemble_{args.unet}_{args.lgbm}_{args.stack}.json"
    (OOF / name).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if not args.no_write:
        dec = {"decode": {"t_burn": cur["decode_group"]["t_burn"], "w": cur["decode_group"]["w"]},
               # nonempty_min_px — структурный приор набора (в каждом BS-чипе есть пожар), по OOF не подбирается
               "postprocess": {**post, "nonempty_min_px": 25}, "ens_weights": {"unet": best_w, "lgbm": round(1.0 - best_w, 3)}}
        (weights_dir() / "bs_decode.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
    log.info("→ %s | decode %s | post %s", name, res["final_decode"], post)
    return res


if __name__ == "__main__":
    main()
