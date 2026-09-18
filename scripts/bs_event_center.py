"""H1: рамочный приор на центр события (ветка task/bs-lovasz). По кэшированным OOF финального стека.

Сейчас центр = argmax ожидаемой степени, сглаженной гауссианом σ=40 (postprocess.event_center, mode=argmax);
варианты: clip / boxmax / cc (центроид крупнейшей компоненты p_burn внутри рамки) × R_box ∈ {170, 200, 230, 260}
× event_zero_r ∈ {300, 250}. Оба сплита + облачные чипы. → reports/bs_event_center.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from firemon.models.bs.common import decode_probs, evaluate_masks  # noqa: E402
from firemon.models.bs.postprocess import event_center, postprocess_bs  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"


def enclosing_center(mask: np.ndarray) -> tuple[float, float]:
    import cv2

    pts = np.argwhere(mask > 0).astype(np.float32)
    (cy, cx), _ = cv2.minEnclosingCircle(pts)
    return float(cy), float(cx)


def main() -> None:
    from firemon.models.bs.common import get_splits, load_chips

    ens = json.loads((OOF / "ensemble_r34_cedice_main_avg14nt_event.json").read_text(encoding="utf-8"))
    post = json.loads((ROOT / "weights" / "bs_decode.json").read_text(encoding="utf-8"))["postprocess"]
    decs = {"group": ens["final"]["decode_group"], "temporal": ens["final"]["decode_temporal"]}
    chips = load_chips(ROOT / "data" / "train")
    ids = [c["chip_id"] for c in chips]
    splits = get_splits(ids)
    probs = {s: np.load(OOF / f"stack_avg14nt_event_{s}.npy", mmap_mode="r") for s in ("group", "temporal")}
    idx = {"group": np.arange(len(ids)), "temporal": splits["temporal"][0][1]}
    cloudy = [i for i in idx["group"] if float(chips[i]["meta"].get("cloud_frac") or 0) > 0.25]
    P = {s: {i: np.asarray(probs[s][i], np.float32) for i in idx[s]} for s in idx}

    def evaluate(cfg: dict) -> dict:
        out = {}
        for s in idx:
            preds = {i: postprocess_bs(decode_probs(P[s][i], decs[s]["t_burn"], decs[s]["w"]), chips[i], cfg, probs=P[s][i],
                                       decode=decs[s]) for i in idx[s]}
            out[s] = evaluate_masks(preds, {i: chips[i]["mask"] for i in idx[s]})
            if s == "group":
                out["cloudy"] = evaluate_masks({i: preds[i] for i in cloudy}, {i: chips[i]["mask"] for i in cloudy})
        return out

    # диагностика текущей оценки центра (group OOF)
    gt_c = np.array([enclosing_center(chips[i]["mask"]) for i in idx["group"]])
    est = np.array([event_center(P["group"][i]) for i in idx["group"]])
    off_chip = np.abs(est - 255.5).max(axis=1)          # чебышёвское смещение от центра чипа (рамка)
    err = np.hypot(*(est - gt_c).T)
    diag = {"how": "argmax гауссиана σ=40 px от ожидаемой степени p1+2·p2+3·p3 (вероятности стека)",
            "est_outside_box170_frac": float((off_chip > 170).mean()), "est_outside_box230_frac": float((off_chip > 230).mean()),
            "gt_outside_box170_frac": float((np.abs(gt_c - 255.5).max(axis=1) > 170).mean()),
            "err_to_gt_center_px_q25_q50_q75_q95": [float(v) for v in np.percentile(err, [25, 50, 75, 95])],
            "err_when_est_outside_box170_median": float(np.median(err[off_chip > 170])) if (off_chip > 170).any() else None,
            "err_when_est_inside_box170_median": float(np.median(err[off_chip <= 170]))}
    print("диагностика:", json.dumps(diag, ensure_ascii=False))
    for mode in ("clip", "boxmax", "cc"):
        e = np.array([event_center(P["group"][i], 40.0, mode, 200, decs["group"]["t_burn"]) for i in idx["group"]])
        diag[f"err_median_{mode}_box200"] = float(np.median(np.hypot(*(e - gt_c).T)))
    print("медианная ошибка центра, px: argmax %.0f | clip200 %.0f | boxmax200 %.0f | cc200 %.0f" % (
        np.median(err), diag["err_median_clip_box200"], diag["err_median_boxmax_box200"], diag["err_median_cc_box200"]))

    res = {"diagnostics": diag, "variants": {}}
    base = evaluate(post)
    res["variants"]["текущий (argmax, zero_r=300)"] = base
    print(f"{'вариант':34s} | group BS (Δ) | temporal BS (Δ) | облачные BS (Δ)")
    print(f"{'текущий (argmax, zero_r=300)':34s} | {base['group']['bs_part']:.4f} | {base['temporal']['bs_part']:.4f} | {base['cloudy']['bs_part']:.4f}")
    for zero_r in (300, 250):
        for mode in ("clip", "boxmax", "cc"):
            for r_box in (170, 200, 230, 260):
                cfg = {**post, "event_zero_r": zero_r, "event_center": {"mode": mode, "r_box": r_box}}
                m = evaluate(cfg)
                name = f"{mode}, R_box={r_box}, zero_r={zero_r}"
                m["gain"] = {s: m[s]["bs_part"] - base[s]["bs_part"] for s in ("group", "temporal", "cloudy")}
                res["variants"][name] = m
                print(f"{name:34s} | {m['group']['bs_part']:.4f} ({m['gain']['group']:+.4f}) | {m['temporal']['bs_part']:.4f} "
                      f"({m['gain']['temporal']:+.4f}) | {m['cloudy']['bs_part']:.4f} ({m['gain']['cloudy']:+.4f})")
    best = max((k for k in res["variants"] if "gain" in res["variants"][k]),
               key=lambda k: min(res["variants"][k]["gain"]["group"], res["variants"][k]["gain"]["temporal"]))
    g = res["variants"][best]["gain"]
    res["best"] = best
    res["accepted"] = bool(g["group"] >= 0.005 and g["temporal"] >= 0.005)
    print("лучший:", best, {k: round(v, 4) for k, v in g.items()}, "→", "ПРИНЯТО" if res["accepted"] else "не принято")
    (ROOT / "reports" / "bs_event_center.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False),
                                                           encoding="utf-8")


if __name__ == "__main__":
    main()
