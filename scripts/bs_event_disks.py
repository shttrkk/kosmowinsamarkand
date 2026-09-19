"""Гипотеза «регион эталона = объединение дисков вокруг ядер сильной степени».

    python scripts/bs_event_disks.py [--stack night_avg3s075] [--decode bs_decode_ms.json]

По кэшированным OOF-вероятностям стека v5, без обучения. Ядра: (а) p3 > q, q ∈ {0.5, 0.7}; (б) компоненты «правило
эталона = 3-я степень» (медианный dNBR 3×3 выше порога сильной степени по покрову) площадью ≥ 25 px; (в) компоненты
класса 3 финальной маски v5; оракул — класс 3 ЭТАЛОНА. Регион = точки не дальше R px от ядра (диск, через EDT),
R ∈ {50, 75, 100, 150, 200}. Итог = маска v5 ∩ регион: «вместо» event_zero_r (он отключается) и «в дополнение» к нему;
остальная постобработка без изменений. Чип без ядер → регион не применяется (маска v5 как есть).
Метрики — микро: group, temporal (val 2024), cloudy (cloud_frac > 0.25). Результат: reports/bs_event_disks.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.features.bs_features import compute_bs_features, feature_names  # noqa: E402
from firemon.models.bs.common import decode_probs, fit_decode, get_splits, load_chips  # noqa: E402
from firemon.models.bs.postprocess import postprocess_bs, remove_small  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
RADII = (50, 75, 100, 150, 200)


def conf(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(gt.ravel().astype(np.int64) * 4 + pred.ravel().astype(np.int64), minlength=16).reshape(4, 4)


def metrics(c: np.ndarray) -> dict:
    c = c.astype(np.float64)
    tp = c[1:, 1:].sum()
    iou_burn = tp / max(tp + c[0, 1:].sum() + c[1:, 0].sum(), 1)
    miou = float(np.mean([c[k, k] / max(c[k, :].sum() + c[:, k].sum() - c[k, k], 1) for k in (1, 2, 3)]))
    return {"iou_burn": float(iou_burn), "miou_sev": miou, "bs_part": float(0.35 * iou_burn + 0.30 * miou)}


def chip_variants(chip: dict, probs: np.ndarray, dec: dict, post: dict) -> tuple[dict[str, np.ndarray], dict]:
    gt = chip["mask"]
    raw = decode_probs(probs, dec["t_burn"], dec["w"])
    final = postprocess_bs(raw, chip, post, probs=probs, decode=dec)
    post_no_event = {k: v for k, v in post.items() if k not in ("event_zero_r", "event_keep_r")}
    no_event = postprocess_bs(raw, chip, post_no_event, probs=probs, decode=dec)
    rule = np.rint(compute_bs_features(chip, groups=["rule"])[feature_names(["rule"]).index("rule_sev")]).astype(np.uint8)

    kernels = {
        "a p3>0.5": probs[3] > 0.5,
        "a p3>0.7": probs[3] > 0.7,
        "b rule_sev=3 (>=25px)": remove_small((rule == 3).astype(np.uint8), 25) > 0,
        "c v5 class 3": final == 3,
        "ORACLE gt class 3": gt == 3,
    }
    out = {"v5_final": final, "v5_no_event_zero": no_event}
    info = {}
    for kname, kern in kernels.items():
        info[kname] = bool(kern.any())
        dist = ndi.distance_transform_edt(~kern) if kern.any() else None
        for r in RADII:
            region = np.ones(gt.shape, bool) if dist is None else dist <= r
            # «вместо»: регион ставится на место event_zero_r — до удаления мелких компонент и заливки дыр
            out[f"{kname} | R={r} | вместо"] = postprocess_bs(np.where(region, raw, 0).astype(np.uint8), chip, post_no_event,
                                                             probs=probs, decode=dec)
            out[f"{kname} | R={r} | в дополнение"] = postprocess_bs(np.where(region, raw, 0).astype(np.uint8), chip, post,
                                                                   probs=probs, decode=dec)
    # для справки: какая доля эталонной гари лежит в пределах R от эталонного класса 3
    if (gt == 3).any():
        d = ndi.distance_transform_edt(~(gt == 3))
        info["gt_within"] = {r: [int(((gt > 0) & (d <= r)).sum()), int((gt > 0).sum())] for r in RADII}
    return out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="night_avg3s075")
    ap.add_argument("--decode", default="bs_decode_ms.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    chips = load_chips(ROOT / "data" / "train")
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    t_tr, t_va = get_splits(ids)["temporal"][0]
    pa = {s: np.load(OOF / f"stack_{args.stack}_{s}.npy", mmap_mode="r") for s in ("group", "temporal")}
    post = json.loads((ROOT / "weights" / args.decode).read_text(encoding="utf-8"))["postprocess"]
    decs = {"group": fit_decode(np.asarray(pa["group"], np.float32), gts),
            "temporal": fit_decode(np.asarray(pa["group"][np.sort(t_tr)], np.float32), gts[np.sort(t_tr)])}

    def work(job):
        split, i = job
        masks, info = chip_variants(chips[i], np.asarray(pa[split][i], np.float32), decs[split], post)
        return split, i, {k: conf(chips[i]["mask"], v) for k, v in masks.items()}, info

    acc = {"group": {}, "temporal": {}, "cloudy": {}}
    has_kernel: dict[str, int] = {}
    within = {r: [0, 0] for r in RADII}
    jobs = [("group", i) for i in range(len(chips))] + [("temporal", int(i)) for i in t_va]
    with ThreadPoolExecutor(8) as ex:
        for split, i, confs, info in ex.map(work, jobs):
            cloudy = split == "group" and float(chips[i]["meta"].get("cloud_frac") or 0) > 0.25
            for k, c in confs.items():
                acc[split][k] = acc[split].get(k, 0) + c
                if cloudy:
                    acc["cloudy"][k] = acc["cloudy"].get(k, 0) + c
            if split == "group":
                for k, v in info.items():
                    if k == "gt_within":
                        for r, (a, b) in v.items():
                            within[r][0] += a
                            within[r][1] += b
                    else:
                        has_kernel[k] = has_kernel.get(k, 0) + int(v)
    res = {s: {k: metrics(c) for k, c in d.items()} for s, d in acc.items()}
    base = {s: res[s]["v5_final"]["bs_part"] for s in res}
    rows = sorted(res["group"], key=lambda k: -min(res["group"][k]["bs_part"] - base["group"],
                                                   res["temporal"][k]["bs_part"] - base["temporal"]))
    print(f"{'вариант':<44}{'group IoU/mIoU/BS':>25}{'temporal IoU/mIoU/BS':>25}{'cloudy':>8}{'dG':>9}{'dT':>9}{'dCl':>9}")
    for k in rows:
        g, t, c = res["group"][k], res["temporal"][k], res["cloudy"][k]
        print(f"{k:<44}{g['iou_burn']:>9.4f}{g['miou_sev']:>8.4f}{g['bs_part']:>8.4f}{t['iou_burn']:>9.4f}{t['miou_sev']:>8.4f}"
              f"{t['bs_part']:>8.4f}{c['bs_part']:>8.4f}{g['bs_part'] - base['group']:>+9.4f}"
              f"{t['bs_part'] - base['temporal']:>+9.4f}{c['bs_part'] - base['cloudy']:>+9.4f}")
    print("\nчипов с ядрами (из 224):", has_kernel)
    print("доля эталонной гари в пределах R от эталонного класса 3:",
          {r: round(a / b, 4) for r, (a, b) in within.items()})
    out = {"stack": args.stack, "decode": decs, "post": post, "metrics": res, "chips_with_kernels": has_kernel,
           "gt_burn_within_R_of_gt_class3": {str(r): a / b for r, (a, b) in within.items()},
           "seconds": time.perf_counter() - t0}
    (ROOT / "reports" / "bs_event_disks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nготово за {time.perf_counter() - t0:.0f} с → reports/bs_event_disks.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
