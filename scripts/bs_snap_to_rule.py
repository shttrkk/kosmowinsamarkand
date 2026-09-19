"""Гипотеза «snap-to-rule»: модель выбирает области, граница и степень берутся из правила эталона.

    python scripts/bs_snap_to_rule.py [--stack night_ms] [--decode bs_decode_ms.json] [--no-post-all]

Только кэшированные OOF-вероятности стека (data/cache/bs_oof/stack_<tag>_{group,temporal}.npy), ничего не обучается.
Правило: rule_sev (пороги по медианному dNBR 3×3 по покрову, запреты SCL) + вода → 0 + MMU 25 px.
Метрики — микро по пулу пикселей: group (224 чипа), temporal (val = 2024), cloudy (cloud_frac > 0.25, групповой OOF).
Результат: reports/bs_snap_to_rule.json + таблица в stdout.
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
EIGHT = np.ones((3, 3), bool)


def conf(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.bincount(gt.ravel().astype(np.int64) * 4 + pred.ravel().astype(np.int64), minlength=16).reshape(4, 4)


def metrics(c: np.ndarray) -> dict:
    c = c.astype(np.float64)
    tp_b = c[1:, 1:].sum()
    iou_burn = tp_b / max(tp_b + c[0, 1:].sum() + c[1:, 0].sum(), 1)
    ious = [c[k, k] / max(c[k, :].sum() + c[:, k].sum() - c[k, k], 1) for k in (1, 2, 3)]
    miou = float(np.mean(ious))
    return {"iou_burn": float(iou_burn), "miou_sev": miou, "bs_part": float(0.35 * iou_burn + 0.30 * miou),
            "iou": [float(x) for x in ious]}


def rule_of(chip: dict) -> np.ndarray:
    """rule_sev 0..3 (запреты SCL уже внутри) + вода → 0 + MMU 25 px."""
    sev = compute_bs_features(chip, groups=["rule"])[feature_names(["rule"]).index("rule_sev")]
    sev = np.rint(sev).astype(np.uint8)
    sev[chip["aux"][2] == 80] = 0
    return remove_small(sev, 25)


def variants_for_chip(chip: dict, probs: np.ndarray, dec: dict, post: dict, do_post: set | None) -> dict[str, np.ndarray]:
    """Все варианты масок для одного чипа: имя → маска 0..3."""
    gt = chip["mask"]
    base_raw = decode_probs(probs, dec["t_burn"], dec["w"])
    base = postprocess_bs(base_raw, chip, post, probs=probs, decode=dec)
    rule = rule_of(chip)
    rmask = rule > 0
    p_burn = 1.0 - probs[0].astype(np.float32)
    sev_model = (probs[1:] * np.asarray(dec["w"], np.float32)[:, None, None]).argmax(0).astype(np.uint8) + 1
    lab, n = ndi.label(rmask, structure=EIGHT)
    idx = np.arange(1, n + 1)
    size = ndi.sum(rmask, lab, idx) if n else np.zeros(0)

    out: dict[str, np.ndarray] = {"v5_final": base, "v5_raw": base_raw, "rule_all": rule}

    def keep(sel: np.ndarray) -> np.ndarray:  # sel — булев вектор по компонентам
        table = np.zeros(n + 1, bool)
        table[1:] = sel
        return table[lab]

    def emit(name: str, m: np.ndarray) -> None:
        for sev_name, sev in (("rule", rule), ("model", sev_model)):
            mask = np.where(m, sev, 0).astype(np.uint8)
            out[f"{name}|sev={sev_name}|post=0"] = mask
            if do_post is None or name in do_post:
                # постобработка v5 поверх: запреты, event_zero_r, MMU, дыры, приор непустого чипа
                out[f"{name}|sev={sev_name}|post=1"] = postprocess_bs(mask, chip, post, probs=probs, decode=dec)

    # 0. оракул: компоненты правила с долей эталона > 0.5
    if n:
        gt_frac = ndi.sum(gt > 0, lab, idx) / size
        out["oracle_rule_components"] = np.where(keep(gt_frac > 0.5), rule, 0).astype(np.uint8)
    else:
        out["oracle_rule_components"] = np.zeros_like(rule)
    out["oracle_rule_in_gt_region"] = np.where(rmask & ndi.binary_dilation(gt > 0, iterations=1), rule, 0).astype(np.uint8)

    # A. rule ∩ dilate(model, r)
    model_mask = base > 0
    for r in (0, 3, 7, 15):
        dil = model_mask if r == 0 else ndi.binary_dilation(model_mask, structure=EIGHT, iterations=r)
        emit(f"A r={r}", rmask & dil)

    # B/C. выбор компонент правила по вероятностям модели
    if n:
        mean_p = ndi.mean(p_burn, lab, idx)
        for t in (0.2, 0.3, 0.42):
            frac = ndi.sum(p_burn > t, lab, idx) / size
            for f in (0.3, 0.5, 0.7):
                emit(f"B t={t} f={f}", keep(frac >= f))
        for q in (0.2, 0.3, 0.42):
            emit(f"B mean>={q}", keep(mean_p >= q))
        n_hot = ndi.sum(p_burn > 0.6, lab, idx)
        frac3 = ndi.sum(p_burn > 0.3, lab, idx) / size
        for nn in (10, 25, 50):
            emit(f"C N={nn}", keep(n_hot >= nn))
            emit(f"C N={nn} | B t=0.3 f=0.5", keep((n_hot >= nn) | (frac3 >= 0.5)))
    else:
        empty = np.zeros(rmask.shape, bool)
        for t in (0.2, 0.3, 0.42):
            for f in (0.3, 0.5, 0.7):
                emit(f"B t={t} f={f}", empty)
        for q in (0.2, 0.3, 0.42):
            emit(f"B mean>={q}", empty)
        for nn in (10, 25, 50):
            emit(f"C N={nn}", empty)
            emit(f"C N={nn} | B t=0.3 f=0.5", empty)

    # диагностика ошибок финала v5 относительно компонент правила (пиксели)
    diag = np.zeros(6, np.int64)  # FP/FN: вне правила, в частично захваченной компоненте, в целой (взятой/не взятой)
    pb, gb = base > 0, gt > 0
    if n:
        cap = np.zeros(n + 1)
        cap[1:] = ndi.sum(pb, lab, idx) / size
        capmap = cap[lab]
        partial = rmask & (capmap > 0.05) & (capmap < 0.95)
        whole = rmask & ~partial
    else:
        partial = whole = np.zeros(rmask.shape, bool)
    for k, err in enumerate((pb & ~gb, ~pb & gb)):
        diag[3 * k + 0] = int((err & ~rmask).sum())
        diag[3 * k + 1] = int((err & partial).sum())
        diag[3 * k + 2] = int((err & whole).sum())
    out["__diag__"] = diag
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="night_ms", help="тег OOF стека (финал v5 — night_ms)")
    ap.add_argument("--decode", default="bs_decode_ms.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--post-for", nargs="*", default=None, help="для каких вариантов считать post=1 (по умолчанию — для всех)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    chips = load_chips(ROOT / "data" / "train")
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    splits = get_splits(ids)
    t_tr, t_va = splits["temporal"][0]
    pa = {s: np.load(OOF / f"stack_{args.stack}_{s}.npy", mmap_mode="r") for s in ("group", "temporal")}
    cfg = json.loads((ROOT / "weights" / args.decode).read_text(encoding="utf-8"))
    post = cfg["postprocess"]
    print(f"чипов {len(chips)}, temporal val {len(t_va)}, загрузка {time.perf_counter() - t0:.1f} с; post = {post}")

    # декодирование — как в ensemble.evaluate_variant: group — по всему OOF, temporal — по групповому OOF чипов не из 2024
    decs = {"group": fit_decode(np.asarray(pa["group"], np.float32), gts),
            "temporal": fit_decode(np.asarray(pa["group"][np.sort(t_tr)], np.float32), gts[np.sort(t_tr)])}
    print("decode:", decs)

    do_post = None if args.post_for is None else set(args.post_for)
    acc: dict[str, dict[str, np.ndarray]] = {"group": {}, "temporal": {}, "cloudy": {}}
    diag = {"group": np.zeros(6, np.int64), "temporal": np.zeros(6, np.int64)}

    def work(job):
        split, i = job
        probs = np.asarray(pa[split][i], np.float32)
        res = variants_for_chip(chips[i], probs, decs[split], post, do_post)
        d = res.pop("__diag__")
        return split, i, {k: conf(chips[i]["mask"], v) for k, v in res.items()}, d

    jobs = [("group", i) for i in range(len(chips))] + [("temporal", int(i)) for i in t_va]
    with ThreadPoolExecutor(args.workers) as ex:
        for split, i, confs, d in ex.map(work, jobs):
            diag[split] += d
            cloudy = split == "group" and float(chips[i]["meta"].get("cloud_frac") or 0) > 0.25
            for k, c in confs.items():
                acc[split][k] = acc[split].get(k, 0) + c
                if cloudy:
                    acc["cloudy"][k] = acc["cloudy"].get(k, 0) + c
    res = {s: {k: metrics(c) for k, c in d.items()} for s, d in acc.items()}
    base = {s: res[s]["v5_final"]["bs_part"] for s in res}

    rows = []
    for k in res["group"]:
        g, t, c = res["group"][k], res["temporal"][k], res["cloudy"][k]
        rows.append((k, g, t, c, g["bs_part"] - base["group"], t["bs_part"] - base["temporal"], c["bs_part"] - base["cloudy"]))
    rows.sort(key=lambda r: -min(r[4], r[5]))
    print(f"\n{'вариант':<46}{'group IoU/mIoU/BS':>24}{'temporal':>24}{'cloudy BS':>10}{'Δg':>9}{'Δt':>9}{'Δcl':>9}")
    for k, g, t, c, dg, dt, dc in rows:
        print(f"{k:<46}{g['iou_burn']:>8.4f}{g['miou_sev']:>8.4f}{g['bs_part']:>8.4f}{t['iou_burn']:>8.4f}{t['miou_sev']:>8.4f}"
              f"{t['bs_part']:>8.4f}{c['bs_part']:>10.4f}{dg:>+9.4f}{dt:>+9.4f}{dc:>+9.4f}")
    names = ["FP вне правила", "FP в частично захваченной компоненте", "FP в целой компоненте (взята целиком)",
             "FN вне правила", "FN в частично захваченной компоненте", "FN в целой компоненте (не взята)"]
    print("\nдиагностика ошибок финала v5 (пиксели, доля от всех ошибок):")
    for s in diag:
        tot = diag[s].sum()
        print(f"  {s}: " + "; ".join(f"{n} {v} ({100 * v / tot:.1f} %)" for n, v in zip(names, diag[s])))
    out = {"stack": args.stack, "decode": decs, "post": post, "metrics": res,
           "diag": {s: dict(zip(names, map(int, d))) for s, d in diag.items()}, "seconds": time.perf_counter() - t0}
    (ROOT / "reports" / "bs_snap_to_rule.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nготово за {time.perf_counter() - t0:.0f} с → reports/bs_snap_to_rule.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
