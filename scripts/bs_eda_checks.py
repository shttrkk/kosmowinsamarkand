"""Проверка четырёх фактов EDA об эталоне BS на OOF финальной модели (оба сплита + облачные чипы отдельно).

    python scripts/bs_eda_checks.py --probs stack_v1_geo

1. Буфер события (окружность ~4 км): обнуление дальше R, мягкий приор, «компоненты, пересекающие круг» —
   для центра чипа и для оценённого центра события (пик сглаженной ожидаемой степени).
2. SCL_post ∈ {0,3,9,11} → 0 (и то же на чипах с cloud_frac > 0.25).
3. Минимальная единица картографирования: удаление компонент < 25 / 50 / 100 px.
4. Степень внутри предсказанного контура: (а) модель, (б) пороги dNBR по покрову, (в) смесь.
Результат → reports/bs_eda_checks.json (+ печать таблицы Δ BS-части).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.features.bs_features import RULE_THR, RULE_THR_DEFAULT, _gauss, _median  # noqa: E402
from firemon.models.bs.common import decode_probs, evaluate_masks, fit_decode, get_splits, load_chips  # noqa: E402
from firemon.models.bs.postprocess import remove_small  # noqa: E402
from firemon.models.bs.rules import nbr  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
EDA_THR = {10: (0.16, 0.32, 0.60), 30: (0.08, 0.22, 0.40), 40: (0.08, 0.18, 0.38), 90: (0.14, 0.34, 0.62)}
YY, XX = np.mgrid[:512, :512]
D_CENTER = np.hypot(YY - 255.5, XX - 255.5)


def rule_sev(chip: dict, table: dict) -> np.ndarray:
    d = _median(nbr(chip["s2_pre"]) - nbr(chip["s2_post"]), 3)
    lc = chip["aux"][2]
    sev = np.digitize(d, RULE_THR_DEFAULT).astype(np.uint8)
    for code, thr in table.items():
        s = lc == code
        sev[s] = np.digitize(d[s], thr)
    return sev


def event_dist(p: np.ndarray) -> np.ndarray:
    g = _gauss(p[1] + 2.0 * p[2] + 3.0 * p[3], 40.0)
    cy, cx = np.unravel_index(int(g.argmax()), g.shape)
    return np.hypot(YY - cy, XX - cx)


def keep_components(mask: np.ndarray, inside: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask > 0, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask
    ok = np.zeros(n + 1, bool)
    ok[np.unique(lab[inside & (lab > 0)])] = True
    ok[0] = False
    return np.where(ok[lab], mask, 0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--probs", default="stack_v1_geo")
    args = ap.parse_args()

    chips = load_chips(args.data_dir)
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    splits = get_splits(ids)
    probs = {s: np.load(OOF / f"{args.probs}_{s}.npy", mmap_mode="r") for s in ("group", "temporal")}
    t_tr, t_va = splits["temporal"][0]
    cloudy = np.array([float(c["meta"].get("cloud_frac") or 0) > 0.25 for c in chips])
    dec = {"group": fit_decode(probs["group"][:], gts), "temporal": fit_decode(probs["group"][t_tr], gts[t_tr])}
    subsets = {"group": np.arange(len(ids)), "temporal": t_va, "cloudy>0.25 (group OOF)": np.nonzero(cloudy)[0]}
    src = {"group": "group", "temporal": "temporal", "cloudy>0.25 (group OOF)": "group"}
    scl_post0 = [np.isin(c["s2_post"][9], (0, 3, 9, 11)) for c in chips]
    rules = {"mine": [rule_sev(c, RULE_THR) for c in chips], "eda": [rule_sev(c, EDA_THR) for c in chips]}

    def evaluate(fn) -> dict:
        """fn(i, p (4,H,W) float32, dec) → mask. Метрики по трём подмножествам."""
        out = {}
        for name, idx in subsets.items():
            s = src[name]
            preds = {i: fn(i, np.asarray(probs[s][i], np.float32), dec[s]) for i in idx}
            out[name] = evaluate_masks(preds, {i: gts[i] for i in idx})
        return out

    def base(i, p, d):
        return decode_probs(p, d["t_burn"], d["w"])

    res = {"probs": args.probs, "decode": dec, "n_cloudy": int(cloudy.sum()), "variants": {}}
    V = res["variants"]
    V["0. базовая (декодирование порогами)"] = evaluate(base)

    # 1. буфер события
    for R in (200, 240, 280, 320):
        V[f"1a. обнулить дальше R={R} от центра чипа"] = evaluate(lambda i, p, d, R=R: np.where(D_CENTER <= R, base(i, p, d), 0))
    for R in (240, 300):
        def soft(i, p, d, R=R):
            q = p.copy()
            prior = 1.0 / (1.0 + np.exp((D_CENTER - R) / 25.0))
            q[1:] *= prior
            q[0] = 1.0 - q[1:].sum(0)
            return decode_probs(q, d["t_burn"], d["w"])
        V[f"1b. мягкий приор (сигмоида, R={R}, s=25) от центра чипа"] = evaluate(soft)
    for R in (150, 200, 250):
        V[f"1c. компоненты, пересекающие круг R={R} (центр чипа)"] = evaluate(
            lambda i, p, d, R=R: keep_components(base(i, p, d), D_CENTER <= R))
    for R in (200, 250, 300):
        V[f"1d. обнулить дальше R={R} от оценённого центра события"] = evaluate(
            lambda i, p, d, R=R: np.where(event_dist(p) <= R, base(i, p, d), 0))
    for R in (150, 200, 250):
        V[f"1e. компоненты, пересекающие круг R={R} (центр события)"] = evaluate(
            lambda i, p, d, R=R: keep_components(base(i, p, d), event_dist(p) <= R))

    # 2. SCL_post
    V["2. SCL_post ∈ {0,3,9,11} → 0"] = evaluate(lambda i, p, d: np.where(scl_post0[i], 0, base(i, p, d)))

    # 3. MMU
    for n in (25, 50, 100):
        V[f"3. удалить компоненты < {n} px"] = evaluate(lambda i, p, d, n=n: remove_small(base(i, p, d), n))

    # 4. степень внутри предсказанного контура
    for key, tab in rules.items():
        def by_rule(i, p, d, tab=tab):
            m = base(i, p, d)
            return np.where((m > 0) & (tab[i] > 0), tab[i], m).astype(np.uint8)
        V[f"4б. степень = пороги dNBR по покрову ({key}); где правило даёт 0 — модель"] = evaluate(by_rule)

        def mix(i, p, d, tab=tab, a=0.5):
            q = p.copy()
            r = tab[i]
            for k in (1, 2, 3):
                q[k] = np.where(r > 0, (1 - a) * p[k] + a * (p[1:].sum(0)) * (r == k), p[k])
            return decode_probs(q, d["t_burn"], d["w"])
        V[f"4в. смесь 50/50 модель + правило ({key})"] = evaluate(mix)

    ref = V["0. базовая (декодирование порогами)"]
    print(f"{'вариант':70s} | " + " | ".join(f"{n:>24s}" for n in subsets))
    for name, r in V.items():
        cells = []
        for s in subsets:
            m = r[s]
            cells.append(f"{m['iou_burn']:.4f}/{m['miou_sev']:.4f} Δ{m['bs_part'] - ref[s]['bs_part']:+.4f}")
        print(f"{name:70s} | " + " | ".join(f"{c:>24s}" for c in cells))
    (ROOT / "reports" / "bs_eda_checks.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
