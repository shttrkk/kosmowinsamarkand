"""Гипотеза «SAR отличает своё событие»: у компонент «своей» гари есть изменение VH/VV в окне дат S1, у чужих — нет.

    python scripts/bs_sar_event.py [--stack night_avg3s075] [--decode bs_decode_ms.json]

Объекты: (1) компоненты предсказанной маски финала v5 (групповой OOF, после постобработки); (2) компоненты «лишней» гари —
маска правила эталона минус эталон, ≥ 25 px. Метка — доля эталона в компоненте > 0.5. Признаки — S1 (дБ, медиана 5×5):
средний/медианный dVH, dVV, d(VH−VV), доля пикселей с dVH < −1.5 дБ и те же величины минус фон (кольцо 10 px снаружи).
AUC каждого признака и логистической регрессии (OOF по групповым фолдам), отдельно лес/кустарник и трава/пашня.
Ничего не обучается, кроме логрегрессии на десятках признаков. Результат: reports/bs_sar_event.json.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bs_snap_to_rule import EIGHT, rule_of  # noqa: E402
from firemon.features.bs_features import _median  # noqa: E402
from firemon.models.bs.common import decode_probs, fit_decode, get_splits, load_chips  # noqa: E402
from firemon.models.bs.postprocess import postprocess_bs  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
FEATS = ["dVH_mean", "dVH_med", "dVV_mean", "dVV_med", "dCR_mean", "dCR_med", "frac_dVH_lt_-1.5"]
ALL_FEATS = FEATS + [f + "_vs_ring" for f in FEATS]
LC_GROUPS = {"лес/кустарник": (10, 20), "трава/пашня": (30, 40)}


def sar_maps(chip: dict):
    pre, post = chip["s1_pre"].astype(np.float32) / 100.0, chip["s1_post"].astype(np.float32) / 100.0
    valid = (chip["s1_pre"] != 0).all(0) & (chip["s1_post"] != 0).all(0)
    sm = lambda x: _median(x, 5)  # noqa: E731
    dvv, dvh = sm(post[0]) - sm(pre[0]), sm(post[1]) - sm(pre[1])
    return dvh, dvv, dvh - dvv, valid


def stats(sel: np.ndarray, dvh, dvv, dcr) -> list[float]:
    return [float(dvh[sel].mean()), float(np.median(dvh[sel])), float(dvv[sel].mean()), float(np.median(dvv[sel])),
            float(dcr[sel].mean()), float(np.median(dcr[sel])), float((dvh[sel] < -1.5).mean())]


def components(mask: np.ndarray, chip: dict, gt: np.ndarray, kind: str, chip_i: int, maps) -> list[dict]:
    dvh, dvv, dcr, valid = maps
    lab, n = ndi.label(mask, structure=EIGHT)
    rows = []
    lc = chip["aux"][2]
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        pad = tuple(slice(max(s.start - 12, 0), s.stop + 12) for s in sl)
        comp = lab[pad] == k
        size = int(comp.sum())
        if size < 25:
            continue
        ring = ndi.binary_dilation(comp, structure=EIGHT, iterations=10) & ~comp & ~(mask[pad])
        v = valid[pad]
        cv, rv = comp & v, ring & v
        row = {"chip": chip_i, "kind": kind, "size": size, "label": int((gt[pad][comp] > 0).mean() > 0.5),
               "s1_missing": int(cv.sum() < 0.5 * size or rv.sum() < 25)}
        codes, counts = np.unique(lc[pad][comp], return_counts=True)
        row["lc"] = int(codes[counts.argmax()])
        if row["s1_missing"]:
            row.update({f: np.nan for f in ALL_FEATS})
        else:
            a = stats(cv, dvh[pad], dvv[pad], dcr[pad])
            b = stats(rv, dvh[pad], dvv[pad], dcr[pad])
            row.update(dict(zip(FEATS, a)))
            row.update({f + "_vs_ring": x - y for f, x, y in zip(FEATS, a, b)})
        rows.append(row)
    return rows


def auc(y, x, w=None):
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, x, sample_weight=w))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="night_avg3s075")
    ap.add_argument("--decode", default="bs_decode_ms.json")
    args = ap.parse_args()
    t0 = time.perf_counter()
    chips = load_chips(ROOT / "data" / "train")
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    splits = get_splits(ids)
    pg = np.load(OOF / f"stack_{args.stack}_group.npy", mmap_mode="r")
    post = json.loads((ROOT / "weights" / args.decode).read_text(encoding="utf-8"))["postprocess"]
    dec = fit_decode(np.asarray(pg, np.float32), gts)

    def work(i: int) -> list[dict]:
        chip, gt = chips[i], chips[i]["mask"]
        probs = np.asarray(pg[i], np.float32)
        pred = postprocess_bs(decode_probs(probs, dec["t_burn"], dec["w"]), chip, post, probs=probs, decode=dec) > 0
        extra = (rule_of(chip) > 0) & ~(gt > 0)
        maps = sar_maps(chip)
        return components(pred, chip, gt, "pred", i, maps) + components(extra, chip, gt, "extra", i, maps)

    with ThreadPoolExecutor(8) as ex:
        rows = [r for part in ex.map(work, range(len(chips))) for r in part]
    import pandas as pd

    df = pd.DataFrame(rows)
    fold_of = np.zeros(len(chips), int)
    for f, (_, va) in enumerate(splits["group"]):
        fold_of[va] = f
    df["fold"] = fold_of[df["chip"].to_numpy()]
    res: dict = {"stack": args.stack, "n_components": int(len(df)), "sets": {}}
    print(f"компонент: {len(df)} (pred {int((df.kind == 'pred').sum())}, extra {int((df.kind == 'extra').sum())}); "
          f"без S1: {100 * df.s1_missing.mean():.1f} %; {time.perf_counter() - t0:.0f} с")

    def evaluate(name: str, d):
        d = d[d.s1_missing == 0]
        out = {"n": int(len(d)), "pos": int(d.label.sum()), "auc": {}, "auc_px": {}}
        if len(d) < 30 or d.label.nunique() < 2:
            res["sets"][name] = out
            print(f"\n[{name}] n={len(d)}, положительных {int(d.label.sum())} — мало данных")
            return
        y, w = d.label.to_numpy(), d["size"].to_numpy()
        for f in ALL_FEATS:
            a, apx = auc(y, -d[f].to_numpy()), auc(y, -d[f].to_numpy(), w)  # знак: у гари VH падает → «−признак»
            if f.startswith("frac"):
                a, apx = auc(y, d[f].to_numpy()), auc(y, d[f].to_numpy(), w)
            out["auc"][f], out["auc_px"][f] = a, apx
        oof = np.full(len(d), np.nan)
        X = d[ALL_FEATS].to_numpy()
        for f in sorted(d.fold.unique()):
            tr, va = (d.fold != f).to_numpy(), (d.fold == f).to_numpy()
            if len(set(y[tr])) < 2:
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            model.fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])[:, 1]
        ok = np.isfinite(oof)
        out["auc"]["logreg_oof"], out["auc_px"]["logreg_oof"] = auc(y[ok], oof[ok]), auc(y[ok], oof[ok], w[ok])
        res["sets"][name] = out
        best = sorted(out["auc"].items(), key=lambda kv: -abs((kv[1] or 0.5) - 0.5))[:4]
        print(f"\n[{name}] n={out['n']}, «своих» {out['pos']} ({100 * out['pos'] / out['n']:.0f} %)")
        print("  AUC логрегрессии (OOF по групповым фолдам): "
              f"{out['auc']['logreg_oof']:.3f}; взвешенно по пикселям {out['auc_px']['logreg_oof']:.3f}")
        print("  лучшие одиночные признаки: " + "; ".join(f"{k} {v:.3f}" for k, v in best))

    for kind_name, sel in (("pred + extra", df), ("только pred", df[df.kind == "pred"])):
        evaluate(f"{kind_name} | все покровы", sel)
        for lc_name, codes in LC_GROUPS.items():
            evaluate(f"{kind_name} | {lc_name}", sel[sel.lc.isin(codes)])
    res["seconds"] = time.perf_counter() - t0
    (ROOT / "reports" / "bs_sar_event.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    df.to_csv(ROOT / "out" / "bs_sar_event_components.csv", index=False)
    print(f"\nготово за {time.perf_counter() - t0:.0f} с → reports/bs_sar_event.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
