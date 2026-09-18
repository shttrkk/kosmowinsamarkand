"""Шаг 4: оценка опций постобработки на OOF (GroupKFold) и на временном сплите. Опция принимается,
только если даёт прирост на ОБОИХ. Запуск: python scripts/af/postprocess_eval.py"""
from __future__ import annotations

import numpy as np

from firemon.io import read_meta
from firemon.models.af.dataset import load_or_build
from firemon.models.af.train_lgbm import run_cv, run_split, DEFAULT_PARAMS, best_threshold
from firemon.postprocess import drop_isolated, hysteresis
from firemon.splits import load_folds
from firemon.utils import REPO_ROOT, seed_everything


def main() -> None:
    seed_everything(42, with_torch=False)
    data = REPO_ROOT / "data"
    meta = read_meta(data / "train", "af").merge(load_folds("af"), on="chip_id")
    ds = load_or_build(data / "train", meta["chip_id"].tolist(), data / "cache" / "af_train.npz")
    names = list(ds["feature_names"])
    folds, ts = meta["fold"].to_numpy(), meta["temporal_split"].to_numpy()
    cand = ds["is_cand"]
    cols = np.arange(len(names))
    oof, _, _ = run_cv(ds, folds, None, cols, cand)
    _, va, p = run_split(ds, ts == "train", ts == "val", DEFAULT_PARAMS, cols, cand)
    tprob = np.full(len(cand), np.nan, np.float32)
    tprob[va] = p
    thr0, _ = best_threshold(ds, oof)

    ci, yy, xx, y = ds["chip_idx"], ds["yy"].astype(int), ds["xx"].astype(int), ds["y"].astype(bool)
    X = ds["X"]
    glint = (X[:, names.index("landcover")] == 80) & (X[:, names.index("is_night")] < 0.5)
    i3 = X[:, names.index("I3")]

    def maps(prob):
        out = {}
        for c in np.unique(ci[np.isfinite(prob)]):
            r = (ci == c) & np.isfinite(prob)
            pm = np.zeros((256, 256), np.float32)
            pm[yy[r], xx[r]] = prob[r]
            gt = np.zeros((256, 256), bool)
            rr = (ci == c) & y
            gt[yy[rr], xx[rr]] = True
            g = np.zeros((256, 256), np.float32)
            rg = r & glint
            g[yy[rg], xx[rg]] = i3[rg]
            out[c] = (pm, gt, g)
        return out

    def f1(mp, fn, n_fire):
        tp = fp = 0
        for pm, gt, g in mp.values():
            pred = fn(pm, g)
            tp += (pred & gt).sum()
            fp += (pred & ~gt).sum()
        return 2 * tp / (2 * tp + fp + (n_fire - tp))

    m_oof, m_t = maps(oof), maps(tprob)
    n_oof, n_t = ds["n_fire"].sum(), ds["n_fire"][ts == "val"].sum()

    def report(name, fn):
        a, b = f1(m_oof, fn, n_oof), f1(m_t, fn, n_t)
        print(f"{name:42s} OOF {a:.4f}  temporal {b:.4f}", flush=True)
        return a, b

    base = report(f"порог {thr0:.2f} (без постобработки)", lambda pm, g: pm > thr0)
    best = (base[0], None)
    for th in (0.4, 0.5, 0.6, 0.7, 0.8):
        for tl in (0.1, 0.15, 0.2, 0.3, 0.4):
            if tl >= th:
                continue
            a, b = report(f"гистерезис high {th} low {tl}", lambda pm, g, th=th, tl=tl: hysteresis(pm, th, tl))
            if a > best[0]:
                best = (a, (th, tl, b))
    print("лучший гистерезис по OOF:", best)
    for t_iso in (0.6, 0.7, 0.8, 0.9):
        report(f"порог {thr0:.2f} + удаление одиночных p<{t_iso}", lambda pm, g, t=t_iso: drop_isolated(pm > thr0, pm, t))
    for i3_thr in (0.15, 0.2, 0.3):
        report(f"порог {thr0:.2f} + вода днём I3>{i3_thr} -> 0", lambda pm, g, t=i3_thr: (pm > thr0) & ~(g > t))


if __name__ == "__main__":
    main()
