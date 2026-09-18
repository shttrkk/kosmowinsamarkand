"""Эксперименты AF (шаги 3–4): варианты LightGBM, абляции, день/ночь — на GroupKFold и на временном сплите.
Запуск: python scripts/af/experiments.py [--only имя,имя]  → печать таблицы + reports/af_experiments.json"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from firemon.features.af_features import AUX_FEATURES, CONTEXT_FEATURES, GLINT_FEATURES
from firemon.io import read_meta
from firemon.models.af.dataset import load_or_build
from firemon.models.af.train_lgbm import DEFAULT_PARAMS, best_threshold, run_cv, run_split, score_rows
from firemon.splits import load_folds
from firemon.utils import REPO_ROOT, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    seed_everything(42, with_torch=False)
    data = REPO_ROOT / "data"
    meta = read_meta(data / "train", "af").merge(load_folds("af"), on="chip_id")
    ds = load_or_build(data / "train", meta["chip_id"].tolist(), data / "cache" / "af_train.npz")
    names = list(ds["feature_names"])
    folds = meta["fold"].to_numpy()
    ts = meta["temporal_split"].to_numpy()
    cand, night = ds["is_cand"], ds["X"][:, names.index("is_night")] > 0.5
    allc = np.arange(len(names))

    def without(drop):
        return np.array([i for i, n in enumerate(names) if n not in set(drop)])

    w7_drop = [n for n in CONTEXT_FEATURES if n[-1].isdigit() and not n.endswith("7") and not n.startswith(("i4_m_p", "di_m_p"))
               and any(n.endswith(str(w)) for w in (3, 5, 11, 15))]
    ring_far = cand & ~(ds["is_ring"] & (ds["y"] == 0))          # кандидаты без негативов в кольце r=2 у огня
    variants = {
        "lgbm_base": dict(),
        "lgbm_spw5": dict(params={"scale_pos_weight": 5.0}),
        "lgbm_unbalance": dict(params={"is_unbalance": True}),
        "lgbm_leaves31": dict(params={"num_leaves": 31}),
        "lgbm_leaves127": dict(params={"num_leaves": 127}),
        "lgbm_allrows": dict(rows=np.ones(len(cand), bool)),
        "abl_no_context": dict(cols=without(CONTEXT_FEATURES)),
        "abl_no_aux": dict(cols=without(AUX_FEATURES)),
        "abl_no_glint": dict(cols=without(GLINT_FEATURES)),
        "abl_window7_only": dict(cols=without(w7_drop)),
        "abl_no_hard_neg": dict(rows=ring_far),
        "day_night_separate": dict(split_dn=True),
    }
    only = [s for s in args.only.split(",") if s]
    out_path = REPO_ROOT / "reports" / "af_experiments.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name, v in variants.items():
        if only and name not in only:
            continue
        t0 = time.time()
        params = {**DEFAULT_PARAMS, **v.get("params", {})}
        cols, rows = v.get("cols", allc), v.get("rows", cand)
        if v.get("split_dn"):
            oof = np.full(len(cand), np.nan, np.float32)
            tprob = np.full(len(cand), np.nan, np.float32)
            iters = []
            for part in (night, ~night):
                o, _, it = run_cv(ds, folds, params, cols, rows & part)
                oof[part & np.isfinite(o)] = o[part & np.isfinite(o)]
                _, va, p = run_split(ds, ts == "train", ts == "val", params, cols, rows & part)
                idx = np.where(va)[0]
                tprob[idx[part[idx]]] = p[part[idx]]
                iters += it
        else:
            oof, _, iters = run_cv(ds, folds, params, cols, rows)
            _, va, p = run_split(ds, ts == "train", ts == "val", params, cols, rows)
            tprob = np.full(len(cand), np.nan, np.float32)
            tprob[va] = p
        thr, r = best_threshold(ds, oof)
        # раздельные пороги день/ночь (подбор на OOF)
        n_night = int((ds["y"].astype(bool) & night).sum())
        n_day = int(ds["n_fire"].sum()) - n_night
        tn, rn = best_threshold(ds, oof, row_sel=night, n_fire_total=n_night)
        td, rd = best_threshold(ds, oof, row_sel=~night, n_fire_total=n_day)
        thr_rows = np.where(night, tn, td)
        r_dn = score_rows(ds, oof, thr_rows)
        val = ts == "val"
        rt = score_rows(ds, tprob, thr, chip_sel=val)             # порог с GroupKFold-OOF, без подгонки к 2024
        rt_dn = score_rows(ds, tprob, thr_rows, chip_sel=val)
        results[name] = {
            "n_features": int(len(cols)), "n_train_rows": int(rows.sum()), "iters": [int(i) for i in iters],
            "thr": thr, "oof_f1": r["f1"], "oof_p": r["precision"], "oof_r": r["recall"],
            "thr_day": td, "thr_night": tn, "oof_f1_daynight_thr": r_dn["f1"],
            "oof_f1_day": rd["f1"], "oof_f1_night": rn["f1"],
            "temporal_f1": rt["f1"], "temporal_p": rt["precision"], "temporal_r": rt["recall"],
            "temporal_f1_daynight_thr": rt_dn["f1"], "sec": round(time.time() - t0, 1),
        }
        x = results[name]
        print(f"{name:20s} OOF F1 {x['oof_f1']:.4f} (P {x['oof_p']:.4f} R {x['oof_r']:.4f}) thr {thr:.2f} | "
              f"dn-thr {x['oof_f1_daynight_thr']:.4f} ({td:.2f}/{tn:.2f}) day {x['oof_f1_day']:.4f} night {x['oof_f1_night']:.4f} | "
              f"temporal {x['temporal_f1']:.4f} dn {x['temporal_f1_daynight_thr']:.4f} | it {iters} | {x['sec']} c", flush=True)
        out_path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
