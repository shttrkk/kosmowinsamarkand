"""Идея (h), AF: явные тесты VNP14 (Schroeder et al. 2014, RSE 143:85–96, https://doi.org/10.1016/j.rse.2013.12.008)
как дополнительные признаки LightGBM. Базовый датасет/фолды/параметры — как у AF-17; веса НЕ сохраняются, если не принято.

Новые признаки: v_bright_surface (ρ3>0.3 ∧ ρ3>ρ2 ∧ ρ2>0.25 ∧ BT4≤335, день), v_night_cold (ночь ∧ BT4<295),
v_ncand5 / v_ncand7 (число кандидатов-соседей), z-score I4 и I4−I5 по фону окна 7/15 С ИСКЛЮЧЕНИЕМ кандидатов.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.features.af_features import candidate_mask, night_mask  # noqa: E402
from firemon.io import read_af_chip, read_meta  # noqa: E402
from firemon.models.af.dataset import load_or_build  # noqa: E402
from firemon.models.af.train_lgbm import DEFAULT_PARAMS, best_threshold, run_cv, run_split, score_rows  # noqa: E402
from firemon.splits import load_folds  # noqa: E402
from firemon.utils import load_config, seed_everything  # noqa: E402

NEW = ["v_bright_surface", "v_night_cold", "v_ncand5", "v_ncand7", "v_i4_cz7", "v_di_cz7", "v_i4_cz15", "v_di_cz15"]


def chip_maps(data_dir, cid):
    chip = read_af_chip(data_dir, cid, with_mask=False)
    v, valid = chip["viirs"], chip["valid"]
    i1, i2, i3, i4, i5 = (np.nan_to_num(v[k], nan=0.0) for k in range(5))
    night = night_mask(v)
    cand = candidate_mask(v, valid)
    di = i4 - i5
    out = np.zeros((len(NEW), 256, 256), np.float32)
    out[0] = (~night) & (i3 > 0.3) & (i3 > i2) & (i2 > 0.25) & (i4 <= 335)
    out[1] = night & (i4 < 295)
    c = cand.astype(np.float32)
    out[2] = ndi.uniform_filter(c, 5, mode="constant") * 25 - c
    out[3] = ndi.uniform_filter(c, 7, mode="constant") * 49 - c
    bg = (valid & ~cand).astype(np.float32)
    for j, k in ((4, 7), (6, 15)):
        n = ndi.uniform_filter(bg, k, mode="constant") * k * k
        for off, a in ((0, i4), (1, di)):
            s1 = ndi.uniform_filter(a * bg, k, mode="constant") * k * k
            s2 = ndi.uniform_filter(a * a * bg, k, mode="constant") * k * k
            mean = s1 / np.maximum(n, 1)
            std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
            out[j + off] = np.where(n >= 8, (a - mean) / np.maximum(std, 0.5), 0.0)
    return out


def evaluate(ds, folds, ts, params, n_rounds, cols):
    oof, boosters, iters = run_cv(ds, folds, params, cols, ds["is_cand"], n_rounds)
    thr, r = best_threshold(ds, oof)
    _, va, p = run_split(ds, ts == "train", ts == "val", params, cols, ds["is_cand"], n_rounds)
    tprob = np.full(len(ds["y"]), np.nan, np.float32)
    tprob[va] = p
    rt = score_rows(ds, tprob, thr, chip_sel=ts == "val")
    return {"oof": r, "temporal": rt, "threshold": thr, "iters": [int(i) for i in iters]}, boosters


def main() -> None:
    cfg = load_config(ROOT / "configs" / "af_lgbm.yaml")
    seed_everything(42, with_torch=False)
    data_dir = ROOT / "data" / "train"
    meta = read_meta(data_dir, "af").merge(load_folds("af"), on="chip_id", how="left")
    cache = ROOT / cfg["cache"]
    if cache.exists():  # общий кэш AF не перезаписываем: при несовпадении признаков строим датасет в памяти
        from firemon.features.af_features import FEATURE_NAMES

        with np.load(cache, allow_pickle=False) as z:
            if list(z["feature_names"]) != FEATURE_NAMES or list(z["chip_ids"]) != meta["chip_id"].tolist():
                cache = None
    else:
        cache = None
    ds = load_or_build(data_dir, meta["chip_id"].tolist(), cache, 42, False)
    params = {**DEFAULT_PARAMS, **cfg.get("params", {}), "seed": 42}
    folds, ts = meta["fold"].to_numpy().astype(int), meta["temporal_split"].to_numpy()
    n0 = ds["X"].shape[1]
    base, _ = evaluate(ds, folds, ts, params, int(cfg.get("n_rounds", 2000)), np.arange(n0))
    print("AF-17 (пересчёт): OOF F1 %.4f | temporal F1 %.4f" % (base["oof"]["f1"], base["temporal"]["f1"]))

    ids = meta["chip_id"].tolist()
    with ThreadPoolExecutor(8) as ex:
        maps = list(ex.map(lambda c: chip_maps(data_dir, c), ids))
    extra = np.zeros((len(ds["y"]), len(NEW)), np.float32)
    for ci in range(len(ids)):
        r = np.nonzero(ds["chip_idx"] == ci)[0]
        extra[r] = maps[ci][:, ds["yy"][r], ds["xx"][r]].T
    ds2 = dict(ds)
    ds2["X"] = np.concatenate([ds["X"], extra], axis=1)
    ds2["feature_names"] = np.array(list(ds["feature_names"]) + NEW)
    new, boosters = evaluate(ds2, folds, ts, params, int(cfg.get("n_rounds", 2000)), np.arange(n0 + len(NEW)))
    print("+ тесты VNP14:     OOF F1 %.4f | temporal F1 %.4f" % (new["oof"]["f1"], new["temporal"]["f1"]))
    gain = np.sum([b.feature_importance("gain") for b in boosters], axis=0)
    imp = {n: float(gain[n0 + j] / gain.sum()) for j, n in enumerate(NEW)}
    d = {"oof": new["oof"]["f1"] - base["oof"]["f1"], "temporal": new["temporal"]["f1"] - base["temporal"]["f1"]}
    ok = d["oof"] >= 0.003 and d["temporal"] >= 0.003
    print("Δ F1:", {k: round(v, 4) for k, v in d.items()}, "→", "ПРИНЯТО" if ok else "не принято", "| важность новых:", imp)
    (ROOT / "reports" / "af_night_vnp14.json").write_text(json.dumps(
        {"base": base, "new": new, "delta_f1": d, "accepted": ok, "importance_new": imp}, indent=1, default=float,
        ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
