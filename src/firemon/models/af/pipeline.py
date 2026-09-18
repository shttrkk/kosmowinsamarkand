"""Полный цикл обучения AF: признаки → кэш → 5 фолдов → OOF-метрики/срезы → временной сплит → веса."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ...io import read_meta
from ...splits import load_folds
from ...utils import REPO_ROOT, get_logger, seed_everything
from .dataset import load_or_build
from .predictor import META_NAME
from .train_lgbm import DEFAULT_PARAMS, best_threshold, run_cv, run_split, score_rows

log = get_logger("train.af")
LC_NAMES = {10: "лес", 20: "кустарник", 30: "трава", 40: "пашня", 50: "застройка", 60: "голая почва",
            80: "вода", 90: "болото"}


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def slices(ds: dict, meta: pd.DataFrame, oof: np.ndarray, thr: float) -> dict:
    """F1 по срезам (день/ночь, спутник, landcover) и полнота по размеру эталонной компоненты."""
    names = list(ds["feature_names"])
    y = ds["y"].astype(bool)
    pred = np.nan_to_num(oof, nan=0.0) > thr
    out: dict = {}

    def f1(sel):
        tp, fp, fn = (pred & y & sel).sum(), (pred & ~y & sel).sum(), (~pred & y & sel).sum()
        return {"f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    night = ds["X"][:, names.index("is_night")] > 0.5
    out["день"], out["ночь"] = f1(~night), f1(night)
    sat = meta["satellite"].to_numpy()[ds["chip_idx"]]
    for s in pd.unique(sat):
        out[f"спутник {s}"] = f1(sat == s)
    lc = ds["X"][:, names.index("landcover")]
    for code, nm in LC_NAMES.items():
        if ((lc == code) & (y | pred)).any():
            out[f"landcover {code} ({nm})"] = f1(lc == code)
    size = np.zeros(len(y), np.int64)
    for c in np.unique(ds["chip_idx"][y]):
        r = np.where((ds["chip_idx"] == c) & y)[0]
        gt = np.zeros((256, 256), bool)
        gt[ds["yy"][r], ds["xx"][r]] = True
        lab, _ = ndi.label(gt, structure=np.ones((3, 3), bool))
        size[r] = np.bincount(lab.ravel())[lab[ds["yy"][r], ds["xx"][r]]]
    for nm, sel in (("1 px", size == 1), ("2-5 px", (size >= 2) & (size <= 5)), (">5 px", size > 5)):
        out[f"recall, компонента {nm}"] = {"recall": float((pred & sel).sum() / max(sel.sum(), 1)), "n": int(sel.sum())}
    return out


def append_experiment(row: str) -> None:
    """Дописать строку в таблицу reports/experiments.md с очередным номером."""
    path = REPO_ROOT / "reports" / "experiments.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    table = [i for i, ln in enumerate(lines) if ln.startswith("|")]
    nums = [int(c) for c in (lines[i].split("|")[1].strip() for i in table) if c.isdigit()]
    lines.insert(table[-1] + 1, f"| {max(nums, default=0) + 1} {row}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_af(config: dict, data_dir: str | Path | None = None, log_experiment: bool = True,
             rebuild_cache: bool = False) -> dict:
    t0 = time.time()
    seed = int(config.get("seed", 42))
    seed_everything(seed, with_torch=False)
    data_dir = _resolve(data_dir or config.get("data_dir", "data/train"))
    weights_dir = _resolve(config.get("weights_dir", "weights"))
    params = {**DEFAULT_PARAMS, **config.get("params", {}), "seed": seed}
    n_rounds = int(config.get("n_rounds", 2000))

    meta = read_meta(data_dir, "af").merge(load_folds("af"), on="chip_id", how="left")
    if meta["fold"].isna().any():
        raise RuntimeError("в configs/folds_af.csv нет части чипов — пересоздайте фолды (python -m firemon.splits)")
    cache = _resolve(config["cache"]) if config.get("cache") else None
    ds = load_or_build(data_dir, meta["chip_id"].tolist(), cache, seed, rebuild_cache)
    cand = ds["is_cand"]
    log.info("датасет: %d строк, %d признаков, кандидатов %d, огня %d (в кандидатах %.2f %%)", len(cand),
             ds["X"].shape[1], cand.sum(), ds["n_fire"].sum(), 100 * ds["y"][cand].sum() / max(ds["n_fire"].sum(), 1))

    folds, ts = meta["fold"].to_numpy().astype(int), meta["temporal_split"].to_numpy()
    oof, boosters, iters = run_cv(ds, folds, params, None, cand, n_rounds)
    thr, r = best_threshold(ds, oof)
    log.info("OOF GroupKFold: F1 %.4f P %.4f R %.4f при пороге %.2f; итерации %s",
             r["f1"], r["precision"], r["recall"], thr, iters)

    _, va, p = run_split(ds, ts == "train", ts == "val", params, np.arange(ds["X"].shape[1]), cand, n_rounds)
    tprob = np.full(len(cand), np.nan, np.float32)
    tprob[va] = p
    rt = score_rows(ds, tprob, thr, chip_sel=ts == "val")
    log.info("временной сплит (train <=2023 → val 2024, порог с OOF): F1 %.4f P %.4f R %.4f",
             rt["f1"], rt["precision"], rt["recall"])

    sl = slices(ds, meta, oof, thr)
    for k, v in sl.items():
        log.info("  срез %-28s %s", k, v)
    gain = np.sum([b.feature_importance("gain") for b in boosters], axis=0)
    importance = [(str(ds["feature_names"][i]), float(gain[i] / gain.sum())) for i in np.argsort(-gain)[:25]]

    weights_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for k, b in enumerate(boosters):
        name = f"af_lgbm_fold{k}.txt"
        b.save_model(str(weights_dir / name), num_iteration=b.best_iteration)
        models.append(name)
    info = {"threshold": thr, "models": models, "feature_names": [str(n) for n in ds["feature_names"]],
            "best_iterations": [int(i) for i in iters], "params": params,
            "oof": {k: float(v) for k, v in r.items()}, "temporal": {k: float(v) for k, v in rt.items()}}
    (weights_dir / META_NAME).write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    report = {"oof": r, "temporal": rt, "threshold": thr, "iters": iters, "slices": sl, "importance": importance}
    (REPO_ROOT / "reports" / "af_train_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    if log_experiment:
        append_experiment(
            f"| {dt.date.today().isoformat()} | AF | `train.py --module af`: LightGBM по кандидатам, "
            f"{ds['X'].shape[1]} признаков, среднее 5 фолд-моделей | `configs/af_lgbm.yaml` | GroupKFold-5 по тайлу (OOF) "
            f"+ temporal (val=2024) | {seed} | {r['f1']:.4f} (P {r['precision']:.3f} / R {r['recall']:.3f}); "
            f"temporal {rt['f1']:.4f} | — | — | — | — | порог {thr:.2f}, итерации {iters}; автозапись train.py |")
    log.info("веса → %s; всего %.1f с", weights_dir, time.time() - t0)
    return report
