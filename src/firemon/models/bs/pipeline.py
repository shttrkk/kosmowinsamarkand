"""Полный пайплайн обучения BS (вызывается из `python train.py --module bs`).

Этапы (можно выбрать через --stages): rules → lgbm → unet → stack → ensemble.
  rules     шаг 0: пороги USGS и пороги по landcover (OOF по обоим сплитам) → data/cache/bs_oof/rules.json
  lgbm      выборка пикселей, CV (group+temporal), финальная пиксельная LGBM → weights/bs_lgbm.txt
  unet      UNet: CV (group+temporal) → OOF-вероятности; обучение на всём train → weights/bs_unet.pt
  stack     LGBM 2-го уровня по OOF UNet: CV + финальная → weights/bs_stack.txt
  ensemble  веса ансамбля, декодирование, постобработка → weights/bs_decode.json
Итоговые цифры всех этапов → reports/bs_results.json (из него собирается reports/bs_module.md).
"""
from __future__ import annotations

import json
import shutil

from ...utils import REPO_ROOT, get_logger, seed_everything
from .common import evaluate_masks, fmt, get_splits, load_chips

log = get_logger("bs.pipeline")
OOF = REPO_ROOT / "data" / "cache" / "bs_oof"
STAGES = ["rules", "lgbm", "unet", "stack", "ensemble"]

# финальная конфигурация (выбрана по экспериментам, см. reports/experiments.md)
UNET_TAG = "r34_cedice"
UNET_ARGS = ["--encoder", "resnet34", "--loss", "ce_dice", "--epochs", "60", "--channels", "v1"]
LGBM_ROUNDS_FULL = 100   # ≈ средняя best_iteration по фолдам
STACK_ROUNDS_FULL = 150


def stage_rules(data_dir) -> dict:
    from .rules import fit_lc_thresholds, predict_bs_rules

    chips = load_chips(data_dir)
    ids = [c["chip_id"] for c in chips]
    gts = {c["chip_id"]: c["mask"] for c in chips}
    res: dict = {}
    for split, folds in get_splits(ids).items():
        usgs, by_lc, thr = {}, {}, None
        for tr, va in folds:
            thr = fit_lc_thresholds([chips[i] for i in tr])
            for i in va:
                usgs[ids[i]] = predict_bs_rules(chips[i])
                by_lc[ids[i]] = predict_bs_rules(chips[i], lc_thresholds=thr)
        res[split] = {"usgs": evaluate_masks(usgs, {k: gts[k] for k in usgs}),
                      "by_landcover": evaluate_masks(by_lc, {k: gts[k] for k in by_lc}),
                      "lc_thresholds_last_fold": {str(k): v for k, v in thr.items()}}
        log.info("rules %s USGS:        %s", split, fmt(res[split]["usgs"]))
        log.info("rules %s по покрову:  %s", split, fmt(res[split]["by_landcover"]))
    OOF.mkdir(parents=True, exist_ok=True)
    (OOF / "rules.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def run(data_dir: str, stages: list[str] | None = None) -> dict:
    from . import ensemble, lgbm, stack, unet

    seed_everything(42)
    stages = stages or STAGES
    out: dict = {}
    weights = REPO_ROOT / "weights"
    if "rules" in stages:
        out["rules"] = stage_rules(data_dir)
    if "lgbm" in stages:
        lgbm.main(["--data-dir", data_dir, "--stage", "sample"])
        out["lgbm"] = lgbm.main(["--data-dir", data_dir, "--stage", "cv", "--tag", "main"])
        lgbm.main(["--data-dir", data_dir, "--stage", "full", "--rounds", str(LGBM_ROUNDS_FULL)])
    if "unet" in stages:
        for split in ("group", "temporal", "full"):
            r = unet.main(["--data-dir", data_dir, "--split", split, "--tag", UNET_TAG, *UNET_ARGS])
            out[f"unet_{split}"] = {k: v for k, v in r.items() if k != "history"}
        shutil.copyfile(weights / f"bs_unet_{UNET_TAG}.pt", weights / "bs_unet.pt")
    if "stack" in stages:
        out["stack"] = stack.main(["--data-dir", data_dir, "--stage", "cv", "--unet", UNET_TAG])
        stack.main(["--data-dir", data_dir, "--stage", "full", "--unet", UNET_TAG, "--rounds", str(STACK_ROUNDS_FULL)])
    if "ensemble" in stages:
        out["ensemble"] = ensemble.main(["--data-dir", data_dir, "--unet", UNET_TAG, "--lgbm", "main",
                                         "--stack", UNET_TAG])
    path = REPO_ROOT / "reports" / "bs_results.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=1, default=float), encoding="utf-8")
    log.info("итоги → %s", path)
    return out
