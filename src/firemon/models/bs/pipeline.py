"""Полный пайплайн обучения BS (вызывается из `python train.py --module bs`).

Этапы (можно выбрать через --stages): rules → lgbm → unet → stack → ensemble.
  rules     шаг 0: пороги USGS и пороги по landcover (OOF по обоим сплитам) → data/cache/bs_oof/rules.json
  lgbm      выборка пикселей, CV (group+temporal), финальная пиксельная LGBM → weights/bs_lgbm.txt
  unet      два UNet-resnet34 (каналы v1 и v4): CV (group+temporal) → OOF-вероятности; обучение на всём train →
            weights/bs_unet_<tag>.pt; усреднённые OOF-вероятности → data/cache/bs_oof/avg14_*_tta.npy
  stack     LGBM 2-го уровня по усреднённым OOF UNet (без TTA): CV + финальная → weights/bs_stack.txt
  ensemble  веса ансамбля, декодирование, постобработка → weights/bs_decode.json
  ms_unet   (v5) UNet v4 на масштабах 0.75 и 0.5: CV + full → weights/bs_unet_r34_v4_s075.pt, …_s050.pt (нужен этап unet)
  ms_stack  (v5) 5-канальный OOF (среднее v1, v4, v4@0.75 + p_burn сети 0.5×) → стек с воротами 0.01 → weights/bs_stack_ms.txt;
            декодирование/постобработка → weights/bs_decode_ms.json. Включение на инференсе: bs.variant: ms
Итоговые цифры всех этапов → reports/bs_results.json (из него собирается reports/bs_module.md).
"""
from __future__ import annotations

import json

import numpy as np

from ...utils import REPO_ROOT, get_logger, seed_everything
from .common import cache_dir, evaluate_masks, fmt, get_splits, load_chips

log = get_logger("bs.pipeline")
STAGES = ["rules", "lgbm", "unet", "stack", "ensemble", "ms_unet", "ms_stack"]
# ms_* — многомасштабный вариант (кандидат v5): UNet v4, обученные на масштабах 0.75 и 0.5 (больше контекст → лучше выбор
# «своего» события); база стека = среднее v1, v4 (1×) и v4@0.75; p_burn сети 0.5× — доп. признаки стека; ворота 0.01.
MS_UNETS = {"r34_v4_s075": 0.75, "r34_v4_s050": 0.5}   # тег → масштаб входа; каналы v4
MS_BASE, MS_AUX = ["r34_cedice", "r34_v4", "r34_v4_s075"], ["r34_v4_s050"]
MS_OOF_TAG, MS_STACK_TAG, MS_GATE, MS_FRAC = "ms_base_aux", "ms", "0.01", "0.3"

# финальная конфигурация (выбрана по экспериментам, см. reports/experiments.md)
UNETS = {"r34_cedice": "v1", "r34_v4": "v4"}   # тег → набор каналов; вероятности двух сетей усредняются
UNET_ARGS = ["--encoder", "resnet34", "--loss", "ce_dice", "--epochs", "60"]
# финал — БЕЗ TTA (прирост TTA < 0.005 BS-части, а без него результат не зависит от устройства); средние с TTA
# (avg14_*_tta.npy) тоже сохраняются — для режима с TTA: stack --unet avg14 → weights/bs_stack_tta.txt
AVG_TAG, STACK_TAG = "avg14nt", "avg14nt_event"
LGBM_ROUNDS_FULL = 85    # ≈ средняя best_iteration по фолдам
STACK_ROUNDS_FULL = 90


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
    oof = cache_dir() / "bs_oof"
    oof.mkdir(parents=True, exist_ok=True)
    (oof / "rules.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def run(data_dir: str, stages: list[str] | None = None, weights_dir: str | None = None,
        unet_loss: str | None = None) -> dict:
    """weights_dir — куда писать веса (по умолчанию weights/). Для другого каталога кэш и итоги тоже отдельные
    (data/cache/bs_<имя>/, reports/bs_results_<имя>.json): повторный прогон не затирает основную модель."""
    import os
    from pathlib import Path

    suffix = ""
    if weights_dir and Path(weights_dir).resolve() != (REPO_ROOT / "weights").resolve():
        wd = Path(weights_dir)
        wd = wd if wd.is_absolute() else REPO_ROOT / wd
        wd.mkdir(parents=True, exist_ok=True)
        suffix = "_" + wd.name
        os.environ["FIREMON_BS_WEIGHTS_DIR"] = str(wd)
        os.environ["FIREMON_BS_CACHE_DIR"] = str(REPO_ROOT / "data" / "cache" / f"bs{suffix}")
    from . import ensemble, lgbm, stack, unet  # импорт ПОСЛЕ установки каталогов: пути читаются при импорте

    OOF = cache_dir() / "bs_oof"
    OOF.mkdir(parents=True, exist_ok=True)

    seed_everything(42)
    stages = stages or STAGES
    out: dict = {}
    if "rules" in stages:
        out["rules"] = stage_rules(data_dir)
    if "lgbm" in stages:
        lgbm.main(["--data-dir", data_dir, "--stage", "sample"])
        out["lgbm"] = lgbm.main(["--data-dir", data_dir, "--stage", "cv", "--tag", "main"])
        lgbm.main(["--data-dir", data_dir, "--stage", "full", "--rounds", str(LGBM_ROUNDS_FULL)])
    if "unet" in stages:
        for tag, channels in UNETS.items():
            for split in ("group", "temporal", "full"):
                uargs = list(UNET_ARGS)
                if unet_loss:  # эксперименты с лоссом (например ce_lovasz) — только вместе с отдельным --weights-dir
                    uargs[uargs.index("--loss") + 1] = unet_loss
                r = unet.main(["--data-dir", data_dir, "--split", split, "--tag", tag, "--channels", channels, *uargs])
                out[f"unet_{tag}_{split}"] = {k: v for k, v in r.items() if k != "history"}
        for split in ("group", "temporal"):
            for sfx, name in (("", f"{AVG_TAG}_{split}.npy"), ("_tta", f"avg14_{split}_tta.npy")):
                avg = sum(np.load(OOF / f"{tag}_{split}{sfx}.npy").astype(np.float32) for tag in UNETS) / len(UNETS)
                np.save(OOF / name, avg.astype(np.float16))
    if "stack" in stages:
        out["stack"] = stack.main(["--data-dir", data_dir, "--stage", "cv", "--unet", AVG_TAG, "--unet-suffix", "",
                                   "--tag", STACK_TAG])
        stack.main(["--data-dir", data_dir, "--stage", "full", "--unet", AVG_TAG, "--unet-suffix", "",
                    "--rounds", str(STACK_ROUNDS_FULL)])
    if "ensemble" in stages:
        out["ensemble"] = ensemble.main(["--data-dir", data_dir, "--unet", "r34_cedice", "--lgbm", "main",
                                         "--stack", STACK_TAG])
    if "ms_unet" in stages:
        for tag, scale in MS_UNETS.items():
            for split in ("group", "temporal", "full"):
                r = unet.main(["--data-dir", data_dir, "--split", split, "--tag", tag, "--channels", "v4",
                               "--scale", str(scale), *UNET_ARGS])
                out[f"unet_{tag}_{split}"] = {k: v for k, v in r.items() if k != "history"}
    if "ms_stack" in stages:
        for split in ("group", "temporal"):
            base = sum(np.load(OOF / f"{t}_{split}.npy").astype(np.float32) for t in MS_BASE) / len(MS_BASE)
            aux = [(1.0 - np.load(OOF / f"{t}_{split}.npy")[:, 0].astype(np.float32))[:, None] for t in MS_AUX]
            np.save(OOF / f"{MS_OOF_TAG}_{split}.npy", np.concatenate([base, *aux], axis=1).astype(np.float16))
        common = ["--data-dir", data_dir, "--unet", MS_OOF_TAG, "--unet-suffix", "", "--gate", MS_GATE, "--frac", MS_FRAC]
        out["ms_stack"] = stack.main([*common, "--stage", "cv", "--tag", MS_STACK_TAG])
        stack.main([*common, "--stage", "full", "--rounds", str(STACK_ROUNDS_FULL), "--out-name", "bs_stack_ms.txt"])
        out["ms_ensemble"] = ensemble.main(["--data-dir", data_dir, "--stack", MS_STACK_TAG, "--stack-only",
                                            "--decode-name", "bs_decode_ms.json"])
    path = REPO_ROOT / "reports" / f"bs_results{suffix}.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=1, default=float), encoding="utf-8")
    log.info("итоги → %s", path)
    return out
