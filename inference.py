"""Единственная точка входа инференса (ТЗ):

    python inference.py --data-dir /path/to/test --output /path/to/submission.csv

Читает чипы из <data-dir>/af и <data-dir>/bs, пишет submission.csv по шаблону
sample_submission.csv. Без интерактива, код выхода 0 при успехе.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

T_START = time.perf_counter()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # работает и без `pip install -e .`

import numpy as np  # noqa: E402

from firemon.io import list_chips, read_af_chip, read_bs_chip  # noqa: E402
from firemon.models.af.rules import predict_af_rules  # noqa: E402
from firemon.models.bs.rules import predict_bs_rules  # noqa: E402
from firemon.submission import (DEFAULT_SHAPES, build_submission, read_submission,  # noqa: E402
                                shapes_from_meta, validate_submission, write_submission)
from firemon.utils import get_logger, load_config, seed_everything  # noqa: E402

log = get_logger("inference")


def find_sample_submission(data_dir: Path, explicit: str | None) -> Path | None:
    candidates = [Path(explicit)] if explicit else [data_dir / "sample_submission.csv",
                                                    ROOT / "baseline" / "sample_submission.csv"]
    for c in candidates:
        if c.exists():
            return c
    if explicit:
        raise FileNotFoundError(f"sample submission не найден: {explicit}")
    return None


def chips_of_kind(data_dir: Path, kind: str, template_ids: list[str] | None) -> list[str]:
    """Чипы к обработке: из шаблона (он определяет состав submission), иначе из meta.csv/каталога."""
    if template_ids is not None:
        return [c for c in template_ids if c[:2].lower() == kind]
    try:
        return list_chips(data_dir, kind)
    except FileNotFoundError:
        return []


def run(data_dir: Path, output: Path, sample: Path | None, cfg: dict) -> int:
    template_ids = None
    if sample is not None:
        template_ids = list(dict.fromkeys(read_submission(sample)["chip_id"]))
    af_cfg, bs_cfg = cfg.get("af", {}), cfg.get("bs", {})
    shapes = shapes_from_meta(data_dir)
    n_failed = 0

    t0 = time.perf_counter()
    af_preds: dict[str, np.ndarray] = {}
    for cid in chips_of_kind(data_dir, "af", template_ids):
        try:
            chip = read_af_chip(data_dir, cid, with_mask=False, layers=("viirs",))
            af_preds[cid] = predict_af_rules(chip, af_cfg.get("i4_min", 320.0), af_cfg.get("di45_min", 15.0))
        except Exception as e:  # один битый чип не должен ронять весь submission
            n_failed += 1
            log.error("AF %s: %s — пустая маска", cid, e)
            af_preds[cid] = np.zeros(shapes.get(cid, DEFAULT_SHAPES["af"]), np.uint8)
    t_af = time.perf_counter() - t0

    t0 = time.perf_counter()
    bs_ids = chips_of_kind(data_dir, "bs", template_ids)
    if bs_cfg.get("model", "unet") == "rules":  # аварийный режим без моделей
        bs_preds: dict[str, np.ndarray] = {}
        thresholds = tuple(bs_cfg.get("dnbr_thresholds", (0.10, 0.27, 0.44)))
        for cid in bs_ids:
            try:
                chip = read_bs_chip(data_dir, cid, with_mask=False, layers=("s2_pre", "s2_post", "aux"))
                bs_preds[cid] = predict_bs_rules(chip, thresholds)
            except Exception as e:
                n_failed += 1
                log.error("BS %s: %s — пустая маска", cid, e)
                bs_preds[cid] = np.zeros(shapes.get(cid, DEFAULT_SHAPES["bs"]), np.uint8)
    else:  # UNet (+LGBM) — конфиг configs/bs_inference.yaml, секция bs общего конфига его перекрывает
        from firemon.models.bs.infer import load_bs_config, run_bs_inference
        bs_preds, bs_failed, bs_timing = run_bs_inference(data_dir, bs_ids, load_bs_config(bs_cfg), shapes)
        n_failed += bs_failed
        log.info("BS: загрузка моделей %.2f с, чтение+предсказание %.2f с", bs_timing["load"], bs_timing["predict"])
    t_bs = time.perf_counter() - t0

    t0 = time.perf_counter()
    if sample is not None:
        n_rows = build_submission(af_preds, bs_preds, sample, output, strict=False)
    else:
        log.warning("sample_submission.csv не найден — строки формируются по списку чипов")
        from firemon.rle import rle_encode
        rows = [(c, 1, rle_encode(m == 1)) for c, m in af_preds.items()]
        rows += [(c, k, rle_encode(m == k)) for c, m in bs_preds.items() for k in (1, 2, 3)]
        write_submission(rows, output)
        n_rows = len(rows)
    errors = validate_submission(output, sample or output, shapes)
    t_sub = time.perf_counter() - t0

    log.info("AF: %d чипов за %.2f с | BS: %d чипов за %.2f с | сборка+валидация: %.2f с",
             len(af_preds), t_af, len(bs_preds), t_bs, t_sub)
    log.info("submission: %s (%d строк), ошибок чтения чипов: %d", output, n_rows, n_failed)
    for e in errors[:20]:
        log.error("валидация: %s", e)
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Инференс AF + BS → submission.csv")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sample-submission", default=None,
                    help="по умолчанию <data-dir>/sample_submission.csv, затем baseline/sample_submission.csv")
    ap.add_argument("--config", default=str(ROOT / "configs" / "inference.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)), with_torch=False)  # заглушки на правилах torch не используют
    sample = find_sample_submission(args.data_dir, args.sample_submission)
    code = run(args.data_dir, args.output, sample, cfg)
    log.info("Полное время инференса: %.2f с", time.perf_counter() - T_START)
    return code


if __name__ == "__main__":
    sys.exit(main())
