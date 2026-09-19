"""Метрика соревнования (постановка кейса, раздел «Метрика оценки»): F1_af, IoU_burn, mIoU_sev, Score.

Микро-усреднение: TP/FP/FN накапливаются по пикселям всех чипов, величины
считаются один раз по общему пулу. Знаменатель 0 → величина = 1.
Пиксели эталона со значением 255 (nodata) исключаются из подсчёта — переключатель
`ignore_nodata` (default True). Официального скрипта метрики организаторы не выдают —
реализация строго по тексту ТЗ и является окончательной.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

NODATA = 255
W_AF, W_BURN, W_SEV = 0.35, 0.35, 0.30


def _ratio(num: float, den: float) -> float:
    return 1.0 if den == 0 else float(num) / float(den)


def _counts(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.count_nonzero(pred & gt & valid))
    fp = int(np.count_nonzero(pred & ~gt & valid))
    fn = int(np.count_nonzero(~pred & gt & valid))
    return tp, fp, fn


def _check(preds: Mapping[str, np.ndarray], gts: Mapping[str, np.ndarray]) -> None:
    missing = set(gts) - set(preds)
    if missing:
        raise KeyError(f"нет предсказаний для {len(missing)} чипов, например {sorted(missing)[:3]}")


def af_f1(preds: Mapping[str, np.ndarray], gts: Mapping[str, np.ndarray], ignore_nodata: bool = True) -> dict:
    """Микро-F1 по пикселям класса 1 на AF-чипах. Оценка идёт по чипам из `gts`."""
    _check(preds, gts)
    tp = fp = fn = 0
    for cid, gt in gts.items():
        gt = np.asarray(gt)
        pred = np.asarray(preds[cid])
        if pred.shape != gt.shape:
            raise ValueError(f"{cid}: форма предсказания {pred.shape} != эталона {gt.shape}")
        valid = gt != NODATA if ignore_nodata else np.ones(gt.shape, bool)
        a, b, c = _counts(pred == 1, gt == 1, valid)
        tp, fp, fn = tp + a, fp + b, fn + c
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def bs_metrics(preds: Mapping[str, np.ndarray], gts: Mapping[str, np.ndarray], ignore_nodata: bool = True) -> dict:
    """IoU_burn (pred>0 vs gt>0) и IoU по классам 1..3, микро по всем BS-чипам из `gts`."""
    _check(preds, gts)
    cnt = {k: [0, 0, 0] for k in ("burn", 1, 2, 3)}
    for cid, gt in gts.items():
        gt = np.asarray(gt)
        pred = np.asarray(preds[cid])
        if pred.shape != gt.shape:
            raise ValueError(f"{cid}: форма предсказания {pred.shape} != эталона {gt.shape}")
        valid = gt != NODATA if ignore_nodata else np.ones(gt.shape, bool)
        gt_burn = (gt >= 1) & (gt <= 3)
        for key, p, g in [("burn", pred > 0, gt_burn)] + [(k, pred == k, gt == k) for k in (1, 2, 3)]:
            for i, v in enumerate(_counts(p, g, valid)):
                cnt[key][i] += v
    iou = {k: _ratio(tp, tp + fp + fn) for k, (tp, fp, fn) in cnt.items()}
    return {
        "iou_burn": iou["burn"],
        "iou_1": iou[1], "iou_2": iou[2], "iou_3": iou[3],
        "miou_sev": (iou[1] + iou[2] + iou[3]) / 3.0,
        "counts": {str(k): dict(zip(("tp", "fp", "fn"), v)) for k, v in cnt.items()},
    }


def score(af_f1_value: float, iou_burn: float, miou_sev: float) -> float:
    return W_AF * af_f1_value + W_BURN * iou_burn + W_SEV * miou_sev


def evaluate(af_preds: Mapping[str, np.ndarray], af_gts: Mapping[str, np.ndarray],
             bs_preds: Mapping[str, np.ndarray], bs_gts: Mapping[str, np.ndarray]) -> dict:
    """Полная метрика по словарям масок (удобно для OOF)."""
    af = af_f1(af_preds, af_gts)
    bs = bs_metrics(bs_preds, bs_gts)
    return {
        "score": score(af["f1"], bs["iou_burn"], bs["miou_sev"]),
        "f1_af": af["f1"], "precision_af": af["precision"], "recall_af": af["recall"],
        "iou_burn": bs["iou_burn"], "miou_sev": bs["miou_sev"],
        "iou_1": bs["iou_1"], "iou_2": bs["iou_2"], "iou_3": bs["iou_3"],
        "n_af": len(af_gts), "n_bs": len(bs_gts),
    }


def evaluate_submission(submission_csv: str | Path, data_dir_with_masks: str | Path,
                        sample_submission_csv: str | Path | None = None) -> dict:
    """Декодирует submission.csv и считает метрику по маскам из `data_dir_with_masks`.

    Проверяет всю цепочку «маска → RLE → csv → декод → метрика». Оцениваются чипы
    из шаблона (или из самого submission, если шаблон не задан); маски ищутся в
    `<data_dir>/af/masks` и `<data_dir>/bs/masks`.
    """
    from .io import read_mask
    from .rle import rle_decode
    from .submission import chip_kind, read_submission

    sub = read_submission(submission_csv)
    template = read_submission(sample_submission_csv) if sample_submission_csv else sub
    rles = {(c, int(k)): r for c, k, r in zip(sub["chip_id"], sub["class_id"], sub["rle"])}

    af_preds, af_gts, bs_preds, bs_gts = {}, {}, {}, {}
    for cid in dict.fromkeys(template["chip_id"]):
        kind = chip_kind(cid)
        gt = read_mask(data_dir_with_masks, kind, cid)
        if kind == "af":
            af_gts[cid] = gt
            af_preds[cid] = rle_decode(rles.get((cid, 1), ""), gt.shape)
        else:
            pred = np.zeros(gt.shape, np.uint8)
            for k in (1, 2, 3):
                pred[rle_decode(rles.get((cid, k), ""), gt.shape) == 1] = k
            bs_gts[cid] = gt
            bs_preds[cid] = pred
    return evaluate(af_preds, af_gts, bs_preds, bs_gts)


def _main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Оценка submission.csv по маскам")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--masks-dir", required=True, help="каталог с af/masks и bs/masks (например data/train)")
    ap.add_argument("--sample-submission", default=None)
    args = ap.parse_args()
    print(json.dumps(evaluate_submission(args.submission, args.masks_dir, args.sample_submission), indent=2))


if __name__ == "__main__":
    _main()
