"""Прирост BS-части от TTA (4 отражения UNet) на уровне ВСЕГО пайплайна: стек обучен и применён на OOF UNet
с TTA (stack_avg14_event) против без TTA (stack_avg14nt_event). Декодирование подбирается для каждого варианта
(для temporal — на групповом OOF чипов не из 2024), постобработка — финальная из weights/bs_decode.json.

    python scripts/bs_tta_check.py   → reports/bs_tta_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.models.bs.common import decode_probs, evaluate_masks, fit_decode, fmt, get_splits, load_chips  # noqa: E402
from firemon.models.bs.postprocess import postprocess_bs  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"


def main() -> None:
    post = json.loads((ROOT / "weights" / "bs_decode.json").read_text(encoding="utf-8"))["postprocess"]
    chips = load_chips(ROOT / "data" / "train")
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    t_tr, t_va = get_splits(ids)["temporal"][0]
    res: dict = {"postprocess": post}
    for name, tag in (("tta", "stack_avg14_event"), ("no_tta", "stack_avg14nt_event")):
        pg = np.load(OOF / f"{tag}_group.npy", mmap_mode="r")
        pt = np.load(OOF / f"{tag}_temporal.npy", mmap_mode="r")
        res[name] = {}
        for split, probs, fit_idx, val_idx in (("group", pg, np.arange(len(ids)), np.arange(len(ids))),
                                               ("temporal", pt, t_tr, t_va)):
            dec = fit_decode(pg[fit_idx], gts[fit_idx])
            preds = {}
            for i in val_idx:
                p = np.asarray(probs[i], np.float32)
                preds[i] = postprocess_bs(decode_probs(p, dec["t_burn"], dec["w"]), chips[i], post, probs=p, decode=dec)
            m = evaluate_masks(preds, {i: gts[i] for i in val_idx})
            res[name][split] = {**m, "decode": dec}
            print(f"{name:7s} {split:9s} {fmt(m)}  t_burn {dec['t_burn']:.2f} w {dec['w']}")
    res["gain"] = {s: res["tta"][s]["bs_part"] - res["no_tta"][s]["bs_part"] for s in ("group", "temporal")}
    print("прирост от TTA (BS-часть):", {k: round(v, 4) for k, v in res["gain"].items()})
    (ROOT / "reports" / "bs_tta_check.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
