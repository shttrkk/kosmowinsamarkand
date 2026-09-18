"""Чувствительность BS-части к порогу t_burn: оптимум ±0.05 и ±0.10 на обоих сплитах (OOF финального стека).

    python scripts/bs_tburn_sensitivity.py

Множители классов и постобработка фиксированы (как в финальной модели). Результат → reports/bs_tburn_sensitivity.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.models.bs.common import decode_probs, evaluate_masks, get_splits, load_chips  # noqa: E402
from firemon.models.bs.postprocess import postprocess_bs  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--probs", default="stack_avg14_event")
    ap.add_argument("--ensemble", default="ensemble_r34_cedice_main_avg14_event.json")
    args = ap.parse_args()

    ens = json.loads((OOF / args.ensemble).read_text(encoding="utf-8"))
    post = ens["postprocess"]
    decs = {"group": ens["final"]["decode_group"], "temporal": ens["final"]["decode_temporal"]}
    chips = load_chips(args.data_dir)
    ids = [c["chip_id"] for c in chips]
    idx = {"group": np.arange(len(ids)), "temporal": get_splits(ids)["temporal"][0][1]}
    res: dict = {"probs": args.probs, "postprocess": post, "splits": {}}
    for split in ("group", "temporal"):
        probs = np.load(OOF / f"{args.probs}_{split}.npy", mmap_mode="r")
        t0, w = float(decs[split]["t_burn"]), decs[split]["w"]
        rows = {}
        for d in (-0.10, -0.05, 0.0, 0.05, 0.10):
            preds = {}
            for i in idx[split]:
                p = np.asarray(probs[i], np.float32)
                preds[i] = postprocess_bs(decode_probs(p, t0 + d, w), chips[i], post, probs=p)
            rows[f"{d:+.2f}"] = evaluate_masks(preds, {i: chips[i]["mask"] for i in idx[split]})
        res["splits"][split] = {"t_opt": t0, "w": w, "rows": rows}
        ref = rows["+0.00"]["bs_part"]
        for k, m in rows.items():
            print(f"{split:9s} t={t0 + float(k):.2f} ({k}) IoU_burn {m['iou_burn']:.4f} mIoU_sev {m['miou_sev']:.4f} "
                  f"BS {m['bs_part']:.4f} Δ{m['bs_part'] - ref:+.4f}")
    (ROOT / "reports" / "bs_tburn_sensitivity.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
