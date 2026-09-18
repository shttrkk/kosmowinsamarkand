"""Приор «событие есть в каждом чипе» (postprocess.nonempty_prior): на каких OOF-чипах срабатывает и что даёт.

    python scripts/bs_nonempty_prior_check.py [--probs stack_avg14_event]
→ reports/bs_nonempty_prior.json
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
    post = {k: v for k, v in ens["postprocess"].items() if k != "nonempty_min_px"}
    decs = {"group": ens["final"]["decode_group"], "temporal": ens["final"]["decode_temporal"]}
    chips = load_chips(args.data_dir)
    ids = [c["chip_id"] for c in chips]
    idx = {"group": np.arange(len(ids)), "temporal": get_splits(ids)["temporal"][0][1]}
    res: dict = {}
    for split in ("group", "temporal"):
        probs = np.load(OOF / f"{args.probs}_{split}.npy", mmap_mode="r")
        base, prior, fired = {}, {}, []
        for i in idx[split]:
            p = np.asarray(probs[i], np.float32)
            m = decode_probs(p, decs[split]["t_burn"], decs[split]["w"])
            base[i] = postprocess_bs(m, chips[i], post, probs=p, decode=decs[split])
            prior[i] = postprocess_bs(m, chips[i], {**post, "nonempty_min_px": 25}, probs=p, decode=decs[split])
            if int(np.count_nonzero(base[i])) < 25:
                gt = chips[i]["mask"]
                fired.append({"chip_id": ids[i], "gt_burn_px": int((gt > 0).sum()), "pred_px_before": int((base[i] > 0).sum()),
                              "pred_px_after": int((prior[i] > 0).sum()),
                              "tp_after": int(((prior[i] > 0) & (gt > 0)).sum())})
        gts = {i: chips[i]["mask"] for i in idx[split]}
        mb, mp = evaluate_masks(base, gts), evaluate_masks(prior, gts)
        res[split] = {"n_chips": int(len(idx[split])), "fired": fired, "base": mb, "with_prior": mp,
                      "d_bs_part": mp["bs_part"] - mb["bs_part"]}
        print(split, "сработало на", len(fired), "чипах из", len(idx[split]), "| Δ BS-части %+.5f" % res[split]["d_bs_part"])
        for f in fired:
            print("   ", f)
    (ROOT / "reports" / "bs_nonempty_prior.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
