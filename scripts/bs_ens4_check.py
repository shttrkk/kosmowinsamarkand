"""Гипотеза ens4: стек на среднем ЧЕТЫРЁХ UNet (weights/ v1+v4 и weights_repro/ v1+v4, без TTA) против текущего ens2.
Декодирование подбирается для каждого варианта (temporal — на групповом OOF чипов не из 2024), постобработка — та же
(weights/bs_decode.json). Правило принятия: прирост BS-части ≥ 0.005 на ОБОИХ сплитах.

    python scripts/bs_ens4_check.py   → reports/bs_ens4_check.json
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
VARIANTS = {"ens2 (основная модель)": OOF / "stack_avg14nt_event",
            "ens2 (контрольный прогон weights_repro)": ROOT / "data" / "cache" / "bs_weights_repro" / "bs_oof" / "stack_avg14nt_event",
            "ens4": OOF / "stack_ens4nt_event"}


def main() -> None:
    post = json.loads((ROOT / "weights" / "bs_decode.json").read_text(encoding="utf-8"))["postprocess"]
    chips = load_chips(ROOT / "data" / "train")
    ids = [c["chip_id"] for c in chips]
    gts = np.stack([c["mask"] for c in chips])
    t_tr, t_va = get_splits(ids)["temporal"][0]
    cloudy = [i for i, c in enumerate(chips) if float(c["meta"].get("cloud_frac") or 0) > 0.25]
    res: dict = {"postprocess": post, "variants": {}}
    for name, prefix in VARIANTS.items():
        pg = np.load(f"{prefix}_group.npy", mmap_mode="r")
        pt = np.load(f"{prefix}_temporal.npy", mmap_mode="r")
        out = {}
        for split, probs, fit_idx, val_idx in (("group", pg, np.arange(len(ids)), np.arange(len(ids))),
                                               ("temporal", pt, t_tr, t_va)):
            dec = fit_decode(pg[fit_idx], gts[fit_idx])
            preds = {}
            for i in val_idx:
                p = np.asarray(probs[i], np.float32)
                preds[i] = postprocess_bs(decode_probs(p, dec["t_burn"], dec["w"]), chips[i], post, probs=p, decode=dec)
            out[split] = {**evaluate_masks(preds, {i: gts[i] for i in val_idx}), "decode": dec}
            if split == "group":
                out["cloudy"] = evaluate_masks({i: preds[i] for i in cloudy}, {i: gts[i] for i in cloudy})
        res["variants"][name] = out
        for s in ("group", "temporal", "cloudy"):
            print(f"{name:42s} {s:9s} {fmt(out[s])}")
    base, new = res["variants"]["ens2 (основная модель)"], res["variants"]["ens4"]
    res["gain_vs_main"] = {s: new[s]["bs_part"] - base[s]["bs_part"] for s in ("group", "temporal", "cloudy")}
    res["accepted"] = bool(res["gain_vs_main"]["group"] >= 0.005 and res["gain_vs_main"]["temporal"] >= 0.005)
    print("прирост ens4 к основной модели:", {k: round(v, 4) for k, v in res["gain_vs_main"].items()},
          "→", "ПРИНЯТО" if res["accepted"] else "НЕ принято")
    (ROOT / "reports" / "bs_ens4_check.json").write_text(json.dumps(res, indent=1, default=float, ensure_ascii=False),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
