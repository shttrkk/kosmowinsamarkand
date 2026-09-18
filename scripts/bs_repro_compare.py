"""Воспроизводимость обучения BS: OOF-метрики повторного прогона (`train.py --module bs --weights-dir weights_repro/`)
против основной модели. → reports/bs_repro_compare.json + таблица в stdout.

    python scripts/bs_repro_compare.py [--repro-name weights_repro]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro-name", default="weights_repro")
    args = ap.parse_args()
    a_dir = ROOT / "data" / "cache" / "bs_oof"
    b_dir = ROOT / "data" / "cache" / f"bs_{args.repro_name}" / "bs_oof"
    items = [  # (название, файл, путь к метрикам group, путь к метрикам temporal)
        ("правила по landcover", "rules.json", ("group", "by_landcover"), ("temporal", "by_landcover")),
        ("LightGBM + декодирование", "lgbm_main.json", ("group", "tuned"), ("temporal", "tuned")),
        ("UNet v1, argmax+TTA", "r34_cedice_{split}.json", ("argmax+TTA",), ("argmax+TTA",)),
        ("UNet v4, argmax+TTA", "r34_v4_{split}.json", ("argmax+TTA",), ("argmax+TTA",)),
        ("стек (TTA) + декодирование", "stack_avg14_event.json", ("group", "tuned"), ("temporal", "tuned")),
        ("стек (TTA) + постобработка", "ensemble_r34_cedice_main_avg14_event.json", ("final", "group"), ("final", "temporal")),
        ("стек без TTA + декодирование", "stack_avg14nt_event.json", ("group", "tuned"), ("temporal", "tuned")),
        ("стек без TTA + постобработка (финал)", "ensemble_r34_cedice_main_avg14nt_event.json", ("final", "group"),
         ("final", "temporal")),
    ]

    def get(d: Path, fname: str, split: str, path: tuple):
        r = load(d / fname.format(split=split))
        if r is None:
            return None
        for k in path:
            r = r[k]
        return r

    out, worst = [], 0.0
    print(f"{'этап':40s} | {'group: основная / повтор (Δ)':34s} | temporal: основная / повтор (Δ)")
    for name, fname, pg, pt in items:
        row = {"stage": name}
        cells = []
        for split, path in (("group", pg), ("temporal", pt)):
            a, b = get(a_dir, fname, split, path), get(b_dir, fname, split, path)
            if a is None or b is None:
                cells.append("—")
                continue
            d = b["bs_part"] - a["bs_part"]
            worst = max(worst, abs(d))
            row[split] = {"main": a["bs_part"], "repro": b["bs_part"], "delta": d,
                          "main_iou_burn": a["iou_burn"], "repro_iou_burn": b["iou_burn"],
                          "main_miou_sev": a["miou_sev"], "repro_miou_sev": b["miou_sev"]}
            cells.append(f"{a['bs_part']:.4f} / {b['bs_part']:.4f} ({d:+.4f})")
        out.append(row)
        print(f"{name:40s} | {cells[0]:34s} | {cells[1]}")
    print(f"макс. |Δ BS-части| = {worst:.4f} → {'в пределах ±0.005' if worst <= 0.005 else 'ПРЕВЫШАЕТ ±0.005'}")
    (ROOT / "reports" / "bs_repro_compare.json").write_text(
        json.dumps({"rows": out, "max_abs_delta": worst}, indent=1, default=float, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
