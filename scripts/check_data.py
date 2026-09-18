"""Проверка комплектности данных.

    python scripts/check_data.py                 # data/train
    python scripts/check_data.py --split test    # data/test (+ sample_submission: 447 строк)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.io import LAYERS, MASK_LAYER, read_meta, read_tif, win_path  # noqa: E402

EXPECTED = {"train": {"af": 420, "bs": 224}, "test": {"af": 180, "bs": 89}}
EXPECTED_ROWS = {"test": 447}


def check_kind(data_dir: Path, kind: str, with_masks: bool) -> tuple[int, list[str]]:
    problems: list[str] = []
    meta = read_meta(data_dir, kind)
    chip_ids = meta["chip_id"].astype(str).tolist()
    if len(set(chip_ids)) != len(chip_ids):
        problems.append(f"{kind}: дубликаты chip_id в meta.csv")
    layers = dict(LAYERS[kind])
    if with_masks:
        layers["mask"] = MASK_LAYER
    print(f"\n== {kind.upper()}: {len(chip_ids)} чипов в meta.csv, колонки: {', '.join(meta.columns)}")
    for key, (sub, suffix) in layers.items():
        folder = data_dir / kind / sub
        if not os.path.isdir(win_path(folder)):
            problems.append(f"{kind}: нет каталога {folder}")
            continue
        files = {f for f in os.listdir(win_path(folder)) if f.lower().endswith(".tif")}
        expected = {f"{c}_{suffix}.tif" for c in chip_ids}
        missing, extra = expected - files, files - expected
        print(f"  {sub:<16} файлов: {len(files):>4}  нет: {len(missing)}  лишних: {len(extra)}")
        if missing:
            problems.append(f"{kind}/{sub}: нет {len(missing)} файлов, например {sorted(missing)[:3]}")
        if extra:
            problems.append(f"{kind}/{sub}: {len(extra)} файлов без строки в meta.csv, например {sorted(extra)[:3]}")
        if not missing and chip_ids:
            arr, names, _, crs = read_tif(folder / f"{chip_ids[0]}_{suffix}.tif", with_georef=True)
            print(f"  {'':<16} {arr.shape} {arr.dtype} crs={crs} каналы: {', '.join(names)}")
    return len(chip_ids), problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="train | test | имя каталога внутри data/")
    ap.add_argument("--data-dir", default=None, help="явный путь (по умолчанию data/<split>)")
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / args.split
    with_masks = args.split == "train"
    print(f"Проверка {data_dir} (маски: {'да' if with_masks else 'нет'})")

    problems: list[str] = []
    for kind in ("af", "bs"):
        try:
            n, p = check_kind(data_dir, kind, with_masks)
        except FileNotFoundError as e:
            problems.append(f"нет файла {e}")
            continue
        problems += p
        exp = EXPECTED.get(args.split, {}).get(kind)
        if exp is not None and n != exp:
            problems.append(f"{kind}: {n} чипов, ожидалось {exp}")

    for sample in (data_dir / "sample_submission.csv", ROOT / "baseline" / "sample_submission.csv"):
        if sample.exists() and args.split != "train":
            from firemon.submission import read_submission
            n_rows = len(read_submission(sample)["chip_id"])
            print(f"\n{sample}: {n_rows} строк")
            exp = EXPECTED_ROWS.get(args.split)
            if exp is not None and n_rows != exp:
                problems.append(f"{sample.name}: {n_rows} строк, ожидалось {exp}")
            break

    print("\n" + ("ВСЁ ОК" if not problems else "ПРОБЛЕМЫ:\n" + "\n".join(" - " + p for p in problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
