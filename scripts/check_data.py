"""Проверка комплектности данных.

    python scripts/check_data.py                 # data/train
    python scripts/check_data.py --split test    # data/test: общий meta.csv в корне, sample_submission (447 строк),
                                                 # chip_id meta == шаблон == файлы, 5+5 случайных чипов против train
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.io import (AF_AUX_BANDS, AF_VIIRS_BANDS, BS_AUX_BANDS, LAYERS, MASK_LAYER, S1_BANDS, S2_BANDS,  # noqa: E402
                        meta_path, read_meta, read_tif, win_path)
from firemon.submission import chip_kind, read_submission  # noqa: E402

EXPECTED = {"train": {"af": 420, "bs": 224}, "test": {"af": 180, "bs": 89}}
EXPECTED_ROWS = {"test": 447}
EXPECTED_CLASSES = {"test": {"1": 269, "2": 89, "3": 89}}
# эталон структуры файлов (как в train): слой → (C, H, W), dtype, имена каналов
REFERENCE = {
    "af": {"viirs": ((8, 256, 256), "float32", AF_VIIRS_BANDS), "aux": ((5, 256, 256), "float32", AF_AUX_BANDS)},
    "bs": {"s2_pre": ((10, 512, 512), "uint16", S2_BANDS), "s2_post": ((10, 512, 512), "uint16", S2_BANDS),
           "s1_pre": ((2, 512, 512), "int16", S1_BANDS), "s1_post": ((2, 512, 512), "int16", S1_BANDS),
           "aux": ((3, 512, 512), "int16", BS_AUX_BANDS)},
}


def check_kind(data_dir: Path, kind: str, with_masks: bool) -> tuple[int, list[str]]:
    problems: list[str] = []
    meta = read_meta(data_dir, kind)
    chip_ids = meta["chip_id"].astype(str).tolist()
    if len(set(chip_ids)) != len(chip_ids):
        problems.append(f"{kind}: дубликаты chip_id в meta.csv")
    layers = dict(LAYERS[kind])
    if with_masks:
        layers["mask"] = MASK_LAYER
    print(f"\n== {kind.upper()}: {len(chip_ids)} чипов в {meta_path(data_dir, kind)[0]}, "
          f"колонки: {', '.join(meta.columns)}")
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


def check_random_chips(data_dir: Path, kind: str, n: int = 5, seed: int = 42) -> list[str]:
    """n случайных чипов: форма, dtype и band descriptions каждого слоя должны совпадать с train."""
    problems: list[str] = []
    chip_ids = read_meta(data_dir, kind)["chip_id"].astype(str).tolist()
    chosen = sorted(np.random.RandomState(seed).choice(chip_ids, size=min(n, len(chip_ids)), replace=False).tolist())
    for cid in chosen:
        for key, (sub, suffix) in LAYERS[kind].items():
            arr, names, _, _ = read_tif(data_dir / kind / sub / f"{cid}_{suffix}.tif", with_georef=True)
            got, ref = (arr.shape, str(arr.dtype), names), REFERENCE[kind][key]
            if got != ref:
                problems.append(f"{cid}/{key}: {got} != train {ref}")
    verdict = "совпадают с train" if not problems else "ОТЛИЧАЮТСЯ от train"
    print(f"  случайные чипы {', '.join(chosen)}: формы/dtype/каналы {verdict}")
    return problems


def check_template(sample: Path, data_dir: Path, split: str) -> list[str]:
    """Шаблон: число строк и классов, chip_id совпадают с meta.csv, у AF одна строка, у BS три."""
    problems: list[str] = []
    tpl = read_submission(sample)
    n_rows, classes = len(tpl["chip_id"]), Counter(tpl["class_id"])
    print(f"\n{sample}: {n_rows} строк, class_id: {dict(sorted(classes.items()))}, "
          f"непустых rle: {sum(bool(r.strip()) for r in tpl['rle'])}")
    exp = EXPECTED_ROWS.get(split)
    if exp is not None and n_rows != exp:
        problems.append(f"{sample.name}: {n_rows} строк, ожидалось {exp}")
    exp_cls = EXPECTED_CLASSES.get(split)
    if exp_cls is not None and dict(classes) != exp_cls:
        problems.append(f"{sample.name}: классы {dict(classes)}, ожидалось {exp_cls}")
    by_chip: dict[str, list[str]] = {}
    for cid, cls in zip(tpl["chip_id"], tpl["class_id"]):
        by_chip.setdefault(cid, []).append(cls)
    for kind, want in (("af", ["1"]), ("bs", ["1", "2", "3"])):
        ids_t = {c for c in by_chip if chip_kind(c) == kind}
        ids_m = set(read_meta(data_dir, kind)["chip_id"].astype(str))
        if ids_t != ids_m:
            problems.append(f"{sample.name}: chip_id {kind} не совпадают с meta.csv "
                            f"(только в шаблоне {len(ids_t - ids_m)}, только в meta {len(ids_m - ids_t)})")
        bad = [c for c in ids_t if sorted(by_chip[c]) != want]
        if bad:
            problems.append(f"{sample.name}: у {len(bad)} {kind}-чипов class_id != {want}, например {sorted(bad)[:3]}")
    if not problems:
        print("  chip_id шаблона == meta.csv; AF — по 1 строке, BS — по 3")
    return problems


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
        if not p and not with_masks:
            problems += check_random_chips(data_dir, kind)
        exp = EXPECTED.get(args.split, {}).get(kind)
        if exp is not None and n != exp:
            problems.append(f"{kind}: {n} чипов, ожидалось {exp}")

    if args.split != "train":
        for sample in (data_dir / "sample_submission.csv", ROOT / "baseline" / "sample_submission.csv"):
            if sample.exists():
                problems += check_template(sample, data_dir, args.split)
                break
        else:
            if args.split == "test":
                problems.append("sample_submission.csv не найден ни в <data-dir>, ни в baseline/")

    print("\n" + ("ВСЁ ОК" if not problems else "ПРОБЛЕМЫ:\n" + "\n".join(" - " + p for p in problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
