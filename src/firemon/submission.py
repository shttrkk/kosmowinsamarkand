"""Сборка и валидация submission.csv строго по шаблону организаторов (постановка кейса, раздел «Формат submission.csv»).

CLI: python -m firemon.submission --validate out/submission.csv --sample baseline/sample_submission.csv --data-dir data/test
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np

from .rle import rle_decode, rle_encode, validate_rle

csv.field_size_limit(2**31 - 1)  # RLE шумной BS-маски легко длиннее стандартного лимита 131072 символа

COLUMNS = ["chip_id", "class_id", "rle"]
DEFAULT_SHAPES = {"af": (256, 256), "bs": (512, 512)}


def chip_kind(chip_id: str) -> str:
    """'af' | 'bs' по префиксу chip_id (AF_tr_000001, BS_te_000010, ...)."""
    prefix = str(chip_id)[:2].upper()
    if prefix not in ("AF", "BS"):
        raise ValueError(f"не могу определить тип чипа по id: {chip_id!r}")
    return prefix.lower()


def read_submission(csv_path: str | Path) -> dict[str, list]:
    """Читает csv без pandas-магии с NaN: всё строками, пустой rle остаётся ''.

    Принимает оба стиля: rle в кавычках (`"1 3"`, `""`) и без (`1 3`, пустое поле), переводы строк LF и CRLF.
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or [h.strip() for h in header] != COLUMNS:
            raise ValueError(f"{csv_path}: ожидается заголовок {','.join(COLUMNS)}, получено {header}")
        rows = [r for r in reader if r]
    bad = [i + 2 for i, r in enumerate(rows) if len(r) != 3]
    if bad:
        raise ValueError(f"{csv_path}: строки с числом полей != 3: {bad[:5]}")
    return {"chip_id": [r[0] for r in rows], "class_id": [r[1] for r in rows], "rle": [r[2] for r in rows]}


def write_submission(rows: list[tuple[str, int, str]], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Текстовый формат — как в baseline/example_submission.csv организаторов: без кавычек (QUOTE_MINIMAL; в rle нет
    # запятых и кавычек), пустая маска — пустое поле, переводы строк CRLF, порядок строк — как передан (= шаблон).
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        for chip_id, class_id, rle in rows:
            writer.writerow([chip_id, int(class_id), rle])


def submission_rows(af_preds: Mapping[str, np.ndarray], bs_preds: Mapping[str, np.ndarray]) -> list[tuple[str, int, str]]:
    """Строки submission без шаблона: AF — class 1, BS — class 1/2/3, порядок — по chip_id (детерминированно)."""
    rows = [(c, 1, rle_encode(np.asarray(af_preds[c]) == 1)) for c in sorted(af_preds)]
    rows += [(c, k, rle_encode(np.asarray(bs_preds[c]) == k)) for c in sorted(bs_preds) for k in (1, 2, 3)]
    return rows


def build_submission(af_preds: Mapping[str, np.ndarray], bs_preds: Mapping[str, np.ndarray],
                     sample_submission_csv: str | Path, out_csv: str | Path, strict: bool = True) -> int:
    """Пишет submission с тем же набором и порядком пар (chip_id, class_id), что в шаблоне.

    af_preds: chip_id → маска 0/1; bs_preds: chip_id → маска 0..3 (разбивается на три бинарные).
    strict=False: чип без предсказания получает пустую маску вместо исключения.
    Возвращает число записанных строк.
    """
    template = read_submission(sample_submission_csv)
    cache: dict[tuple[str, int], str] = {}
    rows = []
    for cid, cls in zip(template["chip_id"], template["class_id"]):
        cls = int(cls)
        preds = af_preds if chip_kind(cid) == "af" else bs_preds
        if cid not in preds:
            if strict:
                raise KeyError(f"нет предсказания для чипа {cid}")
            rows.append((cid, cls, ""))
            continue
        if (cid, cls) not in cache:
            cache[(cid, cls)] = rle_encode(np.asarray(preds[cid]) == cls)
        rows.append((cid, cls, cache[(cid, cls)]))
    write_submission(rows, out_csv)
    return len(rows)


def validate_submission(csv_path: str | Path, sample_submission_csv: str | Path,
                        shapes: Mapping[str, tuple[int, int]] | None = None) -> list[str]:
    """Все проверки формата из ТЗ. Возвращает список ошибок (пустой = файл корректен).

    shapes: chip_id → (H, W); для отсутствующих — 256×256 (AF) / 512×512 (BS).
    """
    errors: list[str] = []
    try:
        sub = read_submission(csv_path)
        template = read_submission(sample_submission_csv)
    except (ValueError, OSError) as e:
        return [str(e)]
    shapes = dict(shapes or {})

    n, n_t = len(sub["chip_id"]), len(template["chip_id"])
    if n != n_t:
        errors.append(f"число строк {n} != {n_t} в шаблоне")

    pairs = list(zip(sub["chip_id"], sub["class_id"]))
    pairs_t = set(zip(template["chip_id"], template["class_id"]))
    seen: set = set()
    dups = {p for p in pairs if p in seen or seen.add(p)}
    if dups:
        errors.append(f"дубликаты пар (chip_id, class_id): {sorted(dups)[:5]}")
    missing, extra = pairs_t - set(pairs), set(pairs) - pairs_t
    if missing:
        errors.append(f"пропущено {len(missing)} пар из шаблона, например {sorted(missing)[:5]}")
    if extra:
        errors.append(f"лишние {len(extra)} пар, которых нет в шаблоне, например {sorted(extra)[:5]}")

    masks: dict[str, dict[str, np.ndarray]] = {}
    for line, (cid, cls, rle) in enumerate(zip(sub["chip_id"], sub["class_id"], sub["rle"]), start=2):
        if rle.strip().lower() in ("nan", "none", "null"):
            errors.append(f"строка {line} ({cid}, {cls}): NaN/пропуск в rle — пустая маска кодируется пустым полем")
            continue
        try:
            shape = shapes.get(cid) or DEFAULT_SHAPES[chip_kind(cid)]
            validate_rle(rle, shape)
        except ValueError as e:
            errors.append(f"строка {line} ({cid}, {cls}): {e}")
            continue
        if chip_kind(cid) == "bs" and rle:
            masks.setdefault(cid, {})[cls] = rle_decode(rle, shape, validate=False)

    for cid, by_cls in masks.items():
        keys = sorted(by_cls)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                n_overlap = int(np.count_nonzero(by_cls[a] & by_cls[b]))
                if n_overlap:
                    errors.append(f"{cid}: маски классов {a} и {b} пересекаются ({n_overlap} px)")
    return errors


def shapes_from_meta(data_dir: str | Path) -> dict[str, tuple[int, int]]:
    """chip_id → (height, width) из meta.csv обоих модулей (если есть)."""
    from .io import read_meta

    shapes: dict[str, tuple[int, int]] = {}
    for kind in ("af", "bs"):
        try:
            meta = read_meta(data_dir, kind)
        except FileNotFoundError:
            continue
        if {"height", "width"} <= set(meta.columns):
            shapes.update({c: (int(h), int(w)) for c, h, w in zip(meta["chip_id"], meta["height"], meta["width"])})
    return shapes


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Валидация submission.csv")
    ap.add_argument("--validate", required=True, metavar="CSV")
    ap.add_argument("--sample", required=True, help="sample_submission.csv организаторов")
    ap.add_argument("--data-dir", default=None, help="каталог теста (размеры чипов берутся из meta.csv)")
    args = ap.parse_args()
    errors = validate_submission(args.validate, args.sample, shapes_from_meta(args.data_dir) if args.data_dir else None)
    if errors:
        print(f"ОШИБОК: {len(errors)}")
        for e in errors[:50]:
            print(" -", e)
        return 1
    print(f"OK: {args.validate} корректен ({len(read_submission(args.validate)['chip_id'])} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
