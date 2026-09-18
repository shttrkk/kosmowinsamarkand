"""Проверка submission.csv перед отправкой: валидатор формата + сводка по содержимому.

    python scripts/check_submission.py out/submission.csv
    python scripts/check_submission.py out/submission.csv --sample data/test/sample_submission.csv --data-dir data/test

Код выхода 0 — файл корректен, 1 — есть ошибки формата.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.rle import rle_pixel_count  # noqa: E402
from firemon.submission import chip_kind, read_submission, shapes_from_meta, validate_submission  # noqa: E402


def find_sample(data_dir: Path) -> Path | None:
    for c in (data_dir / "sample_submission.csv", ROOT / "baseline" / "sample_submission.csv"):
        if c.exists():
            return c
    return None


def summarize(csv_path: str | Path) -> dict:
    """Число строк/пикселей/пустых масок по (модуль, класс) и доля AF-чипов с пустой маской."""
    sub = read_submission(csv_path)
    stats: dict[tuple[str, str], dict[str, int]] = {}
    for cid, cls, rle in zip(sub["chip_id"], sub["class_id"], sub["rle"]):
        s = stats.setdefault((chip_kind(cid), cls), {"rows": 0, "pixels": 0, "empty": 0})
        s["rows"] += 1
        s["pixels"] += rle_pixel_count(rle)
        s["empty"] += not rle.strip()
    af = stats.get(("af", "1"), {"rows": 0, "empty": 0})
    return {"n_rows": len(sub["chip_id"]), "by_class": stats,
            "af_empty_frac": af["empty"] / af["rows"] if af["rows"] else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Валидация и сводка submission.csv")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test",
                    help="каталог теста: размеры чипов из meta.csv и шаблон по умолчанию")
    ap.add_argument("--sample", type=Path, default=None,
                    help="по умолчанию <data-dir>/sample_submission.csv, затем baseline/sample_submission.csv")
    args = ap.parse_args()

    sample = args.sample or find_sample(args.data_dir)
    if sample is None:
        print("sample_submission.csv не найден — укажите --sample")
        return 1
    errors = validate_submission(args.csv, sample, shapes_from_meta(args.data_dir))
    print(f"Файл:   {args.csv}\nШаблон: {sample}")
    if errors:
        print(f"ОШИБОК ФОРМАТА: {len(errors)}")
        for e in errors[:50]:
            print(" -", e)
        return 1

    s = summarize(args.csv)
    print(f"Формат: OK ({s['n_rows']} строк)\n")
    print(f"{'модуль':<7}{'class_id':>9}{'строк':>8}{'пикселей':>12}{'пустых масок':>14}")
    for (kind, cls), v in sorted(s["by_class"].items()):
        print(f"{kind.upper():<7}{cls:>9}{v['rows']:>8}{v['pixels']:>12}{v['empty']:>14}")
    print(f"\nДоля AF-чипов с пустой маской: {s['af_empty_frac']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
