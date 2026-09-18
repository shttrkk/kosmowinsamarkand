"""Сравнение двух submission.csv: побайтное равенство, доля совпавших пикселей и IoU контуров по BS-чипам.

    python scripts/compare_submissions.py out/a.csv out/b.csv
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.rle import rle_decode  # noqa: E402
from firemon.submission import read_submission  # noqa: E402


def bs_masks(path: str) -> dict[str, np.ndarray]:
    sub = read_submission(path)
    out: dict[str, np.ndarray] = {}
    for c, k, r in zip(sub["chip_id"], sub["class_id"], sub["rle"]):
        if c.upper().startswith("BS"):
            out.setdefault(c, np.zeros((512, 512), np.uint8))[rle_decode(r, (512, 512)) == 1] = int(k)
    return out


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    ha, hb = (hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (a, b))
    print("побайтно равны:", ha == hb)
    sa, sb = read_submission(a), read_submission(b)
    rows_diff = sum(x != y for x, y in zip(sa["rle"], sb["rle"]))
    af_diff = sum(x != y for c, x, y in zip(sa["chip_id"], sa["rle"], sb["rle"]) if c.upper().startswith("AF"))
    print(f"строк с разным RLE: {rows_diff} из {len(sa['rle'])} (из них AF: {af_diff})")
    ma, mb = bs_masks(a), bs_masks(b)
    n = len(ma) * 512 * 512
    same = sum(int((ma[k] == mb[k]).sum()) for k in ma)
    inter = sum(int(((ma[k] > 0) & (mb[k] > 0)).sum()) for k in ma)
    union = sum(int(((ma[k] > 0) | (mb[k] > 0)).sum()) for k in ma)
    chips_diff = sum(bool((ma[k] != mb[k]).any()) for k in ma)
    print(f"BS: совпало пикселей {same / n:.6f} ({n - same} различаются), IoU контуров {inter / max(union, 1):.6f}, "
          f"чипов с отличиями {chips_diff} из {len(ma)}")


if __name__ == "__main__":
    main()
