"""Сравнение способов распараллелить самый дорогой этап инференса — признаки AF (для reports/speed.md).

    python scripts/bench_inference_parallel.py --data-dir data/test

Потоки (как в inference.py) против процессов (spawn на Windows / fork на Linux) на полном наборе AF-чипов.
Время процессов включает старт пула — именно так он бы оплачивался в inference.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_PREDICTOR = None


def _rows(args: tuple[str, str]) -> int:
    """Чтение чипа + признаки кандидатов; возвращает число строк (сами строки гонять между процессами дорого)."""
    global _PREDICTOR
    from firemon.io import read_af_chip
    from firemon.models.af.predictor import AFPredictor

    if _PREDICTOR is None:
        _PREDICTOR = AFPredictor().load_meta()
    data_dir, cid = args
    return len(_PREDICTOR.candidates_and_rows(read_af_chip(data_dir, cid, with_mask=False))[1])


def main() -> int:
    from firemon.io import list_chips

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    args = ap.parse_args()
    jobs = [(str(args.data_dir), c) for c in list_chips(args.data_dir, "af")]
    print(f"AF-чипов: {len(jobs)}\n{'способ':<10}{'workers':>8}{'с':>8}{'строк':>9}")
    for name, pool_cls in (("потоки", ThreadPoolExecutor), ("процессы", ProcessPoolExecutor)):
        for w in args.workers:
            t0 = time.perf_counter()
            with pool_cls(w) as pool:
                n = sum(pool.map(_rows, jobs, **({"chunksize": 4} if pool_cls is ProcessPoolExecutor else {})))
            print(f"{name:<10}{w:>8}{time.perf_counter() - t0:>8.2f}{n:>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
