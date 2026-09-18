"""Профиль инференса AF по этапам (чтение tif / кандидаты+признаки / LightGBM) на каталоге с чипами.
Запуск: python scripts/af/profile_inference.py --data-dir data/test_fake"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor


from firemon.io import list_chips, read_af_chip
from firemon.models.af.predictor import AFPredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/test_fake")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    ids = list_chips(args.data_dir, "af")
    t = time.perf_counter()
    pred = AFPredictor({"num_threads": args.threads}).load()
    t_load = time.perf_counter() - t
    t = time.perf_counter()
    chips = [read_af_chip(args.data_dir, c, with_mask=False) for c in ids]
    t_read = time.perf_counter() - t
    t = time.perf_counter()
    parts = [pred.candidates_and_rows(c) for c in chips]
    t_feat = time.perf_counter() - t
    t = time.perf_counter()
    pred.masks_from_parts(parts)
    t_pred = time.perf_counter() - t
    n_rows = sum(len(r) for _, r in parts)
    print(f"{len(ids)} чипов, {n_rows} кандидатов, чипов без кандидатов: {sum(len(r) == 0 for _, r in parts)}")
    print(f"последовательно: загрузка весов {t_load:.2f} с | чтение {t_read:.2f} с ({1000 * t_read / len(ids):.1f} мс/чип) | "
          f"признаки {t_feat:.2f} с ({1000 * t_feat / len(ids):.1f} мс/чип) | LightGBM {t_pred:.2f} с")
    t = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as ex:
        parts = list(ex.map(lambda c: pred.candidates_and_rows(read_af_chip(args.data_dir, c, with_mask=False)), ids))
    print(f"чтение+признаки в {args.threads} потоках: {time.perf_counter() - t:.2f} с")


if __name__ == "__main__":
    main()
