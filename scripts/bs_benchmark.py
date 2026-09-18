"""Замер времени BS-инференса на 89 чипах (размер тестовой выборки) для GPU и CPU.

    python scripts/bs_benchmark.py --device cuda
    python scripts/bs_benchmark.py --device cpu [--tta-cpu]

Берутся первые N чипов train (чтение файлов входит в замер, как в inference.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--n", type=int, default=89)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tta-cpu", action="store_true")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    t_start = time.perf_counter()
    from firemon.io import list_chips
    from firemon.models.bs.infer import load_bs_config, run_bs_inference

    cfg = load_bs_config({"device": args.device, "tta_cpu": args.tta_cpu})
    if args.no_stack:
        cfg["use_stack"] = False
    if args.threads:
        cfg["num_threads"] = args.threads
    ids = list_chips(args.data_dir, "bs")[: args.n]
    preds, failed, timing = run_bs_inference(args.data_dir, ids, cfg)
    total = time.perf_counter() - t_start
    print(json.dumps({"device": args.device, "n": len(preds), "failed": failed, "load_s": round(timing["load"], 2),
                      "predict_s": round(timing["predict"], 2), "total_with_imports_s": round(total, 2)}))


if __name__ == "__main__":
    main()
