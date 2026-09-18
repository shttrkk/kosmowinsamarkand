"""Точка входа обучения: python train.py --module af|bs|all

ЗАГЛУШКА (задача 01): обучаемых моделей пока нет, инференс работает на правилах
(src/firemon/models/*/rules.py). Агенты AF/BS-модулей подключают сюда свои пайплайны.
Обязательные требования к ним: seed_everything(42), фолды из configs/folds_{af,bs}.csv,
запись результата в reports/experiments.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение модулей AF/BS")
    ap.add_argument("--module", choices=["af", "bs", "all"], required=True)
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    print(f"[train] модуль '{args.module}': обучаемая модель ещё не подключена — используются правила-заглушки.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
