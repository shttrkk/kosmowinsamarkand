"""Точка входа обучения: python train.py --module af|bs|all

AF: признаки → кэш data/cache/af_train.npz → 5 фолдов LightGBM → OOF/временной сплит → weights/af_lgbm_*.
BS: правила → LightGBM по пикселям → 2×UNet (CV + full) → стек-LGBM → декодирование/постобработка → weights/bs_*
    (src/firemon/models/bs/pipeline.py; этапы выбираются через --bs-stages).
Обязательные требования: seed_everything(42), фолды из configs/folds_{af,bs}.csv,
запись результата в reports/experiments.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение модулей AF/BS")
    ap.add_argument("--module", choices=["af", "bs", "all"], required=True)
    ap.add_argument("--data-dir", default=None, help="по умолчанию — data_dir из конфига (data/train)")
    ap.add_argument("--config", default=None, help="yaml-конфиг модуля (для af по умолчанию configs/af_lgbm.yaml)")
    ap.add_argument("--no-log", action="store_true", help="не дописывать строку в reports/experiments.md")
    ap.add_argument("--rebuild-cache", action="store_true", help="пересчитать кэш признаков")
    ap.add_argument("--bs-stages", nargs="*", default=None,
                    help="этапы BS: rules lgbm unet stack ensemble (по умолчанию все)")
    ap.add_argument("--weights-dir", default="weights",
                    help="BS: куда писать веса (для другого каталога кэш и итоги тоже отдельные — проверка воспроизводимости)")
    ap.add_argument("--bs-unet-loss", default=None, choices=["ce_dice", "focal_dice", "lovasz", "ce_lovasz"],
                    help="BS: лосс UNet (по умолчанию ce_dice); для экспериментов — вместе с --weights-dir")
    args = ap.parse_args()
    if args.module in ("af", "all"):
        from firemon.models.af.pipeline import train_af
        from firemon.utils import load_config

        cfg = load_config(args.config or ROOT / "configs" / "af_lgbm.yaml")
        train_af(cfg, args.data_dir, log_experiment=not args.no_log, rebuild_cache=args.rebuild_cache)
    if args.module in ("bs", "all"):
        from firemon.models.bs.pipeline import run as run_bs

        run_bs(args.data_dir or str(ROOT / "data" / "train"), args.bs_stages, args.weights_dir, args.bs_unet_loss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
