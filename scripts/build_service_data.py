"""Пересборка данных сервиса из обучающих чипов (геопривязка + даты + маски).

    python scripts/build_service_data.py                 # --source both: эталон + предсказания моделей
    python scripts/build_service_data.py --source gt     # только эталон (веса моделей не нужны)

Результат — `service/data/firemon.gpkg` (слои hotspots, burn_scars, chips) + fallback *.geojson.gz.
Слой pred строится финальными AFPredictor/BSPredictor, а пока их нет — заглушками на правилах.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from firemon.utils import get_logger, seed_everything  # noqa: E402
from service.build import build_layers, write_layers  # noqa: E402

log = get_logger("build_service_data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "service" / "data")
    ap.add_argument("--source", choices=("gt", "pred", "both"), default="both")
    ap.add_argument("--min-pixels", type=int, default=3, help="фрагменты гари меньше N пикселей убираются (1 = выкл.)")
    ap.add_argument("--simplify", type=float, default=5.0, help="допуск упрощения контуров, м (0 = выкл.)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-fallback", action="store_true", help="не писать *.geojson.gz")
    args = ap.parse_args()

    seed_everything(42, with_torch=False)
    t0 = time.perf_counter()
    sources = ("gt", "pred") if args.source == "both" else (args.source,)
    layers = build_layers(args.data_dir, sources, min_pixels=args.min_pixels, simplify_m=args.simplify,
                          threads=args.threads)
    write_layers(layers, args.out_dir, fallback=not args.no_fallback)

    scars = layers["burn_scars"]
    for src in sources:
        s = scars[(scars["source"] == src) & (scars["severity_class"] > 0)]
        log.info("%s: термоточек %d, контуров %d, площадь гарей %.1f га", src,
                 int((layers["hotspots"]["source"] == src).sum()), len(s), s["area_ha"].sum())
    if "gt" in sources:  # проверка корректности площадей: вектор (UTM) против burn_area_ha из meta
        got = scars[(scars["source"] == "gt") & (scars["severity_class"] > 0)].groupby("chip_id")["area_ha"].sum()
        ref = layers["chips"].set_index("chip_id")["burn_area_ha"].dropna()
        rel = ((got.reindex(ref.index).fillna(0) - ref).abs() / ref).dropna()
        log.info("сверка с meta.burn_area_ha: макс. отклонение %.3f %%, чипов с отклонением >1 %%: %d",
                 100 * rel.max(), int((rel > 0.01).sum()))
    log.info("готово за %.1f с", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
