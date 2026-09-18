"""Растр → вектор: маска AF → термоточки, маска BS → контуры гарей с площадью в UTM.

Чистые функции (без файлового ввода-вывода) — используются `scripts/build_service_data.py` и тестами.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shapely
from rasterio import features
from rasterio.errors import NotGeoreferencedWarning
from shapely.geometry import shape
from shapely.ops import unary_union

SEVERITY_LABELS = {0: "весь контур", 1: "слабая", 2: "средняя", 3: "сильная"}
M2_PER_HA = 10_000.0

HOTSPOT_COLUMNS = ["hotspot_id", "chip_id", "acq_datetime", "satellite", "i4_bt", "i5_bt", "daynight",
                   "landcover", "confidence", "source", "epsg", "x_utm", "y_utm", "gsd"]
SCAR_COLUMNS = ["contour_id", "chip_id", "fire_event_id", "severity_class", "severity_label", "area_ha",
                "date_pre", "date_post", "source", "epsg"]


def _iso_utc(value) -> str:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def hotspots_from_chip(chip: dict, mask: np.ndarray, source: str, confidence: np.ndarray | None = None):
    """Пиксели mask==1 AF-чипа → (records, xs, ys): атрибуты и центры пикселей в CRS чипа (UTM).

    chip — результат `read_af_chip` (нужны viirs, aux, transform, crs, meta).
    confidence — карта вероятностей модели (H, W) или None (для gt → 1.0, для правил → NaN).
    """
    rows, cols = np.nonzero(mask == 1)
    if rows.size == 0:
        return [], np.empty(0), np.empty(0)
    tr = chip["transform"]
    cc, rr = cols + 0.5, rows + 0.5  # центр пикселя
    xs, ys = tr.c + tr.a * cc + tr.b * rr, tr.f + tr.d * cc + tr.e * rr
    viirs, aux, meta = chip["viirs"], chip["aux"], chip["meta"]
    i4, i5, sza = viirs[3][rows, cols], viirs[4][rows, cols], viirs[5][rows, cols]
    lc = np.nan_to_num(aux[0][rows, cols], nan=0).astype(int)
    if confidence is not None:
        conf = confidence[rows, cols].astype(float)
    else:
        conf = np.full(rows.size, 1.0 if source == "gt" else np.nan)
    acq = _iso_utc(meta["acq_datetime"])
    epsg, cid = int(chip["crs"].to_epsg()), chip["chip_id"]
    recs = [{
        "hotspot_id": f"{cid}_{source}_{r:03d}_{c:03d}", "chip_id": cid, "acq_datetime": acq,
        "satellite": str(meta.get("satellite", "")), "i4_bt": round(float(a), 2), "i5_bt": round(float(b), 2),
        "daynight": "night" if z > 90 else "day", "landcover": int(l),
        "confidence": None if np.isnan(p) else round(float(p), 4), "source": source,
        "epsg": epsg, "x_utm": float(x), "y_utm": float(y), "gsd": float(abs(tr.a)),
    } for r, c, a, b, z, l, p, x, y in zip(rows, cols, i4, i5, sza, lc, conf, xs, ys)]
    return recs, np.asarray(xs, float), np.asarray(ys, float)


def despeckle_severity(mask: np.ndarray, min_pixels: int = 3) -> np.ndarray:
    """Убирает фрагменты < min_pixels, сохраняя общую площадь гари.

    Простой sieve по классам сливает мелочь на границе гари с фоном и теряет до 6 % площади чипа
    (проверено на train) — это нарушает сверку с `burn_area_ha`. Поэтому:
    1) изолированные пятна гари < min_pixels (в бинарной маске) удаляются;
    2) мелкие фрагменты классов ВНУТРИ гари вливаются в соседний класс гари, но не в фон
       (sieve с маской гари) — внешний контур гари не меняется.
    """
    sev = np.where(mask > 3, 0, mask).astype(np.uint8)
    if min_pixels <= 1:
        return sev
    burn = (sev > 0).astype(np.uint8)
    if not burn.any():
        return sev
    with warnings.catch_warnings():  # sieve работает с голым массивом — геопривязка ему не нужна
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        keep = (features.sieve(burn, min_pixels, connectivity=4) > 0) & (burn > 0)
        out = features.sieve(sev, min_pixels, connectivity=4, mask=burn)
    out[~keep] = 0
    return out


def _polygons(binary: np.ndarray, transform, simplify_m: float):
    geoms = [shape(g) for g, _ in features.shapes(binary.astype(np.uint8), mask=binary, connectivity=4,
                                                  transform=transform)]
    if simplify_m > 0:
        geoms = [g.simplify(simplify_m, preserve_topology=True) for g in geoms]
    return [g if g.is_valid else shapely.make_valid(g) for g in geoms]


def scars_from_mask(mask: np.ndarray, transform, epsg: int, chip_id: str, meta: dict, source: str,
                    min_pixels: int = 3, simplify_m: float = 5.0):
    """Маска степени (H, W) 0..3 (255 = nodata → фон) → (records, geoms_utm).

    Полигоны строятся по классам 1/2/3 в CRS чипа (UTM), площадь = geom.area / 10000 га.
    Дополнительно — объединённый контур всей гари (severity_class = 0).
    simplify(5 м) на «лесенке» 20-метровых пикселей убирает только коллинеарные вершины → площадь не меняется.
    """
    sev = despeckle_severity(mask, min_pixels)
    recs, geoms = [], []
    base = {"chip_id": chip_id, "fire_event_id": str(meta.get("fire_event_id", "")),
            "date_pre": str(meta.get("date_pre", "")), "date_post": str(meta.get("date_post", "")),
            "source": source, "epsg": int(epsg)}

    def add(geom, cls, n):
        recs.append({"contour_id": f"{chip_id}_{source}_{cls}_{n:04d}", **base, "severity_class": cls,
                     "severity_label": SEVERITY_LABELS[cls], "area_ha": round(geom.area / M2_PER_HA, 4)})
        geoms.append(geom)

    class_polys = []
    for cls in (1, 2, 3):
        polys = _polygons(sev == cls, transform, simplify_m)
        class_polys.extend(polys)
        for n, g in enumerate(polys, 1):
            add(g, cls, n)
    if class_polys:
        add(unary_union(class_polys), 0, 1)
    return recs, geoms
