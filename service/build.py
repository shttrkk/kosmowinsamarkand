"""Сборка данных сервиса из чипов с геопривязкой: термоточки, контуры гарей, индекс сцен.

Результат: `<out>/firemon.gpkg` (слои hotspots, burn_scars, chips; EPSG:4326) и снимок для git
`<out>/{hotspots,burn_scars,chips}.geojson.gz` (только source=gt) — сервис поднимается из него сразу после
клонирования, без data/train; полный GeoPackage (gt + pred) в git не хранится (сотни МБ).
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import box

from firemon.io import list_chips, read_af_chip, read_bs_chip
from firemon.utils import get_logger
from service.vectorize import HOTSPOT_COLUMNS, SCAR_COLUMNS, hotspots_from_chip, scars_from_mask

log = get_logger("service.build")

GPKG_NAME = "firemon.gpkg"
LAYERS = ("hotspots", "burn_scars", "chips")
WGS84 = "EPSG:4326"
STUB_MIN_PIXELS = 25
COORD_GRID = 1e-7  # ° ≈ 1 см: компактный GeoJSON без заметной потери площади


def to_wgs84(geoms, epsgs) -> np.ndarray:
    """Геометрии в UTM (у каждой свой epsg) → EPSG:4326, координаты на сетке 1e-7°."""
    geoms = np.asarray(geoms, dtype=object)
    epsgs = np.asarray(epsgs)
    out = np.empty(len(geoms), dtype=object)
    for epsg in np.unique(epsgs):
        tr = Transformer.from_crs(int(epsg), 4326, always_xy=True)
        sel = epsgs == epsg
        out[sel] = shapely.transform(geoms[sel], lambda xy, tr=tr: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    out = shapely.set_precision(out, COORD_GRID)
    return out


def _af_chip(data_dir, cid, sources, predictor):
    chip = read_af_chip(data_dir, cid, with_mask=True)
    recs, xs, ys = [], [], []
    if "gt" in sources and chip["mask"] is not None:
        r, x, y = hotspots_from_chip(chip, chip["mask"], "gt")
        recs += r; xs.append(x); ys.append(y)
    if "pred" in sources:
        mask, proba = predictor(chip)
        r, x, y = hotspots_from_chip(chip, mask, "pred", proba)
        recs += r; xs.append(x); ys.append(y)
    return chip, recs, np.concatenate(xs) if xs else np.empty(0), np.concatenate(ys) if ys else np.empty(0)


def _chip_record(chip, kind, date_from, date_to):
    m = chip["meta"]
    return {"chip_id": chip["chip_id"], "kind": kind, "epsg": int(chip["crs"].to_epsg()),
            "date_from": str(date_from)[:10], "date_to": str(date_to)[:10],
            "fire_event_id": "" if kind == "af" else str(m.get("fire_event_id", "")),
            "burn_area_ha": None if kind == "af" else float(m.get("burn_area_ha", np.nan))}, \
        box(m["x_min"], m["y_min"], m["x_max"], m["y_max"])


def build_layers(data_dir: Path, sources=("gt", "pred"), af_chips=None, bs_chips=None,
                 min_pixels: int = 3, simplify_m: float = 5.0, threads: int = 4) -> dict[str, gpd.GeoDataFrame]:
    """Читает чипы и возвращает {"hotspots", "burn_scars", "chips"} — GeoDataFrame в EPSG:4326."""
    data_dir = Path(data_dir)
    af_pred = bs_pred = None
    models = {}
    if "pred" in sources:
        from service.predictors import get_af_predictor, get_bs_predictor

        af_pred, models["af"] = get_af_predictor()
        bs_pred, models["bs"] = get_bs_predictor()
    af_ids = list_chips(data_dir, "af") if af_chips is None else list(af_chips)
    bs_ids = list_chips(data_dir, "bs") if bs_chips is None else list(bs_chips)

    h_recs, h_x, h_y, c_recs, c_geoms = [], [], [], [], []
    # чтение упирается в распаковку deflate; rasterio отпускает GIL → потоки
    with ThreadPoolExecutor(threads) as ex:
        for chip, recs, xs, ys in ex.map(lambda c: _af_chip(data_dir, c, sources, af_pred), af_ids):
            h_recs += recs; h_x.append(xs); h_y.append(ys)
            acq = chip["meta"]["acq_datetime"]
            rec, geom = _chip_record(chip, "af", acq, acq)
            c_recs.append(rec); c_geoms.append(geom)
    log.info("AF: %d чипов → %d термоточек", len(af_ids), len(h_recs))

    s_recs, s_geoms = [], []

    def bs_one(cid):
        layers = None if "pred" in sources else ("aux",)  # для gt хватает геопривязки одного слоя
        chip = read_bs_chip(data_dir, cid, with_mask=True, layers=layers)
        out = []
        masks = []
        if "gt" in sources and chip["mask"] is not None:
            masks.append(("gt", chip["mask"]))
        if "pred" in sources:
            masks.append(("pred", bs_pred(chip)))
        for src, mask in masks:
            # заглушка на порогах dNBR даёт «соль и перец» — для неё фильтр грубее, иначе сотни тысяч полигонов
            mp = max(min_pixels, STUB_MIN_PIXELS) if src == "pred" and models.get("bs") == "rules-stub" else min_pixels
            out.append(scars_from_mask(mask, chip["transform"], chip["crs"].to_epsg(), cid, chip["meta"], src,
                                       min_pixels=mp, simplify_m=simplify_m))
        for key in ("s2_pre", "s2_post", "s1_pre", "s1_post", "aux"):
            chip.pop(key, None)
        return chip, out

    with ThreadPoolExecutor(threads) as ex:
        for chip, out in ex.map(bs_one, bs_ids):
            for recs, geoms in out:
                s_recs += recs; s_geoms += geoms
            rec, geom = _chip_record(chip, "bs", chip["meta"]["date_pre"], chip["meta"]["date_post"])
            c_recs.append(rec); c_geoms.append(geom)
    log.info("BS: %d чипов → %d контуров", len(bs_ids), len(s_recs))

    hdf = pd.DataFrame(h_recs, columns=HOTSPOT_COLUMNS)
    pts = shapely.points(np.concatenate(h_x) if h_x else [], np.concatenate(h_y) if h_y else [])
    hotspots = gpd.GeoDataFrame(hdf, geometry=to_wgs84(pts, hdf["epsg"].to_numpy()), crs=WGS84)
    hotspots["confidence"] = hotspots["confidence"].astype(float)

    sdf = pd.DataFrame(s_recs, columns=SCAR_COLUMNS)
    scars = gpd.GeoDataFrame(sdf, geometry=to_wgs84(s_geoms, sdf["epsg"].to_numpy()), crs=WGS84)

    cdf = pd.DataFrame(c_recs)
    chips = gpd.GeoDataFrame(cdf, geometry=to_wgs84(shapely.segmentize(np.asarray(c_geoms, dtype=object), 5000.0),
                                                   cdf["epsg"].to_numpy()), crs=WGS84)
    for gdf in (hotspots, scars, chips):
        gdf.attrs["models"] = models
    return {"hotspots": hotspots, "burn_scars": scars, "chips": chips}


def write_layers(layers: dict[str, gpd.GeoDataFrame], out_dir: Path, fallback: bool = True) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / GPKG_NAME
    if gpkg.exists():
        gpkg.unlink()
    for name, gdf in layers.items():
        gdf.to_file(gpkg, layer=name, driver="GPKG")
        if fallback:
            # снимок для git: только эталон (pred от заглушек/моделей тяжёлый и меняется); mtime=0 → байт-в-байт
            snap = gdf[gdf["source"] == "gt"] if "source" in gdf.columns else gdf
            with open(out_dir / f"{name}.geojson.gz", "wb") as raw,                     gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0, filename="") as f:
                f.write(snap.to_json(drop_id=True).encode("utf-8"))
    scars = layers["burn_scars"]
    manifest = {"built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pred_models": layers["hotspots"].attrs.get("models", {}),
                "counts": {src: {"hotspots": int((layers["hotspots"]["source"] == src).sum()),
                                 "burn_scars": int(((scars["source"] == src) & (scars["severity_class"] > 0)).sum())}
                           for src in ("gt", "pred")},
                "n_chips": {k: int((layers["chips"]["kind"] == k).sum()) for k in ("af", "bs")}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("записано: %s (%.1f МБ)%s", gpkg, gpkg.stat().st_size / 1e6, " + *.geojson.gz" if fallback else "")


def read_layers(data_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    """GeoPackage, иначе fallback *.geojson.gz; если нет ничего — FileNotFoundError."""
    data_dir = Path(data_dir)
    gpkg = data_dir / GPKG_NAME
    out = {}
    for name in LAYERS:
        if gpkg.exists():
            out[name] = gpd.read_file(gpkg, layer=name)
            continue
        fb = data_dir / f"{name}.geojson.gz"
        if not fb.exists():
            raise FileNotFoundError(f"нет данных сервиса в {data_dir}: запустите python scripts/build_service_data.py")
        with gzip.open(fb, "rt", encoding="utf-8") as f:
            fc = json.load(f)
        out[name] = gpd.GeoDataFrame.from_features(fc["features"], crs=WGS84)
    return out
