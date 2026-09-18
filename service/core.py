"""Ядро сервиса: пространственно-временной отбор, clip контуров, площади в UTM, справка, экспорт.

Чистые функции без HTTP. Все геометрии запроса/ответа — EPSG:4326; площади считаются в UTM-зоне
исходного чипа (32637/32638): для контуров, целиком лежащих в запросе, берётся площадь, посчитанная
при сборке; для пересекающих границу — геометрия обрезается в UTM и площадь пересчитывается.
"""
from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Geod, Transformer
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from service.vectorize import M2_PER_HA, SEVERITY_LABELS

WGS84 = "EPSG:4326"
CRS_NOTE = "площади рассчитаны в UTM 37N/38N (EPSG:32637/32638), пиксель 20 м = 0.04 га"
SOURCES = ("gt", "pred")
_GEOD = Geod(ellps="WGS84")

HOTSPOT_PROPS = ["hotspot_id", "chip_id", "acq_datetime", "satellite", "i4_bt", "i5_bt", "daynight",
                 "landcover", "confidence", "source"]
SCAR_PROPS = ["contour_id", "chip_id", "fire_event_id", "severity_class", "severity_label", "area_ha",
              "date_pre", "date_post", "source"]
# имена полей Shapefile (dbf: не длиннее 10 символов)
SHP_SCAR_FIELDS = {"contour_id": "cid", "chip_id": "chip_id", "fire_event_id": "event_id",
                   "severity_class": "sev_cls", "severity_label": "sev_lbl", "area_ha": "area_ha",
                   "date_pre": "date_pre", "date_post": "date_post", "source": "source"}
SHP_HOTSPOT_FIELDS = {"hotspot_id": "hid", "chip_id": "chip_id", "acq_datetime": "acq_dt", "satellite": "satellite",
                      "i4_bt": "i4_bt", "i5_bt": "i5_bt", "daynight": "daynight", "landcover": "landcover",
                      "confidence": "confidence", "source": "source"}


class QueryError(ValueError):
    """Некорректные параметры запроса (→ HTTP 422)."""


# ───────────────────────── разбор параметров ─────────────────────────

def parse_bbox(bbox) -> BaseGeometry:
    if isinstance(bbox, str):
        try:
            bbox = [float(v) for v in bbox.split(",")]
        except ValueError:
            raise QueryError("bbox: ожидаются 4 числа через запятую: minx,miny,maxx,maxy") from None
    if len(bbox) != 4 or not all(np.isfinite(bbox)):
        raise QueryError("bbox: ожидаются 4 конечных числа: minx,miny,maxx,maxy (EPSG:4326)")
    minx, miny, maxx, maxy = map(float, bbox)
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise QueryError("bbox: нужно minx<maxx, miny<maxy, долгота в [-180,180], широта в [-90,90]")
    return box(minx, miny, maxx, maxy)


def parse_geometry(geometry: dict) -> BaseGeometry:
    """GeoJSON Polygon/MultiPolygon (или Feature с такой геометрией) → shapely."""
    try:
        if geometry.get("type") == "Feature":
            geometry = geometry["geometry"]
        geom = shape(geometry)
    except Exception as e:
        raise QueryError(f"geometry: не удалось разобрать GeoJSON ({e})") from None
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise QueryError(f"geometry: нужен Polygon или MultiPolygon, получен {geom.geom_type}")
    if geom.is_empty:
        raise QueryError("geometry: пустая геометрия")
    if not geom.is_valid:
        raise QueryError(f"geometry: невалидная геометрия — {explain_validity(geom)}")
    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90):
        raise QueryError("geometry: координаты должны быть в EPSG:4326 (долгота, широта)")
    return geom


def parse_date(value, name: str) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise QueryError(f"{name}: ожидается дата в формате YYYY-MM-DD, получено «{value}»") from None


def check_period(d0: date | None, d1: date | None) -> None:
    if d0 and d1 and d0 > d1:
        raise QueryError(f"date_from ({d0}) позже date_to ({d1})")


def check_source(source: str) -> str:
    if source not in SOURCES:
        raise QueryError(f"source: допустимо {'|'.join(SOURCES)}, получено «{source}»")
    return source


# ───────────────────────── хранилище ─────────────────────────

def _features_json(gdf: gpd.GeoDataFrame, props: list[str]) -> np.ndarray:
    """Готовые строки GeoJSON Feature для каждой строки слоя (считаются один раз при старте)."""
    geoms = shapely.to_geojson(gdf.geometry.values)
    records = gdf[props].astype(object).where(gdf[props].notna(), None).to_dict("records")
    return np.array([f'{{"type":"Feature","properties":{json.dumps(p, ensure_ascii=False)},"geometry":{g}}}'
                     for p, g in zip(records, geoms)], dtype=object)


def _days(values) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[D]")


@dataclass
class Store:
    hotspots: gpd.GeoDataFrame
    scars: gpd.GeoDataFrame
    chips: gpd.GeoDataFrame
    aoi: dict | None = None
    manifest: dict = field(default_factory=dict)

    def __post_init__(self):
        h, s = self.hotspots, self.scars
        self.h_geom = np.asarray(h.geometry.values, dtype=object)
        self.h_tree = shapely.STRtree(self.h_geom)
        self.h_day = _days([str(v)[:10] for v in h["acq_datetime"]]) if len(h) else _days([])
        self.h_source = h["source"].to_numpy(dtype=object)
        self.h_json = _features_json(h, HOTSPOT_PROPS)

        self.s_geom = np.asarray(s.geometry.values, dtype=object)
        self.s_tree = shapely.STRtree(self.s_geom)
        self.s_pre = _days(s["date_pre"].astype(str).to_list()) if len(s) else _days([])
        self.s_post = _days(s["date_post"].astype(str).to_list()) if len(s) else _days([])
        self.s_source = s["source"].to_numpy(dtype=object)
        self.s_class = s["severity_class"].to_numpy(dtype=int)
        self.s_area = s["area_ha"].to_numpy(dtype=float)
        self.s_epsg = s["epsg"].to_numpy(dtype=int)
        self.s_json = _features_json(s, SCAR_PROPS)
        # «истинная» геометрия для clip и площадей — в UTM-зоне чипа
        self.s_utm = np.empty(len(s), dtype=object)
        for epsg in np.unique(self.s_epsg):
            sel = self.s_epsg == epsg
            self.s_utm[sel] = _transform(self.s_geom[sel], 4326, int(epsg))

    @classmethod
    def empty(cls) -> "Store":
        def e(cols):
            return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs=WGS84)
        return cls(e(HOTSPOT_PROPS + ["epsg"]), e(SCAR_PROPS + ["epsg"]), e(["chip_id", "kind"]))

    @classmethod
    def load(cls, data_dir: Path) -> "Store":
        from service.build import read_layers

        data_dir = Path(data_dir)
        layers = read_layers(data_dir)
        aoi = manifest = None
        aoi_path = data_dir / "aoi" / "fire_monitoring_aoi.geojson"
        if aoi_path.exists():
            aoi = json.loads(aoi_path.read_text(encoding="utf-8"))
        if (data_dir / "manifest.json").exists():
            manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        return cls(layers["hotspots"], layers["burn_scars"], layers["chips"], aoi, manifest or {})

    def date_ranges(self) -> dict:
        out = {}
        for src in SOURCES:
            h, s = self.h_source == src, (self.s_source == src) & (self.s_class > 0)
            out[src] = {
                "hotspots": {"n": int(h.sum()), "date_min": str(self.h_day[h].min()) if h.any() else None,
                             "date_max": str(self.h_day[h].max()) if h.any() else None},
                "burn_scars": {"n": int(s.sum()), "date_min": str(self.s_pre[s].min()) if s.any() else None,
                               "date_max": str(self.s_post[s].max()) if s.any() else None},
            }
        return out


def _transform(geoms, src: int, dst: int):
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return shapely.transform(geoms, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))


# ───────────────────────── запросы ─────────────────────────

@dataclass
class ScarResult:
    idx: np.ndarray          # индексы строк слоя burn_scars
    area_ha: np.ndarray      # площадь внутри запроса (UTM), га
    clipped: dict            # idx → обрезанная геометрия EPSG:4326 (только для пересекающих границу запроса)


def query_hotspots(store: Store, geom: BaseGeometry, d0: date | None, d1: date | None, source: str) -> np.ndarray:
    """Индексы термоточек внутри geom с датой пролёта в [d0, d1] (включительно, UTC)."""
    idx = np.sort(store.h_tree.query(geom, predicate="intersects"))
    keep = store.h_source[idx] == source
    if d0:
        keep &= store.h_day[idx] >= np.datetime64(d0)
    if d1:
        keep &= store.h_day[idx] <= np.datetime64(d1)
    return idx[keep]


def _polygonal(geoms: np.ndarray) -> np.ndarray:
    """Из результата intersection оставляет только полигональную часть (линии/точки касания выбрасываются)."""
    out = geoms.copy()
    for i, g in enumerate(geoms):
        if g is not None and g.geom_type == "GeometryCollection":
            out[i] = shapely.union_all([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])
    types = shapely.get_type_id(out)
    out[~np.isin(types, (3, 6))] = None  # 3 Polygon, 6 MultiPolygon
    return out


def query_scars(store: Store, geom: BaseGeometry, d0: date | None, d1: date | None, source: str,
                with_union: bool = False) -> ScarResult:
    """Контуры гарей, у которых интервал [date_pre, date_post] пересекается с [d0, d1]; clip по geom.

    Пожар произошёл между снимками pre и post, поэтому гарь относится к периоду, если интервалы пересекаются.
    """
    shapely.prepare(geom)
    cand = np.sort(store.s_tree.query(geom, predicate="intersects"))
    keep = store.s_source[cand] == source
    if not with_union:
        keep &= store.s_class[cand] > 0
    if d0:
        keep &= store.s_post[cand] >= np.datetime64(d0)
    if d1:
        keep &= store.s_pre[cand] <= np.datetime64(d1)
    cand = cand[keep]
    inside = shapely.contains_properly(geom, store.s_geom[cand]) if len(cand) else np.zeros(0, bool)
    area = store.s_area[cand].copy()
    clipped: dict[int, BaseGeometry] = {}
    part = cand[~inside]
    if len(part):
        new_area = np.zeros(len(part))
        for epsg in np.unique(store.s_epsg[part]):
            sel = store.s_epsg[part] == epsg
            ids = part[sel]
            # запрос режем до окрестности кандидатов (UTM-проекция далеко от зоны искажает), уплотняем вершины:
            # прямая в градусах — не прямая в UTM
            local = geom.intersection(box(*shapely.total_bounds(store.s_geom[ids])).buffer(0.05))
            q_utm = _transform(shapely.segmentize(local, 0.005), 4326, int(epsg))
            pieces = _polygonal(shapely.intersection(store.s_utm[ids], q_utm))
            a = np.nan_to_num(shapely.area(pieces), nan=0.0) / M2_PER_HA
            new_area[sel] = a
            ok = a > 0
            if ok.any():
                back = _transform(pieces[ok], int(epsg), 4326)
                clipped.update(zip(ids[ok].tolist(), back))
        area[~inside] = new_area
    nonzero = area > 0
    return ScarResult(cand[nonzero], np.round(area[nonzero], 4), clipped)


def geodesic_area_km2(geom: BaseGeometry) -> float:
    return abs(_GEOD.geometry_area_perimeter(geom)[0]) / 1e6


def summarize(store: Store, scars: ScarResult, hotspot_idx: np.ndarray, geom: BaseGeometry,
              d0: date | None, d1: date | None, source: str) -> dict:
    cls = store.s_class[scars.idx]
    by = {k: float(scars.area_ha[cls == k].sum()) for k in (1, 2, 3)}
    total = sum(by.values())
    dn = store.hotspots["daynight"].to_numpy(dtype=object)[hotspot_idx] if len(hotspot_idx) else np.array([])
    return {
        "area_total_ha": round(total, 2),
        "by_severity": {str(k): {"area_ha": round(v, 2), "share_pct": round(100 * v / total, 2) if total else 0.0,
                                 "label": SEVERITY_LABELS[k]} for k, v in by.items()},
        "n_contours": int((cls > 0).sum()),
        "n_fire_events": int(store.scars["fire_event_id"].iloc[scars.idx].nunique()) if len(scars.idx) else 0,
        "n_hotspots": int(len(hotspot_idx)),
        "n_hotspots_day": int((dn == "day").sum()), "n_hotspots_night": int((dn == "night").sum()),
        "period": {"date_from": d0.isoformat() if d0 else None, "date_to": d1.isoformat() if d1 else None},
        "source": source,
        "aoi_area_km2": round(geodesic_area_km2(geom), 2),
        "crs_note": CRS_NOTE,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ───────────────────────── сериализация ─────────────────────────

def _fc(features: list[str]) -> str:
    return '{"type":"FeatureCollection","features":[' + ",".join(features) + "]}"


def hotspots_geojson(store: Store, idx: np.ndarray, as_pixels: bool = False) -> str:
    if not as_pixels or not len(idx):
        return _fc(store.h_json[idx].tolist())
    gdf = hotspot_pixels(store, idx)
    return _fc(_features_json(gdf, HOTSPOT_PROPS).tolist())


def hotspot_pixels(store: Store, idx: np.ndarray) -> gpd.GeoDataFrame:
    """Термоточки как полигоны пикселей VIIRS (375×375 м в UTM-зоне чипа) → EPSG:4326."""
    h = store.hotspots.iloc[idx]
    half = h["gsd"].to_numpy(float) / 2
    x, y, epsg = h["x_utm"].to_numpy(float), h["y_utm"].to_numpy(float), h["epsg"].to_numpy(int)
    squares = shapely.box(x - half, y - half, x + half, y + half)
    out = np.empty(len(h), dtype=object)
    for e in np.unique(epsg):
        out[epsg == e] = _transform(squares[epsg == e], int(e), 4326)
    return gpd.GeoDataFrame(h[HOTSPOT_PROPS].reset_index(drop=True), geometry=shapely.set_precision(out, 1e-7), crs=WGS84)


def scars_gdf(store: Store, res: ScarResult) -> gpd.GeoDataFrame:
    gdf = store.scars.iloc[res.idx][SCAR_PROPS + ["geometry"]].reset_index(drop=True)
    gdf["area_ha"] = res.area_ha
    if res.clipped:
        geoms = np.asarray(gdf.geometry.values, dtype=object)
        for pos, i in enumerate(res.idx.tolist()):
            if i in res.clipped:
                geoms[pos] = res.clipped[i]
        gdf = gdf.set_geometry(shapely.set_precision(geoms, 1e-7), crs=WGS84)
    return gdf


def scars_geojson(store: Store, res: ScarResult) -> str:
    if not res.clipped:
        return _fc(store.s_json[res.idx].tolist())
    feats = store.s_json[res.idx].copy()
    pos = np.array([p for p, i in enumerate(res.idx.tolist()) if i in res.clipped], dtype=int)
    sub = ScarResult(res.idx[pos], res.area_ha[pos], res.clipped)
    feats[pos] = _features_json(scars_gdf(store, sub), SCAR_PROPS)
    return _fc(feats.tolist())


def to_shapefile_zip(gdf: gpd.GeoDataFrame, fields: dict[str, str], name: str) -> bytes:
    """GeoDataFrame → zip с shp+shx+dbf+prj+cpg (UTF-8, EPSG:4326)."""
    out = gdf[list(fields) + ["geometry"]].rename(columns=fields)
    if len(out) == 0:  # пустой слой: тип геометрии задаём явно, иначе драйвер не создаст .shp
        out = gpd.GeoDataFrame({c: [] for c in fields.values()}, geometry=[], crs=WGS84)
    with tempfile.TemporaryDirectory() as tmp:
        kind = "Point" if name == "hotspots" else "MultiPolygon"
        out.to_file(Path(tmp) / f"{name}.shp", driver="ESRI Shapefile", encoding="utf-8", geometry_type=kind)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(Path(tmp).iterdir()):
                z.write(f, f.name)
    return buf.getvalue()


def summary_csv(summary: dict) -> str:
    """Справка таблицей: показатель, класс, подпись, значение, единица. BOM — чтобы Excel понял UTF-8."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["indicator", "severity_class", "label", "value", "unit"])
    for k, v in summary["by_severity"].items():
        w.writerow(["area", k, v["label"], v["area_ha"], "ha"])
        w.writerow(["share", k, v["label"], v["share_pct"], "%"])
    w.writerow(["area_total", "", "суммарная площадь гари", summary["area_total_ha"], "ha"])
    w.writerow(["n_contours", "", "число контуров", summary["n_contours"], "шт"])
    w.writerow(["n_fire_events", "", "число пожаров (fire_event_id)", summary["n_fire_events"], "шт"])
    w.writerow(["n_hotspots", "", "число термоточек", summary["n_hotspots"], "шт"])
    w.writerow(["n_hotspots_day", "", "термоточки, день", summary["n_hotspots_day"], "шт"])
    w.writerow(["n_hotspots_night", "", "термоточки, ночь", summary["n_hotspots_night"], "шт"])
    w.writerow(["aoi_area", "", "площадь области запроса", summary["aoi_area_km2"], "km2"])
    w.writerow(["date_from", "", "начало периода", summary["period"]["date_from"] or "", ""])
    w.writerow(["date_to", "", "конец периода", summary["period"]["date_to"] or "", ""])
    w.writerow(["source", "", "источник масок", summary["source"], ""])
    w.writerow(["crs_note", "", summary["crs_note"], "", ""])
    w.writerow(["generated_at", "", "время формирования (UTC)", summary["generated_at"], ""])
    return "﻿" + buf.getvalue()
