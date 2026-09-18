"""REST API + веб-карта сервиса мониторинга природных пожаров.

Запуск из корня репозитория:  python -m uvicorn service.app:app --host 0.0.0.0 --port 8000
Данные: `service/data/` (или каталог из переменной окружения FIREMON_SERVICE_DATA);
пересборка — `python scripts/build_service_data.py`. Документация API — /docs.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):  # работает и без `pip install -e .`
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, Query, Request  # noqa: E402
from fastapi.openapi.docs import get_swagger_ui_html  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from firemon.utils import get_logger  # noqa: E402
from service import core  # noqa: E402

log = get_logger("service")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
GEOJSON = "application/geo+json"
MAX_SAVED_QUERIES = 256

Layer = Literal["hotspots", "burn_scars", "summary"]
Source = Literal["gt", "pred"]


class QueryBody(BaseModel):
    """Область (GeoJSON-полигон ИЛИ bbox, EPSG:4326) + интервал дат."""
    geometry: dict | None = Field(None, description="GeoJSON Polygon | MultiPolygon (EPSG:4326)")
    bbox: list[float] | None = Field(None, description="[minx, miny, maxx, maxy], EPSG:4326")
    date_from: str | None = Field(None, description="YYYY-MM-DD, включительно; пусто — с начала архива")
    date_to: str | None = Field(None, description="YYYY-MM-DD, включительно; пусто — до конца архива")
    source: Source = Field("gt", description="gt — эталонная разметка, pred — предсказания моделей")
    layers: list[Layer] = Field(default_factory=lambda: ["hotspots", "burn_scars", "summary"])
    hotspots_as_pixels: bool = Field(False, description="термоточки полигонами пикселей 375×375 м, а не точками")
    with_union: bool = Field(False, description="добавить объединённые контуры гарей (severity_class = 0)")

    model_config = {"json_schema_extra": {"examples": [{
        "bbox": [38.3, 44.6, 48.0, 52.7], "date_from": "2020-07-01", "date_to": "2020-09-30", "source": "gt",
        "layers": ["hotspots", "burn_scars", "summary"]}]}}


class _Params:
    """Разобранные и проверенные параметры запроса."""

    def __init__(self, geometry=None, bbox=None, date_from=None, date_to=None, source="gt"):
        if (geometry is None) == (bbox is None):
            raise core.QueryError("нужно задать ровно одно из: geometry (GeoJSON-полигон) или bbox")
        self.geom = core.parse_geometry(geometry) if geometry is not None else core.parse_bbox(bbox)
        self.d0, self.d1 = core.parse_date(date_from, "date_from"), core.parse_date(date_to, "date_to")
        core.check_period(self.d0, self.d1)
        self.source = core.check_source(source)


def create_app(data_dir: Path | str | None = None) -> FastAPI:
    # docs_url=None: свой /docs на локальном swagger-ui (static/vendor/swagger) — работает без интернета
    app = FastAPI(title="Мониторинг природных пожаров — информационно-аналитический сервис", version="1.0",
                  docs_url=None, redoc_url=None,
                  description="Термоточки VIIRS, контуры гарей Sentinel-2 со степенью поражения и справка "
                              "по площадям для произвольного полигона/bbox и интервала дат. "
                              "Работает на подготовленном наборе сцен (обучающие чипы с геопривязкой).")
    data_dir = Path(data_dir or os.environ.get("FIREMON_SERVICE_DATA") or DEFAULT_DATA_DIR)
    t0 = time.perf_counter()
    try:
        store = core.Store.load(data_dir)
        log.info("данные: %s — %d термоточек, %d контуров, %d сцен (%.1f с)", data_dir, len(store.hotspots),
                 len(store.scars), len(store.chips), time.perf_counter() - t0)
    except FileNotFoundError as e:  # сервис поднимается и без данных — /health подскажет, что делать
        log.error("%s", e)
        store = core.Store.empty()
    app.state.store = store
    app.state.queries = OrderedDict()  # query_id → параметры (для выгрузок результата POST /query)

    @app.exception_handler(core.QueryError)
    async def _query_error(_: Request, exc: core.QueryError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def run_scars(p: _Params, with_union=False):
        return core.query_scars(store, p.geom, p.d0, p.d1, p.source, with_union=with_union)

    def scars_response(p: _Params, fmt: str, with_union: bool):
        res = run_scars(p, with_union)
        if fmt == "shp":
            data = core.to_shapefile_zip(core.scars_gdf(store, res), core.SHP_SCAR_FIELDS, "burn_scars")
            return Response(data, media_type="application/zip",
                            headers={"Content-Disposition": 'attachment; filename="burn_scars_shp.zip"'})
        return Response(core.scars_geojson(store, res), media_type=GEOJSON,
                        headers={"Content-Disposition": 'inline; filename="burn_scars.geojson"'})

    def hotspots_response(p: _Params, fmt: str, as_pixels: bool):
        idx = core.query_hotspots(store, p.geom, p.d0, p.d1, p.source)
        if fmt == "shp":
            gdf = core.hotspot_pixels(store, idx) if as_pixels else store.hotspots.iloc[idx]
            data = core.to_shapefile_zip(gdf, core.SHP_HOTSPOT_FIELDS, "hotspot_pixels" if as_pixels else "hotspots")
            return Response(data, media_type="application/zip",
                            headers={"Content-Disposition": 'attachment; filename="hotspots_shp.zip"'})
        return Response(core.hotspots_geojson(store, idx, as_pixels), media_type=GEOJSON,
                        headers={"Content-Disposition": 'inline; filename="hotspots.geojson"'})

    def summary_response(p: _Params, fmt: str):
        idx = core.query_hotspots(store, p.geom, p.d0, p.d1, p.source)
        summary = core.summarize(store, run_scars(p), idx, p.geom, p.d0, p.d1, p.source)
        if fmt == "csv":
            return Response(core.summary_csv(summary), media_type="text/csv; charset=utf-8",
                            headers={"Content-Disposition": 'attachment; filename="summary.csv"'})
        return JSONResponse(summary)

    # ───────────── API ─────────────

    @app.get("/health", include_in_schema=False)  # короткий алиас: проверки доступности, Docker HEALTHCHECK
    @app.get("/api/v1/health", tags=["служебные"])
    def health():
        ok = len(store.hotspots) > 0 or len(store.scars) > 0
        return {"status": "ok" if ok else "no_data",
                "hint": None if ok else "нет данных: запустите python scripts/build_service_data.py",
                "n_hotspots": len(store.hotspots), "n_burn_scars": int((store.s_class > 0).sum()),
                "n_chips": len(store.chips), "data": store.manifest}

    @app.get("/api/v1/dates", tags=["служебные"], summary="Доступные диапазоны дат по слоям и источникам")
    def dates():
        return store.date_ranges()

    @app.get("/api/v1/aoi", tags=["служебные"], summary="Граница территории мониторинга (GeoJSON)")
    def aoi():
        if store.aoi is None:
            return JSONResponse(status_code=404, content={"detail": "AOI не найден в каталоге данных"})
        return Response(json.dumps(store.aoi, ensure_ascii=False), media_type=GEOJSON)

    @app.get("/api/v1/chips", tags=["служебные"], summary="Охваты подготовленных сцен (чипов) с датами")
    def chips(kind: Literal["af", "bs", "all"] = "all"):
        gdf = store.chips if kind == "all" else store.chips[store.chips["kind"] == kind]
        return Response(gdf.to_json(drop_id=True), media_type=GEOJSON)

    @app.post("/api/v1/query", tags=["запросы"], summary="Термоточки + контуры гарей + справка одним запросом")
    def query(body: QueryBody):
        t = time.perf_counter()
        p = _Params(body.geometry, body.bbox, body.date_from, body.date_to, body.source)
        qid = uuid.uuid4().hex[:12]
        app.state.queries[qid] = p
        while len(app.state.queries) > MAX_SAVED_QUERIES:
            app.state.queries.popitem(last=False)
        idx = core.query_hotspots(store, p.geom, p.d0, p.d1, p.source)
        res = run_scars(p)
        parts = [f'"query_id":"{qid}"']
        if "hotspots" in body.layers:
            parts.append('"hotspots":' + core.hotspots_geojson(store, idx, body.hotspots_as_pixels))
        if "burn_scars" in body.layers:
            out = run_scars(p, True) if body.with_union else res
            parts.append('"burn_scars":' + core.scars_geojson(store, out))
        if "summary" in body.layers:
            parts.append('"summary":' + json.dumps(core.summarize(store, res, idx, p.geom, p.d0, p.d1, p.source),
                                                   ensure_ascii=False))
        meta = {"n_hotspots": int(len(idx)), "n_burn_scars": int(len(res.idx)), "n_clipped": len(res.clipped),
                "source": p.source, "elapsed_ms": round(1000 * (time.perf_counter() - t), 1)}
        parts.append('"meta":' + json.dumps(meta))
        # ответ собирается из заранее сериализованных Feature — быстрее, чем json.dumps десятков тысяч объектов
        return Response("{" + ",".join(parts) + "}", media_type="application/json")

    @app.get("/api/v1/query/{query_id}/{layer}", tags=["запросы"],
             summary="Выгрузка результата ранее выполненного POST /query (в т.ч. для произвольного полигона)")
    def query_export(query_id: str, layer: Layer, format: Literal["geojson", "shp", "json", "csv"] = "geojson",
                     hotspots_as_pixels: bool = False, with_union: bool = False):
        p = app.state.queries.get(query_id)
        if p is None:
            return JSONResponse(status_code=404, content={"detail": f"query_id {query_id} не найден (устарел)"})
        if layer == "summary":
            return summary_response(p, "csv" if format == "csv" else "json")
        if format not in ("geojson", "shp"):
            raise core.QueryError("format: для векторных слоёв допустимо geojson|shp")
        return scars_response(p, format, with_union) if layer == "burn_scars" \
            else hotspots_response(p, format, hotspots_as_pixels)

    bbox_q = Query(..., description="minx,miny,maxx,maxy (EPSG:4326)", examples=["38.3,44.6,48.0,52.7"])
    d0_q = Query(None, description="YYYY-MM-DD, включительно", examples=["2020-07-01"])
    d1_q = Query(None, description="YYYY-MM-DD, включительно", examples=["2020-09-30"])

    @app.get("/api/v1/hotspots", tags=["запросы"], summary="Термоточки (GeoJSON или Shapefile)")
    def hotspots(bbox: str = bbox_q, date_from: str | None = d0_q, date_to: str | None = d1_q,
                 source: Source = "gt", format: Literal["geojson", "shp"] = "geojson", as_pixels: bool = False):
        return hotspots_response(_Params(None, bbox, date_from, date_to, source), format, as_pixels)

    @app.get("/api/v1/burn-scars", tags=["запросы"], summary="Контуры гарей (GeoJSON или zip с Shapefile)")
    def burn_scars(bbox: str = bbox_q, date_from: str | None = d0_q, date_to: str | None = d1_q,
                   source: Source = "gt", format: Literal["geojson", "shp"] = "geojson", with_union: bool = False):
        return scars_response(_Params(None, bbox, date_from, date_to, source), format, with_union)

    @app.get("/api/v1/summary", tags=["запросы"], summary="Аналитическая справка (JSON или CSV)")
    def summary(bbox: str = bbox_q, date_from: str | None = d0_q, date_to: str | None = d1_q,
                source: Source = "gt", format: Literal["json", "csv"] = "json"):
        return summary_response(_Params(None, bbox, date_from, date_to, source), format)

    # ───────────── веб-интерфейс ─────────────

    @app.get("/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title + " — API",
                                   swagger_js_url="/static/vendor/swagger/swagger-ui-bundle.js",
                                   swagger_css_url="/static/vendor/swagger/swagger-ui.css",
                                   swagger_favicon_url="/static/vendor/swagger/favicon-32x32.png")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
