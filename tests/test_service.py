"""Сервис: ядро (clip, площади в UTM) на синтетике + REST API на данных, собранных из подвыборки train."""
from __future__ import annotations

import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
from affine import Affine
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service import core  # noqa: E402
from service.build import build_layers, read_layers, to_wgs84, write_layers  # noqa: E402
from service.vectorize import despeckle_severity, scars_from_mask  # noqa: E402

WORLD = [30.0, 40.0, 55.0, 58.0]        # заведомо шире охвата данных
OUTSIDE = [100.0, 10.0, 101.0, 11.0]    # вне AOI
EPSG = 32637
TRANSFORM = Affine(20.0, 0, 500000.0, 0, -20.0, 5300000.0)  # 20 м, центральный меридиан зоны 37N
META = {"fire_event_id": "FE1", "date_pre": "2024-08-01", "date_post": "2024-08-20"}


# ───────────────────────── синтетика: без data/train ─────────────────────────

def _synthetic_mask() -> np.ndarray:
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:50] = 1      # 800 px = 32 га
    m[30:50, 10:50] = 2      # 800 px = 32 га
    m[50:60, 10:30] = 3      # 200 px = 8 га
    m[15, 15] = 3            # спекл внутри класса 1 → вливается в класс 1
    m[80, 80] = 2            # изолированный пиксель → удаляется
    m[0:5, 0:5] = 255        # nodata → фон
    return m


@pytest.fixture(scope="module")
def synth_store() -> core.Store:
    recs, geoms = scars_from_mask(_synthetic_mask(), TRANSFORM, EPSG, "BS_x", META, "gt")
    import pandas as pd

    df = pd.DataFrame(recs)
    scars = gpd.GeoDataFrame(df, geometry=to_wgs84(geoms, df["epsg"].to_numpy()), crs="EPSG:4326")
    empty = core.Store.empty()
    return core.Store(empty.hotspots, scars, empty.chips)


def test_despeckle_keeps_burn_area():
    m = _synthetic_mask()
    out = despeckle_severity(m, 3)
    assert out[15, 15] == 1 and out[80, 80] == 0 and out[0, 0] == 0
    assert (out > 0).sum() == 1800
    assert np.array_equal(despeckle_severity(m, 1) > 0, (m > 0) & (m < 4))


def test_vector_area_matches_pixel_count():
    recs, geoms = scars_from_mask(_synthetic_mask(), TRANSFORM, EPSG, "BS_x", META, "gt")
    area = {k: sum(r["area_ha"] for r in recs if r["severity_class"] == k) for k in (0, 1, 2, 3)}
    assert area == pytest.approx({0: 72.0, 1: 32.0, 2: 32.0, 3: 8.0})
    assert all(g.is_valid for g in geoms)
    assert {r["severity_label"] for r in recs} == {"весь контур", "слабая", "средняя", "сильная"}


def test_query_full_and_clip(synth_store):
    res = core.query_scars(synth_store, core.parse_bbox(WORLD), None, None, "gt")
    assert res.area_ha.sum() == pytest.approx(72.0, rel=1e-4) and not res.clipped
    # западная половина гари: x < 500000 + 30 px * 20 м → классы 1 и 2 режутся пополам, класс 3 — целиком
    half = to_wgs84([box(499000, 5290000, 500600, 5310000)], [EPSG])[0]
    res = core.query_scars(synth_store, half, None, None, "gt")
    cls = synth_store.s_class[res.idx]
    got = {k: res.area_ha[cls == k].sum() for k in (1, 2, 3)}
    assert got == pytest.approx({1: 16.0, 2: 16.0, 3: 8.0}, rel=2e-3)
    assert len(res.clipped) == 2
    summary = core.summarize(synth_store, res, np.array([], int), half, None, None, "gt")
    assert summary["area_total_ha"] == pytest.approx(40.0, rel=2e-3)
    gdf = core.scars_gdf(synth_store, res)
    assert gdf.geometry.is_valid.all() and gdf["area_ha"].sum() == pytest.approx(40.0, rel=2e-3)
    assert '"FeatureCollection"' in core.scars_geojson(synth_store, res)


def test_query_dates_and_source(synth_store):
    g = core.parse_bbox(WORLD)
    assert len(core.query_scars(synth_store, g, date(2024, 8, 10), date(2024, 8, 11), "gt").idx) == 3  # внутри pre..post
    assert len(core.query_scars(synth_store, g, date(2024, 8, 21), None, "gt").idx) == 0
    assert len(core.query_scars(synth_store, g, None, date(2024, 7, 31), "gt").idx) == 0
    assert len(core.query_scars(synth_store, g, None, None, "pred").idx) == 0
    assert len(core.query_scars(synth_store, g, None, None, "gt", with_union=True).idx) == 4


@pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d", "10,10,5,20", "0,0,200,10"])
def test_bad_bbox(bad):
    with pytest.raises(core.QueryError):
        core.parse_bbox(bad)


def test_bad_geometry_and_dates():
    bowtie = {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}
    for geom in (bowtie, {"type": "Point", "coordinates": [1, 2]}, {"type": "Polygon"}):
        with pytest.raises(core.QueryError):
            core.parse_geometry(geom)
    with pytest.raises(core.QueryError):
        core.parse_date("2024-13-01", "date_from")
    with pytest.raises(core.QueryError):
        core.check_period(date(2024, 2, 1), date(2024, 1, 1))


# ───────────────────────── REST API на подвыборке train ─────────────────────────

BS_SUBSET = ["BS_tr_000001", "BS_tr_000002", "BS_tr_000003", "BS_tr_000202"]


@pytest.fixture(scope="module")
def service_data(train_dir, tmp_path_factory) -> Path:
    from firemon.io import read_meta

    meta = read_meta(train_dir, "af")
    af = meta[meta["n_fire_px"] > 0]["chip_id"].head(8).tolist() + meta[meta["n_fire_px"] == 0]["chip_id"].head(2).tolist()
    out = tmp_path_factory.mktemp("service_data")
    write_layers(build_layers(train_dir, ("gt", "pred"), af_chips=af, bs_chips=BS_SUBSET), out)
    return out


@pytest.fixture(scope="module")
def client(service_data):
    from fastapi.testclient import TestClient

    from service.app import create_app

    return TestClient(create_app(service_data))


def test_health_and_meta_endpoints(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert client.get("/health").json() == r.json()  # короткий алиас
    d = client.get("/api/v1/dates").json()
    assert d["gt"]["hotspots"]["n"] > 0 and d["gt"]["burn_scars"]["date_min"] <= d["gt"]["burn_scars"]["date_max"]
    assert len(client.get("/api/v1/chips").json()["features"]) == 10 + len(BS_SUBSET)
    assert client.get("/docs").status_code == 200 and client.get("/").status_code == 200


def test_query_whole_aoi(client):
    r = client.post("/api/v1/query", json={"bbox": WORLD, "source": "gt"})
    assert r.status_code == 200
    j = r.json()
    assert len(j["hotspots"]["features"]) > 0 and len(j["burn_scars"]["features"]) > 0 and j["query_id"]
    s = j["summary"]
    assert s["area_total_ha"] > 0 and s["n_hotspots"] == len(j["hotspots"]["features"])
    assert sum(v["area_ha"] for v in s["by_severity"].values()) == pytest.approx(s["area_total_ha"], abs=0.02)
    assert sum(v["share_pct"] for v in s["by_severity"].values()) == pytest.approx(100, abs=0.05)
    f = j["burn_scars"]["features"][0]["properties"]
    assert {"contour_id", "severity_class", "severity_label", "area_ha", "chip_id", "date_pre", "date_post"} <= set(f)
    h = j["hotspots"]["features"][0]["properties"]
    assert {"hotspot_id", "acq_datetime", "satellite", "i4_bt", "i5_bt", "daynight", "landcover", "confidence"} <= set(h)
    assert h["confidence"] == 1.0 and h["acq_datetime"].endswith("Z")
    assert client.post("/api/v1/query", json={"bbox": WORLD, "source": "pred"}).json()["meta"]["source"] == "pred"


def test_query_outside_is_empty(client):
    j = client.post("/api/v1/query", json={"bbox": OUTSIDE}).json()
    assert j["hotspots"]["features"] == [] and j["burn_scars"]["features"] == []
    assert j["summary"]["area_total_ha"] == 0 and j["summary"]["n_contours"] == 0


def test_query_polygon_and_exports(client):
    poly = mapping(box(*WORLD))
    j = client.post("/api/v1/query", json={"geometry": poly, "layers": ["summary"]}).json()
    assert "hotspots" not in j and j["summary"]["area_total_ha"] > 0
    qid = j["query_id"]
    assert client.get(f"/api/v1/query/{qid}/summary?format=csv").text.startswith("﻿indicator,")
    assert client.get(f"/api/v1/query/{qid}/burn_scars").json()["type"] == "FeatureCollection"
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/v1/query/{qid}/hotspots?format=shp").content))
    assert "hotspots.shp" in z.namelist()
    assert client.get("/api/v1/query/nope/summary").status_code == 404


@pytest.mark.parametrize("payload", [
    {"bbox": WORLD, "date_from": "2024-02-30"},
    {"bbox": WORLD, "date_from": "2024-05-01", "date_to": "2024-01-01"},
    {"bbox": [10, 10, 5, 20]},
    {},
    {"bbox": WORLD, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
    {"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}},
    {"bbox": WORLD, "source": "firms"},
])
def test_invalid_requests_422(client, payload):
    r = client.post("/api/v1/query", json=payload)
    assert r.status_code == 422 and r.json()["detail"]


def test_get_endpoints_validation(client):
    assert client.get("/api/v1/summary", params={"bbox": "1,2,3", "date_from": "2020-01-01"}).status_code == 422
    assert client.get("/api/v1/hotspots", params={"bbox": "30,40,55,58", "date_from": "вчера"}).status_code == 422
    assert client.get("/api/v1/burn-scars").status_code == 422


def test_shapefile_export(client, tmp_path):
    r = client.get("/api/v1/burn-scars", params={"bbox": ",".join(map(str, WORLD)), "format": "shp"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(Path(n).suffix for n in z.namelist()) == [".cpg", ".dbf", ".prj", ".shp", ".shx"]
    assert z.read("burn_scars.cpg").decode().strip().upper() == "UTF-8"
    z.extractall(tmp_path)
    gdf = gpd.read_file(tmp_path / "burn_scars.shp")
    assert gdf.crs.to_epsg() == 4326 and len(gdf) > 0 and gdf.geometry.is_valid.all()
    assert list(gdf.columns[:-1]) == ["cid", "chip_id", "event_id", "sev_cls", "sev_lbl", "area_ha", "date_pre",
                                      "date_post", "source"]
    assert set(gdf["sev_lbl"]) <= {"слабая", "средняя", "сильная"} and set(gdf["sev_cls"]) <= {1, 2, 3}
    gj = client.get("/api/v1/burn-scars", params={"bbox": ",".join(map(str, WORLD))}).json()
    assert len(gj["features"]) == len(gdf)
    assert gdf["area_ha"].sum() == pytest.approx(sum(f["properties"]["area_ha"] for f in gj["features"]), rel=1e-6)
    # площадь из атрибута совпадает с площадью геометрии, пересчитанной в UTM
    sub = gdf[gdf["chip_id"] == "BS_tr_000202"]
    assert sub.to_crs(32637).area.sum() / 1e4 == pytest.approx(sub["area_ha"].sum(), rel=1e-3)


def test_chip_area_matches_meta(client, service_data, train_dir):
    """Сумма площадей gt-контуров по bbox и датам чипа = burn_area_ha из meta.csv (±1 %)."""
    chips = read_layers(service_data)["chips"]
    for row in chips[chips["kind"] == "bs"].itertuples():
        b = shapely.buffer(row.geometry, 1e-4).bounds  # охват чипа в градусах — не прямоугольник: берём с запасом
        s = client.get("/api/v1/summary", params={"bbox": ",".join(f"{v:.6f}" for v in b), "date_from": row.date_from,
                                                  "date_to": row.date_to, "source": "gt"}).json()
        assert s["area_total_ha"] == pytest.approx(row.burn_area_ha, rel=0.01), row.chip_id
        assert sum(v["area_ha"] for v in s["by_severity"].values()) == pytest.approx(s["area_total_ha"], abs=0.02)


def test_clip_reduces_area(client, service_data):
    chips = read_layers(service_data)["chips"]
    g = chips[chips["chip_id"] == "BS_tr_000202"].geometry.iloc[0]
    minx, miny, maxx, maxy = g.bounds
    params = {"date_from": "2024-09-01", "date_to": "2024-10-31"}
    full = client.get("/api/v1/summary", params={**params, "bbox": f"{minx},{miny},{maxx},{maxy}"}).json()
    west = client.get("/api/v1/summary", params={**params, "bbox": f"{minx},{miny},{(minx + maxx) / 2},{maxy}"}).json()
    east = client.get("/api/v1/summary", params={**params, "bbox": f"{(minx + maxx) / 2},{miny},{maxx},{maxy}"}).json()
    assert 0 < west["area_total_ha"] < full["area_total_ha"]
    assert west["area_total_ha"] + east["area_total_ha"] == pytest.approx(full["area_total_ha"], rel=1e-3)


def test_polygon_query_areas_consistent(client, service_data):
    """Произвольный полигон (не bbox): by_severity суммируется в area_total_ha, clip не увеличивает площади."""
    chips = read_layers(service_data)["chips"]
    minx, miny, maxx, maxy = chips[chips["chip_id"] == "BS_tr_000202"].geometry.iloc[0].bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    # ромб с дополнительной вершиной: режет гарь наискось, ни одна сторона не параллельна сетке пикселей
    ring = [[cx, miny], [maxx, cy], [cx + (maxx - cx) * 0.3, maxy], [minx, cy + (maxy - cy) * 0.4], [cx, miny]]
    body = {"geometry": {"type": "Polygon", "coordinates": [ring]}, "date_from": "2024-09-01", "date_to": "2024-10-31"}
    j = client.post("/api/v1/query", json=body).json()
    s = j["summary"]
    assert j["meta"]["n_clipped"] > 0 and s["area_total_ha"] > 0
    assert sum(v["area_ha"] for v in s["by_severity"].values()) == pytest.approx(s["area_total_ha"], abs=0.01)
    feats = j["burn_scars"]["features"]
    assert sum(f["properties"]["area_ha"] for f in feats) == pytest.approx(s["area_total_ha"], abs=0.01)

    full = client.get("/api/v1/burn-scars", params={"bbox": f"{minx - 0.01},{miny - 0.01},{maxx + 0.01},{maxy + 0.01}",
                                                    **{k: body[k] for k in ("date_from", "date_to")}}).json()
    before = {f["properties"]["contour_id"]: f["properties"]["area_ha"] for f in full["features"]}
    assert set(f["properties"]["contour_id"] for f in feats) <= set(before)
    n_smaller = 0
    for f in feats:
        p = f["properties"]
        assert 0 < p["area_ha"] <= before[p["contour_id"]] + 1e-4, p["contour_id"]
        n_smaller += p["area_ha"] < before[p["contour_id"]] - 1e-4
    assert n_smaller > 0 and s["area_total_ha"] < sum(before.values())
    # площадь в атрибуте = площади обрезанной геометрии, пересчитанной в UTM
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    assert gdf.geometry.is_valid.all()
    assert gdf.to_crs(32637).area.sum() / 1e4 == pytest.approx(s["area_total_ha"], rel=1e-3)


def test_docs_served_offline(client):
    """/docs подключает swagger-ui только из локального static/vendor — без CDN."""
    html = client.get("/docs").text
    assert "/static/vendor/swagger/swagger-ui-bundle.js" in html and "/static/vendor/swagger/swagger-ui.css" in html
    assert "cdn." not in html and "https://" not in html
    for f in ("swagger-ui-bundle.js", "swagger-ui.css"):
        assert client.get(f"/static/vendor/swagger/{f}").status_code == 200
    assert client.get("/openapi.json").json()["paths"]["/api/v1/query"]


def test_fallback_snapshot_loads(service_data, tmp_path):
    for f in service_data.glob("*.geojson.gz"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    store = core.Store.load(tmp_path)  # без gpkg — из снимка (только gt)
    assert len(store.scars) > 0 and set(store.s_source) == {"gt"}
    res = core.query_scars(store, core.parse_bbox(WORLD), None, None, "gt")
    assert res.area_ha.sum() > 0


def test_service_starts_without_data(tmp_path):
    from fastapi.testclient import TestClient

    from service.app import create_app

    c = TestClient(create_app(tmp_path))
    assert c.get("/api/v1/health").json()["status"] == "no_data"
    assert c.post("/api/v1/query", json={"bbox": WORLD}).json()["summary"]["area_total_ha"] == 0
