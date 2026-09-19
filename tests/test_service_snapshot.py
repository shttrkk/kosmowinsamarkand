"""Сервис из чистого клона: работает на снимке эталонных слоёв `service/data/*.geojson.gz` (в git), без data/train.

Остальные тесты сервиса (`test_service.py`) пересобирают слои из `data/train` и без данных пропускаются — этот файл
гарантирует, что после `git clone` сервис поднимается и демо-запрос даёт числа из отчёта.
"""
from __future__ import annotations

import pytest

from conftest import ROOT

SNAPSHOT = ROOT / "service" / "data"
DEMO = {"bbox": [39.25, 47.95, 39.50, 48.10], "date_from": "2024-09-15", "date_to": "2024-10-10", "source": "gt"}

pytestmark = pytest.mark.skipif(not (SNAPSHOT / "burn_scars.geojson.gz").exists(),
                                reason="нет снимка service/data/*.geojson.gz")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from service.app import app

    with TestClient(app) as c:
        yield c


def test_health_and_pages(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["n_chips"] == 644
    index = client.get("/")
    assert index.status_code == 200 and "/static/app.js" in index.text
    for path in ("/static/app.js", "/static/style.css", "/static/vendor/leaflet/leaflet.js",
                 "/static/vendor/polygon-clipping/polygon-clipping.umd.min.js", "/docs"):
        assert client.get(path).status_code == 200, path


def test_demo_query_on_gt_snapshot(client):
    """Числа демо-запроса из отчёта и HANDOFF: 98 термоточек, 337 контуров, 1 758.92 га."""
    r = client.post("/api/v1/query", json=DEMO)
    assert r.status_code == 200
    data = r.json()
    assert data["meta"]["n_hotspots"] == 98 and data["meta"]["n_burn_scars"] == 337
    assert data["summary"]["area_total_ha"] == pytest.approx(1758.92, abs=0.01)
    shares = [data["summary"]["by_severity"][k]["share_pct"] for k in ("1", "2", "3")]
    assert sum(shares) == pytest.approx(100, abs=0.1)
    props = data["burn_scars"]["features"][0]["properties"]
    assert {"contour_id", "severity_class", "area_ha", "chip_id", "date_pre", "date_post"} <= set(props)


def test_demo_exports_downloadable(client):
    qid = client.post("/api/v1/query", json=DEMO).json()["query_id"]
    shp = client.get(f"/api/v1/query/{qid}/burn_scars", params={"format": "shp"})
    assert shp.status_code == 200 and shp.content[:2] == b"PK"
    csv = client.get(f"/api/v1/query/{qid}/summary", params={"format": "csv"})
    assert csv.status_code == 200 and "1758.92" in csv.text
