"""Выгрузки демо-запроса сервиса в out/demo_exports/ (GeoJSON, Shapefile, справка) — готовые файлы на случай, если сервис нельзя поднять.

    python scripts/make_demo_exports.py

Запрос тот же, что у кнопки «Демо-запрос»: bbox 39.25,47.95,39.50,48.10, 2024-09-15…2024-10-10, слои gt и pred.
Работает через TestClient (uvicorn не нужен). Слой pred есть только после `python scripts/build_service_data.py --source both`.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from service.app import app  # noqa: E402

OUT = ROOT / "out" / "demo_exports"  # каталог out/ в git не входит; готовый набор — в пакете артефактов
DEMO = {"bbox": [39.25, 47.95, 39.50, 48.10], "date_from": "2024-09-15", "date_to": "2024-10-10"}
EXPORTS = [("burn_scars", "geojson", "burn_scars.geojson"), ("burn_scars", "shp", "burn_scars_shp.zip"),
           ("hotspots", "geojson", "hotspots.geojson"), ("hotspots", "shp", "hotspots_shp.zip"),
           ("summary", "json", "summary.json"), ("summary", "csv", "summary.csv")]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as client:
        for source in ("gt", "pred"):
            resp = client.post("/api/v1/query", json={**DEMO, "source": source})
            resp.raise_for_status()
            data = resp.json()
            meta, summary = data["meta"], data["summary"]
            if meta["n_hotspots"] + meta["n_burn_scars"] == 0:
                print(f"{source}: слой пуст (для pred нужен build_service_data.py --source both) — пропущен")
                continue
            for layer, fmt, name in EXPORTS:
                r = client.get(f"/api/v1/query/{data['query_id']}/{layer}", params={"format": fmt})
                r.raise_for_status()
                if not r.content:
                    raise RuntimeError(f"{source}/{name}: пустой ответ")
                if fmt == "shp":  # zip должен открываться и содержать .shp
                    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
                    if not any(n.endswith(".shp") for n in names):
                        raise RuntimeError(f"{source}/{name}: в архиве нет .shp ({names})")
                elif fmt in ("geojson", "json"):
                    json.loads(r.content)
                path = OUT / f"{source}_{name}"
                path.write_bytes(r.content)
                print(f"{path.relative_to(ROOT)}  {len(r.content) / 1024:.1f} КБ")
            print(f"{source}: гарей {summary['area_total_ha']} га, контуров {meta['n_burn_scars']}, термоточек {meta['n_hotspots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
