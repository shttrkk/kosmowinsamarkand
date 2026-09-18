# service/ — информационно-аналитический сервис

FastAPI + Leaflet. По полигону/bbox и интервалу дат: термоточки, контуры гарей (GeoJSON/Shapefile) и справка
по площадям (JSON/CSV). Полное описание — [`reports/service.md`](../reports/service.md).

```powershell
# из корня репозитория; зависимости — общий requirements.txt
python scripts/build_service_data.py        # необязательно: пересборка слоёв gt + pred из data/train (~2 мин)
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000     # или: python service/run.py
```

- http://localhost:8000 — карта (кнопка «Демо-запрос»), http://localhost:8000/docs — OpenAPI.
- Без пересборки сервис стартует из снимка эталонных слоёв `data/*.geojson.gz` (лежит в git, только `source=gt`).
- Другой каталог данных: переменная окружения `FIREMON_SERVICE_DATA`.

```bash
curl "http://localhost:8000/api/v1/summary?bbox=39.25,47.95,39.50,48.10&date_from=2024-09-15&date_to=2024-10-10"
curl -o burn_scars_shp.zip "http://localhost:8000/api/v1/burn-scars?bbox=39.25,47.95,39.50,48.10&format=shp"
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" \
     -d '{"bbox":[39.25,47.95,39.50,48.10],"date_from":"2024-09-15","date_to":"2024-10-10","source":"gt"}'
```

| файл | роль |
|---|---|
| `app.py` | эндпоинты, разбор параметров, ошибки 422 |
| `core.py` | чистые функции: пространственно-временной отбор, clip и площади в UTM, справка, экспорт |
| `vectorize.py` | маска → термоточки / контуры, фильтр спеклов с сохранением площади |
| `build.py`, `predictors.py` | сборка слоёв (`scripts/build_service_data.py`), адаптеры моделей |
| `static/` | веб-карта; `vendor/` — локальные Leaflet 1.9.4 (BSD-2) и Leaflet.draw 1.0.4 (MIT) |
| `data/` | `firemon.gpkg` (не в git), снимок `*.geojson.gz`, `manifest.json`, `aoi/` |

Тесты: `python -m pytest tests/test_service.py -q`.
