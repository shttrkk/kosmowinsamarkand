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

## Корректность площадей

- Площадь считается по векторной геометрии **в UTM-зоне чипа** (32637/32638): 0.04 га на пиксель 20 м. Контуры,
  пересекающие границу запроса, обрезаются в UTM, площадь пересчитывается.
- **Фильтр мелких фрагментов (< 3 px) сохраняет площадь гари**: изолированные пятна удаляются, а мелочь внутри гари
  вливается в соседний класс гари, но не в фон (обычный `sieve` терял до 5.7 % площади чипа).
- **Сверка с `meta.burn_area_ha` по всем 224 BS-чипам: максимальное отклонение 0.025 %, вне допуска 1 % — ни одного
  чипа** (печатается при каждой сборке). Сдвиг между классами из-за фильтра: медиана 0.9 %, максимум 4 % площади гари.
- `python scripts/build_service_data.py --min-pixels 1` отключает фильтр: площади по классам совпадают с растром точно
  (ценой ~99 тыс. полигонов вместо 43.7 тыс.).
- `/docs` работает без интернета (swagger-ui лежит в `static/vendor/swagger/`).

| файл | роль |
|---|---|
| `app.py` | эндпоинты, разбор параметров, ошибки 422 |
| `core.py` | чистые функции: пространственно-временной отбор, clip и площади в UTM, справка, экспорт |
| `vectorize.py` | маска → термоточки / контуры, фильтр спеклов с сохранением площади |
| `build.py`, `predictors.py` | сборка слоёв (`scripts/build_service_data.py`), адаптеры моделей |
| `static/` | веб-карта; `vendor/` — локальные Leaflet 1.9.4 (BSD-2), Leaflet.draw 1.0.4 (MIT), swagger-ui 5.17.14 (Apache-2.0) |
| `data/` | `firemon.gpkg` (не в git), снимок `*.geojson.gz`, `manifest.json`, `aoi/` |

Тесты: `python -m pytest tests/test_service.py -q`.
