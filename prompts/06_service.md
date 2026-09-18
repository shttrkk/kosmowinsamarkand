# Промпт 6 — Информационно-аналитический сервис (REST API + веб-карта)

> После промпта 1 (нужны `io.py`, геопривязка train-чипов). Параллельно с 2, 3, 4. Позже (после 3–4) — обновить данные предсказаниями финальных моделей. Прочитай `CLAUDE.md`. Работай в `service/`, `scripts/build_service_data.py`, `tests/test_service.py`, `reports/service.md`.
> Раздел критериев на 12 баллов: работает + REST (4, «0 или 4»), карта + выгрузка векторов с корректной геометрией и атрибутами (4), аналитическая справка с корректным расчётом площади (4).

## Цель
Веб-приложение и REST API, которые по **полигону или bbox + интервалу дат** возвращают: (1) термоточки за период, (2) векторные контуры гарей с атрибутами (id, класс степени, площадь га) с выгрузкой в GeoJSON и Shapefile, (3) аналитическую справку (суммарная площадь гари, га, и распределение по трём степеням) с выгрузкой в JSON/CSV. ТЗ прямо разрешает работать на заранее подготовленном наборе сцен — полный конвейер скачивания снимков не нужен.

## Данные сервиса (`scripts/build_service_data.py`)
Источник — обучающие чипы (у них есть геопривязка, даты и маски) + предсказания наших моделей на них.
1. **Термоточки**: для каждого AF-чипа train — пиксели маски (режим `--source gt` — эталон, `--source pred` — наш `AFPredictor`; по умолчанию собирать оба слоя с полем `source`). Каждый пиксель → точка-центроид в EPSG:4326 с атрибутами: `hotspot_id`, `chip_id`, `acq_datetime` (UTC ISO), `satellite`, `i4_bt` (K), `i5_bt`, `daynight`, `landcover`, `confidence` (вероятность модели, для gt = 1.0), `source`. Также полигон пикселя 375×375 м (опционально в выгрузке). Формат хранения: GeoPackage `service/data/hotspots.gpkg` (слой `hotspots`) + GeoParquet/GeoJSON fallback.
2. **Гари**: для каждого BS-чипа train — маска (gt и pred) → `rasterio.features.shapes` по классам 1/2/3 в CRS чипа (UTM 32637/32638) → полигоны, площадь считается **в UTM** (`geom.area / 10000` га — пиксель 20 м = 0.04 га; сверь с `burn_area_ha` из meta для gt — должно совпасть с точностью ~1 %, это проверка корректности для критерия «корректность расчёта площади») → перепроецировать в EPSG:4326 для хранения/отдачи. Атрибуты: `contour_id`, `chip_id`, `fire_event_id`, `severity_class` (1/2/3), `severity_label` (слабая/средняя/сильная), `area_ha`, `date_pre`, `date_post`, `source`. Плюс объединённый контур всей гари на чип (`severity_class=0`, «весь контур»). Сглаживание/упрощение геометрии — `simplify(5 м)` с сохранением топологии, удаление полигонов < 3 пикселей. Слой `burn_scars` в том же gpkg.
3. **Индексы**: bbox чипов, даты — для быстрого пространственно-временного отбора (GeoPandas `sindex` при старте или SQLite/Spatialite). 420+224 чипов — всё влезет в память.
4. AOI-полигон из `fire-aoi/fire_monitoring_aoi.geojson` — для отображения границы.
Скрипт должен работать и без обученных моделей (только gt) — чтобы сервис можно было поднять сразу; когда появятся веса, перезапуск добавит слой pred.

## REST API (`service/app.py`, FastAPI + uvicorn)
Запуск: `python -m uvicorn service.app:app --host 0.0.0.0 --port 8000` (и `python service/run.py`). OpenAPI-доки на `/docs` — это уже демонстрация REST.
- `POST /api/v1/query` — тело: `{"geometry": <GeoJSON Polygon|MultiPolygon> | "bbox": [minx,miny,maxx,maxy], "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD", "source": "gt|pred", "layers": ["hotspots","burn_scars","summary"]}` → JSON: `hotspots` (GeoJSON FeatureCollection), `burn_scars` (FeatureCollection, только части полигонов внутри запроса — clip, площади пересчитать после clip в UTM!), `summary` (см. ниже), `query_id`, `meta` (число объектов, время обработки мс).
- `GET /api/v1/hotspots?bbox=&date_from=&date_to=&source=` — GeoJSON.
- `GET /api/v1/burn-scars?bbox=&date_from=&date_to=&format=geojson|shp&source=` — GeoJSON или zip с Shapefile (shp+shx+dbf+prj+cpg, кодировка UTF-8, CRS EPSG:4326 в .prj; атрибуты обрезаны до 10 символов имени — задай короткие: `cid, chip_id, event_id, sev_cls, sev_lbl, area_ha, date_pre, date_post, source`).
- `GET /api/v1/summary?bbox=&date_from=&date_to=&format=json|csv&source=` — справка: `{"area_total_ha", "by_severity": {"1": {"area_ha", "share_pct", "label"}, "2":…, "3":…}, "n_contours", "n_hotspots", "period", "aoi_area_km2", "crs_note": "площади рассчитаны в UTM 37N/38N", "generated_at"}`; CSV — таблица. Плюс `GET /api/v1/summary.pdf`? — не обязательно; JSON и CSV достаточно («машиночитаемый формат»).
- `GET /api/v1/aoi` — граница территории; `GET /api/v1/health`; `GET /api/v1/dates` — доступный диапазон дат по слоям (чтобы демо-запрос точно что-то возвращал).
- Ошибки: невалидная геометрия/даты → 422 с текстом. Все ответы за конечное время: цель < 2 с на запрос по всему AOI за весь период.
- Логика клиппинга и площадей в `service/core.py` (чистые функции, тестируемые без HTTP).

## Веб-интерфейс (`service/static/index.html` + JS, Leaflet с CDN и локальной копией на случай отсутствия интернета на защите — положи leaflet.js/css в `static/vendor/`)
- Подложка OSM (и fallback — без подложки, если нет сети; предусмотреть переключатель на локальную «пустую» подложку с AOI-контуром).
- Инструмент рисования прямоугольника/полигона (Leaflet.draw или ручной bbox), два поля дат, переключатель gt/pred, кнопка «Запросить».
- Отображение: термоточки (кластеризация опционально; цвет день/ночь; popup с атрибутами), контуры гарей (заливка по классу: 1 — жёлтый, 2 — оранжевый, 3 — красный; popup с площадью), панель справки (таблица площадей по степеням, суммарная, число объектов), кнопки «Скачать GeoJSON», «Скачать Shapefile», «Скачать справку JSON/CSV».
- Кнопка «Демо-запрос» — заранее заданный bbox и период, где точно есть и термоточки, и гари (найди на данных).
- Всё на русском, аккуратно, без тяжёлых фреймворков.

## Тесты (`tests/test_service.py`, httpx + TestClient)
- `/health`; `/query` по bbox всего AOI за весь период → непустые слои; по bbox вне AOI → пустые слои и `area_total_ha == 0`; невалидные даты → 422; Shapefile-выгрузка → zip с 5 файлами, читается GeoPandas, CRS 4326, атрибуты на месте; суммарная площадь gt по чипу (bbox чипа, даты чипа) равна `burn_area_ha` из meta ±1 %; `by_severity` суммируется в `area_total_ha`.

## Документация
- `reports/service.md`: архитектура, схема данных, эндпоинты с примерами `curl` и ответами (сокращённо), скриншоты интерфейса (сделай через playwright/selenium или вручную — `reports/figures/service_*.png`), как считаются площади (UTM, 0.04 га/пиксель, clip), ограничения (работает на подготовленном наборе сцен; как подключить новые чипы).
- В README раздел «Сервис»: установка, запуск, URL, 3 примера curl.
- `service/README.md` — то же кратко.

## Критерии готовности
- `python -m uvicorn service.app:app` поднимается, `/docs` работает, демо-запрос в UI возвращает карту, контуры, справку и выгрузки;
- тесты зелёные; `scripts/build_service_data.py` пересобирает данные; `reports/service.md`, `reports/tasklog/06_service.md`, коммит.
- После появления финальных весов (промпты 3–4): перезапустить `build_service_data.py --source both`, проверить слой pred.
