# 06 — Информационно-аналитический сервис (REST API + веб-карта)

Ветка `task/service`. Дата: 2026-09-18.

## Что сделано
- **Данные сервиса** (`scripts/build_service_data.py`, `service/build.py`, `service/vectorize.py`): из 420 AF- и 224 BS-чипов
  train собираются слои `hotspots` (72 790: gt 9 725 + pred 63 065), `burn_scars` (gt 43 674 + pred 49 323 контура классов 1–3
  + объединённые контуры `severity_class=0`), `chips` (охваты сцен с датами) → `service/data/firemon.gpkg` (176 МБ, не в git)
  + снимок gt `*.geojson.gz` (16 МБ, в git — сервис стартует сразу после клонирования) + `manifest.json`. Сборка ~110 с.
- **Площади**: векторизация и площадь в UTM-зоне чипа; фильтр спеклов < 3 px, сохраняющий площадь гари
  (`despeckle_severity`). Сверка с `meta.burn_area_ha` по всем 224 чипам: макс. отклонение 0.025 %, чипов > 1 % — 0.
- **REST API** (`service/app.py`, `service/core.py`): `POST /api/v1/query` (bbox или GeoJSON-полигон, clip с пересчётом
  площади в UTM), `GET /hotspots`, `/burn-scars` (geojson|shp), `/summary` (json|csv), `/query/{id}/{layer}` (выгрузки
  результата, в т. ч. для полигона), `/aoi`, `/chips`, `/dates`, `/health`; 422 с текстом на невалидные параметры; `/docs`.
  Весь охват за весь период (52 МБ JSON) — ≈0.4 с на сервере; демо-запрос — 6–15 мс.
- **Веб-карта** (`service/static/`): Leaflet + Leaflet.draw локально, рисование прямоугольника/полигона, даты, gt/pred,
  «Демо-запрос», popup-ы, панель справки, 6 кнопок выгрузки, авто-переключение на пустую подложку без сети. Проверено в
  браузере (playwright), скриншоты — `reports/figures/service_*.png`.
- **Документация**: `reports/service.md`, `service/README.md`, раздел «Сервис» в `README.md`.

## Файлы
`service/{__init__,app,core,build,vectorize,predictors,run}.py`, `service/static/{index.html,app.js,style.css,vendor/}`,
`service/data/{*.geojson.gz,manifest.json}`, `scripts/build_service_data.py`, `tests/test_service.py`,
`reports/service.md`, `reports/figures/service_{demo,demo_pred,full_extent,openapi}.png`, `README.md`, `.gitignore`.

## Как запустить
```powershell
python scripts/build_service_data.py          # необязательно; --source gt|pred|both
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000     # http://localhost:8000 , /docs
python -m pytest tests/test_service.py -q
```

## Тесты
`tests/test_service.py` — 26 тестов, все зелёные: health; /query по всему охвату (непустые слои, атрибуты); вне AOI →
пусто и 0 га; 7 видов невалидных запросов → 422; Shapefile (5 файлов, UTF-8, EPSG:4326, короткие поля, читается
GeoPandas); площадь по bbox+датам чипа = `burn_area_ha` ±1 %; `by_severity` = `area_total_ha`; clip (половины = целое);
загрузка из снимка; старт без данных; синтетика без data/train (clip пополам, фильтр спеклов).
Полный прогон `python -m pytest -q`: **82 passed, 2 skipped, 1 failed** — падает НЕ мой
`tests/test_inference_smoke.py::test_inference_cli_smoke`: общий (junction) каталог `data/test_fake/af` кем-то
наполовину удалён (остался только `aux/`, нет `meta.csv`; изменён в 19:30, до начала моей работы). `data/` я не трогал.
Лечится `python scripts/make_fake_test.py` в основном checkout.

## Не получилось / открытые вопросы
- Слой `pred` построен **заглушками на правилах** (в ветке нет `AFPredictor`/`BSPredictor`): площадь pred 488 тыс. га
  против 236 тыс. га gt — это качество заглушки, не сервиса. После слияния веток моделей выполнить
  `python scripts/build_service_data.py --source both` и проверить `manifest.json: pred_models`.
- ТЗ задачи требовало одновременно «удалять полигоны < 3 px» и «совпадение с burn_area_ha ~1 %» — при обычном sieve это
  несовместимо (до 5.7 % потерь). Выбран фильтр, сохраняющий общую площадь; цена — сдвиг между классами
  (медиана 0.9 %, макс. 4 % площади гари чипа). `--min-pixels 1` даёт точные площади по классам.
- GeoParquet не сделан (в окружении нет pyarrow) — fallback в gzip-GeoJSON.
- Для скриншотов установлен браузер playwright (`npx @playwright/mcp install-browser`, кэш пользователя, вне репозитория).

## Что менять в CLAUDE.md
Добавлен абзац в «Что уже есть»: сервис, команда пересборки, предупреждение про junction `weights/`/`data/`.

## Доработка (правки после ревью)
- `/docs` без интернета: swagger-ui 5.17.14 в `service/static/vendor/swagger/`, `docs_url=None` + свой роут через
  `get_swagger_ui_html`. Проверено в браузере с заблокированными внешними запросами: 9 операций отрисованы, наружу 0 запросов.
- В `service/README.md` и `reports/service.md` явно описаны фильтр мелких фрагментов, `--min-pixels 1` и сверка площадей
  (224 чипа, макс. 0.025 %, вне допуска 1 % — 0).
- Тесты: `test_polygon_query_areas_consistent` (произвольный полигон: by_severity = area_total_ha ±0.01 га, площадь после
  clip ≤ до clip) и `test_docs_served_offline`. `area_total_ha` — округлённая точная сумма контуров; остаток округления классов (≤0.01 га) отдаётся
  крупнейшему классу, поэтому `by_severity` сходится с итогом всегда.
  `tests/test_service.py`: 28 passed.
