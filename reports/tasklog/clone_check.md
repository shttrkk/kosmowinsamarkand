# Репетиция чистого клона (2026-09-19)

`git clone C:\Users\User\hack\fire-monitoring C:\Users\User\hack\clone_check` — в клоне только то, что в git (372 файла, 266 МиБ):
нет `data/`, `baseline/`, `out/`, `submissions/`, `service/data/firemon.gpkg`, весов экспериментов. Команды — буквально из README;
`PYTHONPATH` не задан, `pip install -e .` не делался. Данные теста — по пути основного репозитория (`--data-dir …\fire-monitoring\data\test`).

## Проход 1 — коммит `90d714e` (до правок)

| Шаг README | Результат |
|---|---|
| `py -3.12 -m venv .venv` → `python -m pip install --upgrade pip` → `pip install -r requirements.txt` | код 0, 168 с (колёса частично в кэше pip) |
| `python inference.py --data-dir <data/test> --output out/sub.csv` | код 0; **30.0 с — первый, «холодный» запуск в новом окружении** (компиляция `.pyc`, первый импорт torch/CUDA); sha256 `784fe474040914ed…` = `submission_v5_final.csv` |
| `python -m uvicorn service.app:app --port 8021` | `/health` 200 через 2 с; `/` 200 (есть `/static/app.js`); `app.js`, `style.css`, `vendor/leaflet`, `vendor/polygon-clipping`, `/docs` — 200; демо-запрос `source=gt`: **98 термоточек, 337 контуров, 1 758.92 га** — без `data/train`; `source=pred` — 0 объектов (слой «модель» в git не входит; в интерфейсе переключатель при этом заблокирован) |
| `python -m pytest -q` | **код 1: 1 failed** — `test_pipeline.py::test_missing_inputs_exit_code_and_message[…не найдено ни одного чипа]`: тест неявно рассчитывал, что шаблон найдётся в `baseline/`, которого в клоне нет. Пять тестов `test_submission_example.py` пропускались по той же причине |

## Что исправлено в основном репозитории

1. `tests/test_pipeline.py` — шаблон передаётся явно (`--sample-submission`), добавлен случай «нет ни шаблона, ни чипов» (проверяется
   именно в чистом клоне; рядом с `baseline/` пропускается с пояснением).
2. `tests/fixtures/example_submission.csv`, `sample_submission.csv` — копии файлов организаторов **в git, байт-в-байт**
   (`.gitattributes`: `tests/fixtures/*.csv -text`, иначе `eol=lf` превратил бы CRLF примера в LF и побайтный тест упал бы в клоне);
   `test_submission_example.py` читает их; рядом с `baseline/` дополнительно сверяет копии с оригиналами.
3. `tests/test_service_snapshot.py` (3 теста) — сервис на снимке `service/data/*.geojson.gz` из git: `/health`, страница и статика,
   демо-запрос (98 / 337 / 1 758.92 га), выгрузки Shapefile и CSV. Остальные тесты сервиса пересобирают слои из `data/train` и в клоне пропускаются.
4. `tests/conftest.py` — сообщения `pytest.skip` ссылаются на существующий раздел README («5. Структура входных данных») и говорят, что данные выдают организаторы.
5. `.gitignore` — исключение `!handoff/demo_exports/*.zip` (Shapefile-выгрузки плана В), `.venv_*/`, исправлен нечитаемый комментарий.
6. README — первый экран «5 команд», таблица «каким тестам нужны данные», состав репозитория с размерами.

Пункт (в) задания: в коде нет путей вне репозитория, ссылок на worktree (`wt-*`) и `..\`; скрипты экспериментов (`scripts/bs_night_*`,
`bs_ens4_check.py`, `bs_repro_compare.py`, `configs/inference_bs_repro.yaml`) читают локальные каталоги весов экспериментов
(`weights_repro/`, `weights_night_*/`, `data/cache/…`) — они внутри репозитория, в git не входят и для инференса не нужны (описано в README §12).

## Проход 2 — коммит `9660bd1` (`git pull` в клоне)

| Шаг | Результат |
|---|---|
| `python -m pytest -q` | **код 0: 128 passed, 40 skipped** из 168 за 34 с. Пропуски — только из-за данных: 31 × «нет data/train» (в том числе параметризованный тест сервиса ×7), 4 × «нет data/test», 4 × «нет data/test_fake», 1 × «нет baseline/ — сверять фикстуры не с чем» |
| `python inference.py --data-dir <data/test>\ --output out/sub2.csv` (путь с завершающим слэшем) | код 0, **24.4 с**, sha256 `784fe474040914ed…` — побайтно равен финалу |
| `python scripts/check_release.py --no-write` | обязательные 30 файлов в git; не найдены только два журнала, которые на тот момент ещё не были записаны (этот файл и `release_check.md`) |
| в основном репозитории (с данными) | `python -m pytest -q` — код 0, 168 тестов |

## Вывод

Из чистого клона решение проходит путь README без ручных действий: установка → инференс (файл побайтно равен отправленному) → сервис
(эталонный слой, демо-запрос) → тесты. Замечание для критерия скорости: **первый запуск в свежем окружении — 30.0 с**, повторные —
23–24 с; если замер организаторов делается один раз сразу после установки, результат окажется на границе порога 30 с (см. «Риски»).
`clone_check` после проверки удалён.
