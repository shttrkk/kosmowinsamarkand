# Слияние 1: task/af → main, task/service → main

Дата: 2026-09-18. Исходный `main` — `c66d481` (задача 1b). Общий предок обеих веток — `ba6f059`.

| Шаг | Ветка (коммит) | Коммит слияния | pytest | inference на data/test |
|---|---|---|---|---|
| 1 | `task/af` (`4033548`) | `43937aa` | 88 passed, 0 skipped | код 0, 18.5 / 19.7 с (два прогона) |
| 2 | `task/service` (`3f07919`) | `3e243ae` | 116 passed, 0 skipped; `tests/test_service.py` — 28 passed | код 0, 18.6 с |

Время инференса (CPU, 269 чипов): AF LightGBM 8.2–9.3 с (было 1.9–3.3 с у заглушки), BS-заглушка 6.5–7.3 с,
сборка + валидация 1.9–2.1 с. До порога 30 с остаётся ~11 с — это бюджет BS-модели (сейчас BS — правила).

`scripts/check_submission.py out/submission_af.csv` — формат OK, 447 строк. AF: 3 799 px (у заглушки было 126 992),
44 из 180 чипов с пустой маской (24.4 %); BS без изменений: 3.60 / 1.62 / 1.01 млн px, пустых масок 0 / 0 / 4.
Копия: `submissions/submission_af.csv` (10.4 МБ, не в git — `/submissions/*.csv` в `.gitignore`).

## Перед слиянием

`git status` в `main` был не чистым: неотслеживаемые `weights/af_lgbm_fold{0..4}.txt` + `af_lgbm_meta.json` (копии, оставленные
AF-агентом). Хэши blob'ов совпали с `task/af` байт-в-байт → копии удалены, после чего статус чистый; иначе git отказался бы
сливать («untracked files would be overwritten»). После слияния 7 файлов `weights/` (4.9 МБ) отслеживаются, добавлены
коммитом `587f34a`, который принадлежит `task/af`. `baseline/sample_submission_FAKE.csv` на диске отсутствует
(в `baseline/` только `sample_submission.csv`).

## Конфликты и решения

### task/af
| Файл | Конфликт | Решение |
|---|---|---|
| `CLAUDE.md` | «Что уже есть»: main переписал пункт про фейковый тест и добавил два пункта (реальный тест, NaN в I4); AF изменил тот же пункт и добавил раздел «Модуль AF» | Объединено: три пункта main + раздел «Модуль AF» целиком. Выброшены два устаревших пункта AF: про `sample_submission_FAKE.csv` в `<out-dir>` и «`read_meta` не находит meta.csv в корне теста» — решено в main (задача 1b) |
| `scripts/make_fake_test.py` | main удалил создание FAKE-шаблона, AF перенёс его из `baseline/` в `--out-dir` | Версия main: шаблон не создаётся нигде |
| `tests/test_af.py` (без git-конфликта, но логический) | `test_make_fake_test_does_not_write_to_baseline` требовал `sample_submission_FAKE.csv` в `--out-dir` | Тест переписан: FAKE-файла нет ни в `--out-dir`, ни в `baseline/`; список файлов `baseline/` до и после запуска одинаков |
| `reports/experiments.md` | слилось автоматически | Проверено: строка 0, затем AF-0…AF-17, затем примечание main про отсутствие baseline организаторов |
| `README.md`, `inference.py`, `configs/inference.yaml`, `train.py`, `rules.py`, `postprocess.py` | слились автоматически | Проверено: порядок поиска шаблона в `inference.py` сохранён, разделы README обеих веток на месте |
| `tests/test_rle.py`, `tests/test_metrics.py` | в `task/af` оставались два skip-теста совместимости с официальными скриптами | AF эти файлы не менял → git взял версию main, skip-тестов нет (`pytest -rs`: 0 skipped) |

### task/service
| Файл | Конфликт | Решение |
|---|---|---|
| `CLAUDE.md` | то же место: сервис добавил пункт «Сервис (задача 06)» и нёс старый пункт про FAKE-шаблон | Пункт сервиса оформлен отдельным подразделом «Сервис (задача 06)» после «Модуль AF»; пункт про FAKE — версия main |
| `.gitignore` | автоматически | main: `/submissions/*.csv`; сервис: `.playwright-mcp/` — обе строки на месте |
| `README.md` | автоматически | раздел «Сервис» из ветки + разделы main/AF |
| `service/static/.gitkeep` | удалён в ветке | принято |

## Проверка сервиса

`python -m uvicorn service.app:app --port 8765`: `/` = 200, `/docs` = 200, но **`/health` = 404** — в ветке маршрут только
`/api/v1/health`. Добавлен алиас `/health` (тот же обработчик, вне схемы OpenAPI) + проверка в `test_service.py` — отдельным
коммитом после слияния. Повторно: `/health` = 200 с третьей попытки (~3 с старта),
`{"status":"ok","n_hotspots":9725,"n_burn_scars":43674,"n_chips":644}`; сервер остановлен, порт свободен.

## Что осталось / замечания

1. Слой `pred` в снимке сервиса построен заглушками (`manifest.pred_models: af = rules-stub, bs = rules-stub`). `AFPredictor` теперь
   в main → перезапустить `python scripts/build_service_data.py --source both` (и ещё раз после BS-модели).
2. Снимок `service/data/burn_scars.geojson.gz` — 15.7 МБ в git (лимит 50 МБ соблюдён).
3. Ветки `task/af`, `task/service` и их worktree не тронуты; `task/bs`, `task/eda` не сливались. Им стоит подтянуть `main`
   (корневой meta.csv, отсутствие FAKE-шаблона, удалённые skip-тесты), чтобы следующий merge прошёл без тех же конфликтов.
4. Строка 0 `experiments.md` по-прежнему говорит «≈9 с на 269 тестовых» для заглушки; фактические замеры — в tasklog 01b и здесь.
