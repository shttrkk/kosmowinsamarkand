# Задача 01b — реальный тестовый набор, закрытие темы baseline

Дата: 2026-09-18. Ветка `main`.

## Что сделано

- `fire-test-renamed.tar` распакован в `data/test/` (807 файлов: 180×2 AF + 89×5 BS + `meta.csv` + `sample_submission.csv`;
  пустые каталоги-пустышки архива пропущены). Шаблон скопирован в `baseline/sample_submission.csv` (байт-в-байт).
  `baseline/sample_submission_FAKE.csv` удалён, ссылки на него убраны (`scripts/make_fake_test.py`, CLAUDE.md).
- `firemon.io`: новая `meta_path()`; `read_meta` / `list_chips` / `_meta_row` (→ `read_*_chip`) поддерживают обе раскладки —
  сначала `<data-dir>/<kind>/meta.csv`, иначе `<data-dir>/meta.csv` с фильтром по `kind`. `shapes_from_meta` заработал на тесте
  без правок. `inference.py` менять не пришлось: порядок поиска шаблона уже был нужный, шаблон с `rle` без кавычек читается,
  выход пишет `""`; `chip_kind` по префиксу `AF_te_/BS_te_` работает (тест добавлен).
- `scripts/check_data.py --split test`: комплектность, chip_id meta == шаблон == файлы, 447 строк (269/89/89), AF по 1 и BS по 3
  строки, 5+5 случайных чипов — формы/dtype/band_names против эталона train → «ВСЁ ОК».
- `scripts/check_submission.py <csv>`: `validate_submission` + сводка (пиксели и пустые маски по классам, доля пустых AF).
  `firemon.rle.rle_pixel_count()` — число пикселей без декодирования.
- `eda/test_vs_train.py` → `reports/test_vs_train.md` (таблица сдвига + 5 выводов «наблюдение → решение»), 16 с.
- `reports/baseline_review.md` — раздел отчёта «Baseline и отправная точка»: правила, метрики строки 0, ограничения, H1–H6.
- Тесты: удалены заглушки со skip (`test_official_rle_compat`, `test_official_metric_compat`); новый `tests/test_rle_spec.py`
  (ручные 3×3 и 4×5, каждое правило ТЗ, полный перебор 512 масок 3×3 + 2000 случайных 4×5); `test_io.py` — meta из корня,
  приоритет `<kind>/meta.csv`, реальный тест; `test_submission.py` — шаблон в формате организаторов (rle без кавычек), реальный
  шаблон 447 строк; `test_inference_smoke.py` — полный прогон на реальном тесте (< 30 с, порядок строк как в шаблоне, rle в кавычках).
  Дочерним процессам в smoke-тестах задан `PYTHONIOENCODING=utf-8` (без него тест падал в консоли с cp1251).
- CLAUDE.md (§2, §3, §4 «Тест», §5, «Что уже есть»), README (структура теста, где лежит шаблон, check_submission), `experiments.md`
  (TODO про baseline организаторов заменён), `.gitignore` (`/submissions/*.csv`).

## Как запустить

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts/check_data.py --split test
.\.venv\Scripts\python.exe inference.py --data-dir data/test --output out/submission_stub.csv
.\.venv\Scripts\python.exe scripts/check_submission.py out/submission_stub.csv
.\.venv\Scripts\python.exe eda/test_vs_train.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Результаты

- `python -m pytest -q` — зелёный, 77 тестов, без skip (при наличии `data/train`, `data/test`, `data/test_fake`).
- Инференс заглушки на реальном тесте (269 чипов, CPU): **13.3 с холодный / 10.1 с тёплый запуск**, код 0, валидатор ОК
  (AF 1.9–3.3 с, BS 6.0–7.6 с, сборка + валидация 1.6 с). Запас до порога 30 с — ~17 с на модели.
- `submissions/submission_stub.csv` (10.6 МБ, не в git) — первая отправка для проверки формата. Сводка: AF 126 992 px,
  48 из 180 чипов с пустой маской (26.7 %); BS 3.60 / 1.62 / 1.01 млн px по классам 1/2/3, пустых масок 0 / 0 / 4.

## Чем тест отличается от train

1. Один `meta.csv` в корне (7 колонок) вместо `af/meta.csv` + `bs/meta.csv`; `sample_submission.csv` лежит в корне теста.
2. В шаблоне `rle` пустая БЕЗ кавычек (`AF_te_000001,1,`).
3. **Геопривязка из GeoTIFF вырезана**: `crs=None`, transform единичный, теги пустые (в train — EPSG:32637/38 и реальный transform).
4. Масок нет; формы, dtype, band descriptions идентичны train. Те же пустые каталоги-пустышки в архиве.

## Сдвиг test vs train (5 строк; полностью — `reports/test_vs_train.md`)

1. AF, день: фон теплее — I4 p99 +6.8 K, p99.9 +6.4 K; правило `I4>320 K` даёт 1141 px на дневной чип против 214 в train (×5.3) → абсолютные пороги непереносимы, только контекстные признаки.
2. AF: ночных чипов 38.3 % против 30.2 % → раздельная логика день/ночь, вес ночных чипов в F1 на тесте выше.
3. Покров: пашни меньше (AF −7.1 п.п., BS −12.1), травы (+7.4 / +9.4) и болота (+4.1 / +3.8) больше → landcover как признак, контроль метрик по покрову.
4. BS: облачных чипов (cloud_frac > 0.25) 25.8 % против 10.7 %, valid_frac 0.879 против 0.919 → SCL-политика и S1 важнее, чем покажет train-валидация.
5. BS: хвост dNBR тяжелее (p95 0.42 против 0.35; dNBR > 0.10 у 27.0 % пикселей против 22.2 %) → глобальные пороги дадут ещё больше ложной гари.

## Расхождения с CLAUDE.md (исправлено в CLAUDE.md, кроме п. 4)

1. §4: «data/test — та же структура» — неверно: общий `meta.csv` и шаблон в корне. Исправлено, раскладка описана.
2. §3/§5: сверка с официальными скриптами RLE/метрики невозможна — их не будет. Заменено на «строго по тексту ТЗ, проверяется первой отправкой».
3. Не было сказано, что в тестовых GeoTIFF нет геопривязки — добавлено (важно для сервиса: демо-слои строить только по train).
4. §4 AF: «14 чипов с valid_frac<1» верно по meta, но вводит в заблуждение: NaN в I4/I5 есть примерно в 55 % чипов (в среднем 8.3 % пикселей
   train, 9.4 % test) при `valid=1` и `valid_frac=1.0`. Добавлена заметка в «Что уже есть»; сама статистика §4 не переписывалась —
   природу этих NaN (bow-tie/маска облаков?) стоит выяснить агенту EDA.
5. Строка 0 `experiments.md` оценивала время на тесте «≈9 с» — фактически 10–13 с.

## Открытые вопросы

- Формат (кавычки `""`, порядок строк, RLE) подтвердится только первой отправкой `submissions/submission_stub.csv` на лидерборд.
- Поведение метрики организаторов на nodata=255 неизвестно; оставлено `ignore_nodata=True`.
- В других worktree (AF/BS/EDA/сервис) после merge `main`: `baseline/sample_submission_FAKE.csv` больше не создаётся; `data/` и `baseline/`
  gitignored — каждому worktree нужен доступ к `data/test` (распаковать или использовать общий путь).
