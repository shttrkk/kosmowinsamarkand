# Задача 05 — inference.py, воспроизводимость, Docker, скорость

Дата: 2026-09-18. Ветка `main`. Состояние моделей: AF — LightGBM (AF-17), **BS — заглушка на правилах** (BSPredictor обучается
в `task/bs`, ветка не тронута). Пункты промпта про baseline организаторов пропущены — его не существует.

## Что сделано

1. **`inference.py` → тонкий CLI**, вся работа с моделями — в новом `src/firemon/pipeline.py`:
   - реестры `AF_BACKENDS` / `BS_BACKENDS`; бэкенд = функция `(data_dir, chip_ids, cfg, ctx) -> (preds, n_failed)`; выбор и запасной
     вариант — `configs/inference.yaml` (`model`, `fallback`). **Подключение BSPredictor = одна запись в `BS_BACKENDS` + веса в
     `weights/` + `bs.model` в конфиге**; `inference.py`, Dockerfile и README при этом не меняются (проверено тестом
     `test_new_bs_backend_plugs_in_via_registry`);
   - CLI: `--data-dir --output [--sample-submission --config --device auto|cpu|cuda --workers N --profile]`; коды выхода 0 / 1 (файл не
     прошёл валидацию) / 2 (нет данных/конфига/ни одного чипа из шаблона — понятное сообщение без трассировки);
   - состав строк — строго по шаблону; отсутствующий или битый чип → пустая маска + warning/error, не падение;
   - чтение и признаки AF **и BS** в пуле потоков; `--profile` печатает таблицу этапов.
2. **Скорость:** 18.5–19.7 с → **~10–11 с** на CPU (269 чипов). Потоки против процессов измерены
   (`scripts/bench_inference_parallel.py`): 3.40 с против 3.60 с на 8 воркерах → оставлены потоки. Всё — в `reports/speed.md`
   (конфигурации, профиль, отклонённые приёмы, детерминизм); таблица готова к дополнению строками BS-модели. Бюджет BS на CPU ≈ 16–19 с.
3. **Детерминизм:** submission побайтно одинаков при `--workers` 1/2/4/8/16, потоках LightGBM 1/8, в venv cu124, чистом venv CPU и в
   Docker; совпадает с отправленным `submission_af.csv`. `train.py --module af` повторно: OOF 0.9682 / temporal 0.9597, веса совпадают
   по содержимому (`git diff` пуст; различие только CRLF/LF — на Windows LightGBM пишет CRLF).
4. **Docker:** Docker Desktop установлен, демон был выключен — запущен, образ **собран и реально прогнан на `data/test`**: код 0,
   10.8 с внутри / 11.1–11.5 с с запуском контейнера, файл побайтно как на Windows. Первый прогон упал (`libexpat.so.1` нет в
   `python:3.12-slim`, нужен колесу rasterio) → добавлен `libexpat1`. Образ 2.79 ГБ (torch CPU включён заранее — под BS-модель).
   `Dockerfile.gpu` написан, **не тестировался** (честно отмечено в README).
5. **Зависимости:** `requirements-base.txt` (без torch) ← `requirements.txt` (cu124) и `requirements-cpu.txt` (CPU).
6. **README** переписан по 14 разделам промпта (PowerShell + bash), схема `reports/figures/pipeline.png`
   (`scripts/make_pipeline_figure.py`). **Команды README выполнены буквально в чистом venv** из `requirements-cpu.txt` без
   `PYTHONPATH`: check_data, inference (тот же хэш), check_submission, fake-тест + метрика, pytest (129 passed), uvicorn `/health` = 200.
   Проверено, что штатный `tar` Windows распаковывает каталог `aux`.
7. Раздел **«Разработка в нескольких worktree»** — в README и CLAUDE.md (общий `.venv` без `pip install -e`, `PYTHONPATH` на свой
   `src`, junction-ы `data`/`baseline`/`weights`, предупреждение про `Remove-Item -Recurse`).
8. Мелочи: CLAUDE.md §6 — строка про центр чипа (B4c); `reports/submissions.md` — две отправки (0.3117 и 0.6035) и вывод про
   F1_af ≈ 0.93–0.99; чистка: 6 неиспользуемых импортов (ruff F401), `print("→ …")` в двух скриптах падал в консоли cp1251 — заменён.

## Файлы

Новые: `src/firemon/pipeline.py`, `Dockerfile`, `Dockerfile.gpu`, `.dockerignore`, `requirements-base.txt`, `requirements-cpu.txt`,
`tests/test_pipeline.py`, `tests/test_reproducibility.py`, `scripts/bench_inference_parallel.py`, `scripts/make_pipeline_figure.py`,
`reports/speed.md`, `reports/submissions.md`, `reports/figures/pipeline.png`, этот файл.
Изменены: `inference.py`, `configs/inference.yaml`, `src/firemon/submission.py` (`submission_rows`), `requirements.txt`, `README.md`,
`CLAUDE.md`, `tests/test_inference_smoke.py` (тест детерминизма переехал), удаление неиспользуемых импортов в 5 файлах.

## Как запустить

```powershell
python inference.py --data-dir data/test --output out/submission.csv --profile
python scripts/check_submission.py out/submission.csv
docker build -t firemon .
docker run --rm -v C:\path\to\test:/data:ro -v C:\path\to\out:/out firemon --data-dir /data --output /out/submission.csv
python -m pytest -q
```

## Тесты

`python -m pytest -q` — **129 passed, 0 skipped** (основной venv и чистый venv CPU). Новые: 10 тестов устойчивости/реестра
(`test_pipeline.py`), 4 теста воспроизводимости (`test_reproducibility.py`, включая реальный тест).

## Что не сделано / открытые вопросы

- Строки GPU и BS-модели в `reports/speed.md` и README — после слияния `task/bs`. Тогда же: бэкенд в `BS_BACKENDS`, проверка
  детерминизма UNet, `submissions/final_submission.csv`, пересборка Docker-образа (слой зависимостей закэширован — секунды).
- «Холодный» старт (данные не в кэше ОС) без перезагрузки не воспроизвести; оценка +~3 с по первому запуску после распаковки.
- Замеры шли на фоне обучения BS в соседнем worktree (разброс ±1.5 с); на свободной машине будет не хуже.
- `ruff`/`black` целиком не прогонялись: в проекте нет конфигурации линтера, а у скриптов EDA/AF свой стиль (`;` в строке, длинные
  строки) — массовое переформатирование чужого кода перед слиянием `task/bs` дало бы конфликты. Исправлены только содержательные
  находки (F401). В ядре (`src`, `inference.py`, `tests`, `service`) осталось 13 стилистических замечаний.
- Раздел README «Команда» содержит только Марка — остальных участников нужно вписать.
- Docker Desktop оставлен запущенным (был выключен до задачи).

## Что поменять в CLAUDE.md

Внесено: `pipeline.py` в структуре, раздел «Инференс, Docker, скорость (задача 05)», раздел «Разработка в нескольких worktree».
Для агента BS: в ветке `task/bs` изменён старый `inference.py` — при слиянии брать версию `main`, а вызов
`run_bs_inference(...)` обернуть бэкендом в `src/firemon/pipeline.py`.
