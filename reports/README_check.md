# Репетиция README в чистом клоне (2026-09-19)

Чистый локальный клон коммита `3132b7d` (после чистки репозитория: 310 файлов, 266 МБ — только код и документация к коду). Команды
README выполнялись буквально, в PowerShell, с активированным `.venv`; `PYTHONPATH` не задавался, `pip install -e .` не делался.
Машина: Windows 10, i5-13600KF, RTX 4070.

| Шаг README | Команда | Результат |
|---|---|---|
| 1. Код | `git clone <локальный путь> …` | код 0; в корне только `configs eda reports scripts service src submissions tests weights` + файлы проекта и `LICENSE` |
| 2. Окружение | `py -3.12 -m venv .venv` → `.\.venv\Scripts\Activate.ps1` → `python -m pip install --upgrade pip` → `$env:PYTHONIOENCODING = 'utf-8'` | код 0 |
| 3. Установка | `pip install -r requirements.txt` → `python -m compileall -q src inference.py` | код 0, 157 с |
| 4. Проверка установки | `python -c "import torch, lightgbm, rasterio, segmentation_models_pytorch as smp; print(torch.__version__, torch.cuda.is_available())"` | `2.6.0+cu124 True` |
| 5. Данные | `New-Item -ItemType Directory -Force data` → `tar -xf …\fire-test-renamed.tar -C data` → `python scripts/check_data.py --split test` | tar — код 0, 2 с; в `data/test`: `af bs meta.csv sample_submission.csv`; check_data — «ВСЁ ОК» |
| 6. Инференс | `python inference.py --data-dir data/test --output submission.csv` | код 0; **22.53 с** (первый запуск) и **22.34 с** (второй) |
| 7. Проверка файла | `python scripts/check_submission.py submission.csv`; `Get-FileHash submission.csv, submissions\submission_v5_final.csv -Algorithm SHA256` | «Формат: OK (447 строк)»; sha256 равны — `784fe474040914ed…` (побайтно тот же файл, что сдан) |
| Сервис 1. Запуск | `python -m uvicorn service.app:app --host 127.0.0.1 --port 8000` | **выполнено на порту 8061** (см. ниже); `/health` — 200 через 2 с |
| Сервис 2. Адреса | `/`, `/docs` | 200 / 200; страница — финальный интерфейс (панель подложек на месте) |
| Сервис 3. Примеры curl | `curl.exe` summary / burn-scars `format=shp` / `POST /api/v1/query` | summary: `area_total_ha` 1758.92; Shapefile-zip 161 745 байт; query: 98 термоточек, 337 контуров, 1 758.92 га. Слой «Модель» в клоне пуст по замыслу (нужны `data/train` + `build_service_data.py`) |
| Тесты | `python -m pytest -q` | код 0: **134 passed, 36 skipped** за 105 с (в клоне есть `data/test`, поэтому пропусков меньше, чем «130 / 40» без данных); все пропуски — из-за данных: `data/train` (25 строк, 31 тест), `data/test_fake` (4), `baseline/` (1) |
| Состав | `python scripts/check_release.py --no-write` | ссылки из README и отчётов есть в git, файлов ≥ 100 МБ нет, вынесенных каталогов в git нет; единственное замечание в клоне — незакоммиченные результаты самой репетиции (`data/`, `submission.csv`) |

## Отступления и замечания

- **Порт 8000 на машине проверки был занят** посторонним экземпляром сервиса (запущен не в рамках репетиции, не останавливался). Первая
  попытка поэтому попала в чужой процесс и не засчитана; проверка повторена на свободном порту 8061 с контролем, что порт слушает
  именно процесс из клона. В README порт 8000 оставлен; там же указано, что при занятом порте нужно взять другой (`--port 8001`).
- Первый запуск инференса в этом окружении занял 22.5 с, т.е. «холодной» добавки не было; в других свежих окружениях в тот же день
  первый запуск занимал до 42 с (`reports/speed.md`, §1d) — совет README «замеряйте второй запуск» остаётся в силе.
- Команды для Linux (`python3.12 -m venv`, `source .venv/bin/activate`, `sha256sum`, `curl`) отличаются от проверенных только
  синтаксисом оболочки; сам код на Linux проверялся ранее в Docker-образе (CPU).
- Клон после проверки удалён.
