# Мониторинг природных пожаров — КосмоХакатон 2026

Решение кейса «Мониторинг природных пожаров»:

1. **AF (Active Fire)** — по чипу VIIRS 256×256 @375 м: бинарная маска пикселей природного горения.
2. **BS (Burn Severity)** — по чипу 512×512 @20 м (Sentinel-2/1 pre/post + DEM/slope/landcover): маска степени поражения 0–3.
3. **`inference.py`** → `submission.csv` в формате RLE.
4. **Сервис** (FastAPI + Leaflet): термоточки, контуры гарей (GeoJSON/Shapefile), справка по площадям.
5. Отчёт `reports/report.md`, журнал экспериментов `reports/experiments.md`.

Метрика: `Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev` (микро-усреднение по пикселям всех чипов).

> Статус: инфраструктура готова (I/O, RLE, метрика, сабмит, фолды, тесты). Модели AF/BS пока — правила-заглушки.

## Структура

```
inference.py            единственная точка входа инференса
train.py                обучение: python train.py --module af|bs|all (пока заглушка)
configs/                inference.yaml, folds_af.csv, folds_bs.csv (фиксированная валидация)
src/firemon/
  io.py                 чтение чипов → dict массивов (C,H,W) + имена каналов, геопривязка, meta
  rle.py                RLE encode / decode / validate (формат ТЗ)
  metrics.py            F1_af, IoU_burn, mIoU_sev, Score; evaluate_submission
  submission.py         сборка и валидация submission.csv по шаблону
  splits.py             групповые фолды (тайл / пространственный блок) + временной сплит
  models/af, models/bs  модели (сейчас rules.py — заглушки)
  features/, postprocess.py, utils.py
scripts/                check_data.py, make_fake_test.py
service/                веб-сервис (FastAPI + static/)
eda/, reports/          EDA, отчёт, журнал экспериментов, tasklog/
tests/                  pytest
docs/                   ТЗ и критерии (pdf + txt)
data/, baseline/        данные и материалы организаторов (не в git)
```

## Установка (Windows, PowerShell)

Требуется Python 3.10+ (проверено на 3.12.7). GPU нужна только для обучения; инференс работает на CPU.

```powershell
git clone <repo-url> fire-monitoring
cd fire-monitoring
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Без GPU: в `requirements.txt` убрать строку `--extra-index-url …` и суффиксы `+cu124` у `torch`/`torchvision`.
Точный снимок окружения разработки — `requirements.lock.txt`.

Linux/macOS: то же, активация `source .venv/bin/activate`.

## Подготовка данных

```powershell
# train: внутри архива уже лежит каталог train/
New-Item -ItemType Directory -Force data | Out-Null
tar -xf "C:\path\to\fire-train-renamed.tar" -C data
python scripts/check_data.py                 # 420 AF × 3 файла, 224 BS × 6 файлов → «ВСЁ ОК»

# test: внутри архива лежит каталог test/ (вместе с meta.csv и sample_submission.csv)
tar -xf "C:\path\to\fire-test-renamed.tar" -C data
New-Item -ItemType Directory -Force baseline | Out-Null
Copy-Item data\test\sample_submission.csv baseline\sample_submission.csv   # необязательная копия шаблона
python scripts/check_data.py --split test    # 180 AF × 2 файла, 89 BS × 5 файлов, 447 строк шаблона → «ВСЁ ОК»
```

Ожидаемая структура:

```
data/train/af/{meta.csv, viirs/, aux/, masks/}
data/train/bs/{meta.csv, sentinel2_pre/, sentinel2_post/, sentinel1_pre/, sentinel1_post/, aux/, masks/}

data/test/meta.csv                 один общий на оба модуля: chip_id,kind,width,height,gsd,valid_frac,cloud_frac (180 af + 89 bs)
data/test/sample_submission.csv    шаблон: 447 строк chip_id,class_id,rle (rle пустая) — AF по 1 строке, BS по 3
data/test/af/{viirs/, aux/}        AF_te_000001_VIIRS_I1-I5.tif, AF_te_000001_AUX.tif
data/test/bs/{sentinel2_pre/, sentinel2_post/, sentinel1_pre/, sentinel1_post/, aux/}
baseline/sample_submission.csv     копия шаблона (запасное место поиска)
```

Раскладка теста отличается от train: масок нет, `meta.csv` один в корне (а не `af/meta.csv` + `bs/meta.csv`).
Код поддерживает обе: сначала ищется `<data-dir>/<kind>/meta.csv`, иначе `<data-dir>/meta.csv` с фильтром по колонке
`kind`. Если `meta.csv` нет вовсе, список чипов берётся из `sample_submission.csv` / имён файлов. Геопривязки в тестовых
GeoTIFF нет — инференсу она не нужна. Baseline и скрипты RLE/метрики организаторы не выдавали: формат submission
реализован строго по тексту ТЗ (`tests/test_rle_spec.py`).

> **Windows:** `aux` — зарезервированное имя устройства. Файлы внутри `aux/` читаются нормально, но сам каталог
> нельзя открыть в Проводнике/PowerShell или создать обычным `mkdir`. Код использует `firemon.io.win_path()`
> (префикс `\\?\`); вручную с этим каталогом ничего делать не нужно.

## Инференс

```powershell
python inference.py --data-dir data/test --output out/submission.csv
```

- шаблон строк: `--sample-submission`, иначе `<data-dir>/sample_submission.csv`, иначе `baseline/sample_submission.csv`;
- параметры: `--config configs/inference.yaml`;
- без интерактива, код выхода 0, в конце печатается полное время; результат автоматически валидируется.

Проверка файла перед отправкой (валидатор формата + сводка: пиксели по классам, пустые маски) и оценка по известным маскам:

```powershell
python scripts/check_submission.py out/submission.csv      # шаблон и размеры чипов берутся из data/test
python -m firemon.submission --validate out/submission.csv --sample data/test/sample_submission.csv --data-dir data/test
python -m firemon.metrics --submission out/submission_fake.csv --masks-dir data/train --sample-submission data/test_fake/sample_submission.csv
```

Отправленные на лидерборд файлы хранятся в `submissions/` (первый — `submission_stub.csv`, правила-baseline, проверка формата).
Сдвиг распределений test vs train: `python eda/test_vs_train.py` → `reports/test_vs_train.md`.

Сквозная проверка с метрикой (20 AF + 10 BS чипов из train с известными масками, раскладка train):

```powershell
python scripts/make_fake_test.py
python inference.py --data-dir data/test_fake --output out/submission_fake.csv
```

## Валидация (обязательна для всех экспериментов)

Тест разделён по территории и по времени, поэтому используются только зафиксированные фолды:

- `configs/folds_af.csv` — 5 групповых фолдов, группа = тайл `(epsg, x_min, y_min)` (70 тайлов);
- `configs/folds_bs.csv` — 5 групповых фолдов, группа = пространственный блок 50×50 км (один пожар — один фолд);
- колонка `temporal_split`: `val` = 2024 год, `train` = остальные.

```python
from firemon.splits import load_folds
folds = load_folds("af")   # chip_id, fold, temporal_split, group
```

Пересоздание (детерминированно, seed=42): `python -m firemon.splits --data-dir data/train`.
Каждый эксперимент записывается строкой в `reports/experiments.md`.

## Обучение

```powershell
python train.py --module all     # пока заглушка; будет дополнено агентами AF/BS
```

## Сервис

Будет в `service/` (FastAPI + Leaflet). Граница территории мониторинга: `service/data/aoi/fire_monitoring_aoi.geojson`.

## Тесты

```powershell
python -m pytest -q
```

Тесты, которым нужны данные, пропускаются, если `data/train`, `data/test` или `data/test_fake` отсутствуют.
Официальных скриптов RLE/метрики нет — кодек проверяется по правилам ТЗ на ручных примерах (`tests/test_rle_spec.py`).
