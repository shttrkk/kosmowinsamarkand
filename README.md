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

# test (когда выдан): распаковать в data/test, sample_submission.csv — в baseline/ или data/test/
python scripts/check_data.py --split test    # 269 чипов, 447 строк шаблона
```

Ожидаемая структура:

```
data/train/af/{meta.csv, viirs/, aux/, masks/}
data/train/bs/{meta.csv, sentinel2_pre/, sentinel2_post/, sentinel1_pre/, sentinel1_post/, aux/, masks/}
data/test/…            то же без masks/
baseline/sample_submission.csv
```

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

Проверка файла и оценка по известным маскам:

```powershell
python -m firemon.submission --validate out/submission.csv --sample baseline/sample_submission.csv --data-dir data/test
python -m firemon.metrics --submission out/submission_fake.csv --masks-dir data/train --sample-submission data/test_fake/sample_submission.csv
```

Сквозная проверка без настоящего теста (20 AF + 10 BS чипов из train в тестовой структуре):

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

Веб-карта + REST API (FastAPI + Leaflet): по полигону/bbox и интервалу дат — термоточки, контуры гарей
(GeoJSON/Shapefile, атрибуты id/класс/площадь) и справка по площадям (JSON/CSV). Подробности — `reports/service.md`.

```powershell
# зависимости те же (requirements.txt). Сервис стартует сразу после клонирования — из снимка эталонных
# слоёв service/data/*.geojson.gz. Полная пересборка (эталон + предсказания моделей, нужен data/train, ~2 мин):
python scripts/build_service_data.py            # --source gt|pred|both (по умолчанию both)

python -m uvicorn service.app:app --host 0.0.0.0 --port 8000      # или: python service/run.py
```

Открыть: http://localhost:8000 — карта (кнопка «Демо-запрос»), http://localhost:8000/docs — OpenAPI.

```bash
# 1) справка по площадям (JSON; format=csv — таблица)
curl "http://localhost:8000/api/v1/summary?bbox=39.25,47.95,39.50,48.10&date_from=2024-09-15&date_to=2024-10-10"
# 2) контуры гарей в Shapefile (zip: shp+shx+dbf+prj+cpg)
curl -o burn_scars_shp.zip "http://localhost:8000/api/v1/burn-scars?bbox=39.25,47.95,39.50,48.10&format=shp"
# 3) всё одним запросом: термоточки + гари + справка (область — bbox или GeoJSON-полигон в поле geometry)
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json"      -d '{"bbox":[39.25,47.95,39.50,48.10],"date_from":"2024-09-15","date_to":"2024-10-10","source":"gt"}'
```
В PowerShell вместо `curl` — `curl.exe` (или `Invoke-RestMethod`).

## Тесты

```powershell
python -m pytest -q
```

Тесты, которым нужны данные, пропускаются, если `data/train` или `data/test_fake` отсутствуют.
Тесты совместимости с официальными RLE/метрикой пропускаются, пока нет `baseline/`.
