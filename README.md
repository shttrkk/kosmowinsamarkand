# Мониторинг природных пожаров — КосмоХакатон 2026

Решение кейса «Мониторинг природных пожаров»:

1. **AF (Active Fire)** — по чипу VIIRS 256×256 @375 м: бинарная маска пикселей природного горения.
2. **BS (Burn Severity)** — по чипу 512×512 @20 м (Sentinel-2/1 pre/post + DEM/slope/landcover): маска степени поражения 0–3.
3. **`inference.py`** → `submission.csv` в формате RLE.
4. **Сервис** (FastAPI + Leaflet): термоточки, контуры гарей (GeoJSON/Shapefile), справка по площадям.
5. Отчёт `reports/report.md`, журнал экспериментов `reports/experiments.md`.

Метрика: `Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev` (микро-усреднение по пикселям всех чипов).

> Статус: инфраструктура готова (I/O, RLE, метрика, сабмит, фолды, тесты). AF — LightGBM (OOF F1 0.9686, temporal 0.9560, см. `reports/af_module.md`). BS пока — правила-заглушки.

## Структура

```
inference.py            единственная точка входа инференса
train.py                обучение: python train.py --module af|bs|all (af — LightGBM; bs — пока заглушка)
configs/                inference.yaml, af_lgbm.yaml, folds_af.csv, folds_bs.csv (фиксированная валидация)
src/firemon/
  io.py                 чтение чипов → dict массивов (C,H,W) + имена каналов, геопривязка, meta
  rle.py                RLE encode / decode / validate (формат ТЗ)
  metrics.py            F1_af, IoU_burn, mIoU_sev, Score; evaluate_submission
  submission.py         сборка и валидация submission.csv по шаблону
  splits.py             групповые фолды (тайл / пространственный блок) + временной сплит
  models/af             AF: predictor.py (AFPredictor), pipeline.py / train_lgbm.py / dataset.py (обучение), rules.py (baseline-правило)
  models/bs             BS (сейчас rules.py — заглушка)
  features/, postprocess.py, utils.py
scripts/                check_data.py, make_fake_test.py, af/ (grid правила, абляции, постобработка, профиль, рисунки)
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
python train.py --module af      # AF: признаки → data/cache/af_train.npz → 5 фолдов LightGBM → weights/af_lgbm_fold{0..4}.txt + af_lgbm_meta.json
python train.py --module all     # bs — пока заглушка
```

AF (~40 с на 8 потоках): печатает OOF F1 (GroupKFold по тайлу), F1 на временном сплите (val = 2024), срезы, пишет
`reports/af_train_report.json` и строку в `reports/experiments.md` (`--no-log` — не писать; `--rebuild-cache` — пересчитать
кэш признаков). Детерминировано: повторный запуск из чистого состояния даёт OOF F1 0.9686 / temporal 0.9560 и побитово
те же веса. Исследовательские скрипты (не нужны для воспроизведения сабмита):

```powershell
python scripts/af/step0_rule_grid.py      # шаг 0: пороги контекстного правила по фолдам
python scripts/af/experiments.py          # варианты LightGBM и абляции → reports/af_experiments.json
python scripts/af/postprocess_eval.py     # гистерезис / одиночные / вода
python scripts/af/ensemble_vs_single.py   # 5 фолд-моделей vs одна модель
python scripts/af/profile_inference.py --data-dir data/test_fake
python scripts/af/make_report_figures.py  # reports/figures/af_*.png
```

## Сервис

Будет в `service/` (FastAPI + Leaflet). Граница территории мониторинга: `service/data/aoi/fire_monitoring_aoi.geojson`.

## Тесты

```powershell
python -m pytest -q
```

Тесты, которым нужны данные, пропускаются, если `data/train` или `data/test_fake` отсутствуют.
Тесты совместимости с официальными RLE/метрикой пропускаются, пока нет `baseline/`.
