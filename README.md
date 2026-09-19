# Мониторинг природных пожаров — КосмоХакатон 2026

Команда **SAMARKAND**. Решение кейса: два модуля (активное горение VIIRS, степень поражения Sentinel-2/1), `submission.csv`, сервис, отчёт.

## Быстрый старт: от клона до `submission.csv` за 5 команд

**Linux + GPU** (Python 3.12, драйвер NVIDIA ≥ 550; без GPU — `requirements-cpu.txt`, результат тот же):

```bash
git clone <repo-url> fire-monitoring && cd fire-monitoring
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python inference.py --data-dir /path/to/test --output /path/to/submission.csv
```

**Windows** (PowerShell):

```powershell
git clone <repo-url> fire-monitoring; cd fire-monitoring
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python inference.py --data-dir C:\path\to\test --output C:\path\to\submission.csv
```

Что важно знать до запуска:

- **`/path/to/test`** — распакованный тестовый архив организаторов (`fire-test-renamed.tar` → каталог `test/` с `meta.csv`,
  `sample_submission.csv`, `af/`, `bs/`; раздел 5). Данные в репозиторий не входят. Каталог **`baseline/` для инференса не нужен**:
  шаблон строк берётся из `<data-dir>/sample_submission.csv` (или `--sample-submission`).
- **Веса уже в репозитории** (`weights/`, 205 МБ, раздел 8) — скачивать ничего не нужно, **сеть при инференсе не используется**.
- **Ожидаемое время** (269 чипов, от старта процесса до провалидированного файла): RTX 4070 — **≈ 23.5 с**; только CPU (i5-13600KF) —
  **≈ 57 с**; Docker CPU — ≈ 69 с; H100 — оценка ≈ 19–22 с (не проверялось: GPU-часть ≈ 6.5 с из 23.5, остальное — CPU-этапы).
  Нет GPU или старый драйвер → автоматический переход на CPU, код выхода 0. Быстрый режим v4 (−0.005 Score, ≈ 21 с): `--config configs/inference_bs_base.yaml`.
- **Результат детерминирован**: GPU, CPU, Docker, любое `--workers` — побайтно один файл (sha256 финального `784fe474…`), формат — как
  `example_submission.csv` организаторов. Код выхода: 0 — файл записан и прошёл валидацию; 1 — не прошёл; 2 — нет входных данных.
- Python 3.10 / 3.11 → `pip install -r requirements-py310.txt` (тот же файл на выходе); другой Python или нет желания ставить пакеты → Docker (раздел 7).
- **Сервис**: `python -m uvicorn service.app:app --port 8000` → http://localhost:8000 (раздел 10). Слой эталона работает сразу после
  клонирования; для слоя «модель» нужны `data/train` и `python scripts/build_service_data.py --source both`. Подложки «Спутник» (Esri) и
  «Карта» (OSM) требуют интернета; без него интерфейс сам переходит в офлайн-режим «Нет» (граница территории + охваты сцен), API и `/docs` работают офлайн.
- Обучение: `python train.py --module af|bs|all` (раздел 9). Тесты: `python -m pytest -q` (раздел 11). Отчёт — `reports/report.pdf`,
  презентация — `presentation/deck.pdf`, сверка с ТЗ — `reports/tasklog/tz_compliance.md`, проверка релиза — `reports/tasklog/release_check.md`.

## 1. Кратко о решении

Решение кейса «Мониторинг природных пожаров» из пяти частей:

1. **AF (Active Fire)** — по чипу VIIRS 256×256 @375 м: бинарная маска пикселей природного горения. Пиксели-кандидаты
   (раздельные пороги день/ночь) → 110 признаков (контекст окон, z-score, признаки блика, landcover, метео) →
   среднее 5 фолд-моделей LightGBM → порог по OOF.
2. **BS (Burn Severity)** — по чипу 512×512 @20 м (Sentinel-2/1 pre/post + DEM/slope/landcover): маска степени поражения 0–3.
   Три UNet-resnet34 (v1, v4 и v4, обученный на масштабе 0.75×) выбирают регион «своего» пожара → LightGBM 2-го уровня
   (с признаками от UNet 0.5×) уточняет пиксели и степень (медианный dNBR, landcover, SCL, признаки локализации события) →
   пороги → постобработка (`BSPredictor`, бэкенд `unet`, вариант `ms` = v5; быстрый режим `base` = v4).
3. **`inference.py`** → `submission.csv` в формате RLE (как в примере организаторов), одна команда, ≈ 23.5 с на GPU / ≈ 57 с на CPU,
   GPU, CPU и Docker дают побайтно один файл.
4. **Сервис** (FastAPI + Leaflet): термоточки, контуры гарей (GeoJSON/Shapefile), справка по площадям в га.
5. Отчёты в `reports/` (EDA, baseline, эксперименты, скорость, отправки).

![Схема пайплайна](reports/figures/pipeline.png)

Метрика: `Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev` (микро-усреднение по пикселям всех чипов).
Схема перерисовывается командой `python scripts/make_pipeline_figure.py`.

## 2. Результаты

| Модуль | Модель | OOF (GroupKFold-5) | Временной сплит (val = 2024) | Примечание |
|---|---|---|---|---|
| AF | правило-baseline `I4>320 K & I4−I5>15 K` | F1 0.2435 (весь train) | — | строка 0 `reports/experiments.md` |
| AF | **LightGBM по кандидатам (AF-17)** | **F1 0.9682** | **F1 0.9597** | `reports/af_module.md` |
| BS | правило-baseline: dNBR 0.10/0.27/0.44 | IoU_burn 0.3706, mIoU_sev 0.3129 | IoU_burn 0.3571, mIoU_sev 0.2967 | запасной бэкенд `rules` |
| BS | v4 (предыдущий финал, быстрый режим `bs.variant: base`): 2×UNet-resnet34 → стек-LGBM + постобработка (без TTA) | IoU_burn 0.7560, mIoU_sev 0.7554 | IoU_burn 0.7723, mIoU_sev 0.7684 | `reports/bs_module.md` |
| BS | **v5 (финал, `bs.variant: ms`): многомасштабный ансамбль — UNet v1, v4 и v4@0.75× → среднее → стек-LGBM с признаками UNet 0.5×, ворота 0.01 → пороги → постобработка** | **IoU_burn 0.7645, mIoU_sev 0.7632** | **IoU_burn 0.7904, mIoU_sev 0.7831** | `reports/bs_module.md` §4.11 |

**Score по шагам** (`0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev`):

| Шаг (отправка) | OOF (GroupKFold-5) | Временной сплит 2024 | Public LB |
|---|---|---|---|
| оба baseline-правила (`submission_stub.csv`) | 0.3088 | — | 0.3117 |
| + AF LightGBM (`submission_af.csv`) | 0.5625 | 0.5499 | 0.6035 → F1_af на public ≈ 0.93–0.99 |
| + BS 2×UNet → стек-LGBM, вариант с TTA и fp16 (`submission_v3_af_bs.csv`) | 0.8344 | 0.8392 | 0.8267 |
| + BS v4: без TTA, fp32 — результат не зависит от устройства (`submission_v4_final.csv`, предыдущий финал) | 0.8300 | 0.8367 | 0.8262 |
| **+ BS v5, финал: многомасштабный ансамбль (`submission_v5_final.csv`)** | **0.8353** | **0.8478** | **0.8314** |

Public финала на 0.004–0.016 ниже локальных оценок — переобучения нет (в тесте облачных BS-чипов 25.8 % против 10.7 % в train); Δ −0.0005 между
отправками №3 и №4 — цена детерминизма CPU/GPU (отказ от TTA и fp16); v5 даёт +0.0052 на public к v4 ценой +2.5 с на GPU.

| Режим | Конфигурация | GPU (RTX 4070) | CPU | Score OOF / 2024 / public |
|---|---|---|---|---|
| **v5, по умолчанию** | `configs/inference.yaml` (`bs.variant: ms`) | ≈ 23.5 с | ≈ 57 с | 0.8353 / 0.8478 / 0.8314 |
| v4, быстрый режим | `--config configs/inference_bs_base.yaml` (`bs.variant: base`) | ≈ 21 с | ≈ 52 с | 0.8300 / 0.8367 / 0.8262 (−0.005 Score) |

Единый источник всех чисел — `reports/final_numbers.md`; полный отчёт — `reports/report.md` (`reports/report.pdf`),
презентация — `presentation/deck.pptx`, пакет для защиты — `handoff/HANDOFF.md`.

Подробно: `reports/submissions.md`, `reports/experiments.md`, `reports/baseline_review.md`, `reports/eda.md`.

## 3. Требования к окружению

- ОС: Windows 10/11 или Linux x86_64 (glibc ≥ 2.28: Debian 11+, Ubuntu 20.04+). Разработка велась на Windows 10,
  Linux проверен в Docker (`python:3.12-slim`).
- Python **3.12** — основной вариант (проверено на 3.12.7; `pyproject.toml`: `>=3.10,<3.13`). Зафиксированные версии `requirements.txt`
  (numpy 2.5, pandas 3.0 …) имеют колёса только для Python ≥ 3.12 и **не ставятся на 3.10 / 3.11**. Для них есть страховочный набор
  **`requirements-py310.txt`** (GPU) / `requirements-py310-cpu.txt` (CPU): те же torch 2.6.0, LightGBM 4.7.0 и smp 0.5.0, а numpy 2.2.6,
  pandas 2.2.3, scipy 1.15.3, rasterio 1.4.4 и др. — ближайшие версии с колёсами под 3.10–3.11. Проверено на Python 3.10.21: `inference.py`
  даёт **побайтно тот же файл**, что основной набор, `pytest` зелёный (`reports/tasklog/h100_readiness.md`, п. 18); для 3.11 проверено
  разрешение зависимостей (Linux x86_64). Python 3.13 — Docker (раздел 7): образ несёт свой Python 3.12.
- GPU **не обязательна**: на GPU (RTX 4070) полный инференс ≈ 23.5 с, без GPU ≈ 57 с (`--device cpu`); **GPU, CPU и Docker дают побайтно один
  файл**. Нужен драйвер NVIDIA **≥ 550** (CUDA 12.4): `requirements.txt` ставит `torch==2.6.0+cu124` из индекса
  `https://download.pytorch.org/whl/cu124` — CUDA-библиотеки приходят вместе с колесом, системный CUDA Toolkit не нужен (на Linux и обычный
  `torch==2.6.0` с PyPI уже собран с CUDA 12.4). Если драйвер старее или GPU не видна, инференс **сам переходит на CPU** с одной строкой
  предупреждения в логе, код выхода 0.
- Сеть при инференсе **не нужна**: веса энкодеров не скачиваются (`encoder_weights=None`, всё в `weights/`), проверено в Docker с `--network none`.
- RAM: инференс — пик **≈ 8.3 ГБ** (GPU-режим) / 7.9 ГБ (CPU-режим) при 16 потоках, память GPU **≈ 3.1 ГБ**; обучение AF ≤ 4 ГБ.
  Диск: ~6 ГБ на окружение с CUDA-сборкой torch (~1.5 ГБ с CPU-сборкой), данные train + test ≈ 3 ГБ.
- Потоки выбираются сами: `min(доступные ядра, 16)` с учётом квоты cgroup (`docker run --cpus N`) и affinity; на результат не влияют.

## 4. Установка

PowerShell (Windows):

```powershell
git clone <repo-url> fire-monitoring
cd fire-monitoring
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt          # с GPU (torch + CUDA 12.4)
# pip install -r requirements-cpu.txt    # без GPU: те же версии, torch CPU
$env:PYTHONIOENCODING = 'utf-8'          # кириллица в выводе скриптов
$env:PYTHONPATH = "$PWD\src"            # нужно только для команд вида `python -m firemon.…`
```

bash (Linux/macOS):

```bash
git clone <repo-url> fire-monitoring
cd fire-monitoring
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-cpu.txt      # или requirements.txt при наличии GPU
export PYTHONPATH=$PWD/src               # нужно только для команд вида `python -m firemon.…`
```

**Python 3.10 / 3.11** (если 3.12 поставить нельзя и Docker недоступен) — те же шаги, другой файл зависимостей:

```bash
python3.10 -m venv .venv && source .venv/bin/activate        # Windows: py -3.10 -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-py310.txt        # GPU (CUDA 12.4);  без GPU: pip install -r requirements-py310-cpu.txt
```

Результат инференса побайтно тот же; обучение и сервис на этом наборе отдельно не перепроверялись (кроме `pytest`).

`pip install -e .` **не нужен**: `inference.py`, `train.py`, скрипты и тесты сами добавляют `src/` в `sys.path`
(переменная `PYTHONPATH` требуется лишь для `python -m firemon.…`; альтернатива ей — `pip install -e .`).
Файлы зависимостей: `requirements-base.txt` (всё, кроме torch, точные версии) подключается из `requirements.txt`
(torch cu124) и `requirements-cpu.txt` (torch CPU); так же устроены `requirements-py310-base.txt` ← `requirements-py310.txt` /
`requirements-py310-cpu.txt` для Python 3.10–3.11; `requirements.lock.txt` — полный снимок окружения разработки.

## 5. Структура входных данных

Распаковка (внутри архивов уже лежат каталоги `train/` и `test/`):

```powershell
New-Item -ItemType Directory -Force data | Out-Null
tar -xf "C:\path\to\fire-train-renamed.tar" -C data
tar -xf "C:\path\to\fire-test-renamed.tar" -C data
python scripts/check_data.py                 # train: 420 AF × 3 файла, 224 BS × 6 файлов → «ВСЁ ОК»
python scripts/check_data.py --split test    # test: 180 AF × 2, 89 BS × 5, 447 строк шаблона → «ВСЁ ОК»
```

```bash
mkdir -p data && tar -xf /path/to/fire-train-renamed.tar -C data && tar -xf /path/to/fire-test-renamed.tar -C data
python scripts/check_data.py && python scripts/check_data.py --split test
```

Каталог теста (то, что передаётся в `--data-dir`):

```
data/test/
  meta.csv                 один на оба модуля: chip_id,kind,width,height,gsd,valid_frac,cloud_frac (180 af + 89 bs)
  sample_submission.csv    шаблон: 447 строк chip_id,class_id,rle (rle пустая) — AF по 1 строке, BS по 3
  af/viirs/AF_te_000001_VIIRS_I1-I5.tif      af/aux/AF_te_000001_AUX.tif
  bs/sentinel2_pre/BS_te_000001_Sentinel-2_pre.tif   bs/sentinel2_post/…   bs/sentinel1_pre/…   bs/sentinel1_post/…   bs/aux/…
```

- **Где `sample_submission.csv`:** ищется в порядке `--sample-submission` → `<data-dir>/sample_submission.csv` →
  `baseline/sample_submission.csv`. Шаблон определяет состав строк; чип из шаблона, которого нет на диске, получает пустую
  маску и warning. Без шаблона строки формируются по `meta.csv` / именам файлов.
- `meta.csv`: поддерживаются обе раскладки — `<data-dir>/<kind>/meta.csv` (как в train) и общий `<data-dir>/meta.csv`.
  Для инференса он не обязателен (нужен только для нестандартных размеров чипов).
- Train: `data/train/af/{meta.csv, viirs/, aux/, masks/}`, `data/train/bs/{meta.csv, sentinel2_pre/, …, aux/, masks/}`.
- Формат submission — **как в примере организаторов** `baseline/example_submission.csv`: заголовок `chip_id,class_id,rle`,
  строки в порядке шаблона, RLE с нумерацией с 1 (row-major, пары «start length» через пробел), колонка `rle` **без кавычек**,
  пустая маска — пустое поле, переводы строк CRLF. `tests/test_submission_example.py`: пример декодируется нашим `rle_decode`,
  обратное кодирование совпадает посимвольно, файл, собранный нами из масок примера, побайтно равен примеру. Читатель
  (валидатор, `evaluate_submission`, `compare_submissions.py`) принимает оба стиля — с кавычками и без. Скрипты RLE/метрики
  организаторы не выдавали: правила — по тексту ТЗ (`tests/test_rle_spec.py`), подтверждены лидербордом.

> **Windows:** `aux` — зарезервированное имя устройства. Файлы внутри `aux/` читаются нормально, но сам каталог нельзя
> открыть в Проводнике/PowerShell или создать обычным `mkdir`. Код использует `firemon.io.win_path()` (префикс `\\?\`,
> на Linux — no-op); вручную с этим каталогом ничего делать не нужно. `tar -xf` распаковывает его без проблем.

## 6. Запуск инференса

Одна команда (PowerShell и bash одинаково):

```
python inference.py --data-dir data/test --output out/submission.csv
```

`submission.csv` появляется по пути `--output` (каталоги создаются). Без интерактива; в конце печатается полное время.

| Конфигурация (269 чипов теста) | Время |
|---|---|
| для справки: AF LightGBM + BS-правила (`bs.model: rules`), CPU, 8 потоков | ~10–11 с (9.3–12.4) |
| **Docker CPU-образ `firemon`, обе модели, v5** (файл побайтно равен прогонам в venv; `--network none`) | **68.7 с** (Docker Desktop/WSL2, 16 потоков); с `--cpus 2` — 170 с |
| для справки: Docker-образ с BS-правилами | 11.1–11.5 с |
| для справки: то же, один поток (`--workers 1`) | ~19–21 с |
| **GPU (RTX 4070), AF LightGBM + BS v5 (3 UNet + UNet 0.5× + стек-LGBM)** — конфигурация по умолчанию | **23.2–24.1 с** (AF 4.3, BS ≈ 18; без TTA, fp32) |
| CPU (`--device cpu`), те же модели и настройки | **56.3–57.7 с** (BS 51 с; при 8 потоках — 68.8–69.2 с); файл **побайтно равен** GPU |
| быстрый режим v4 (`--config configs/inference_bs_base.yaml`), GPU / CPU | ~21 с / ~52 с; −0.005 Score |

Времена измерены на свободной машине (i5-13600KF, 20 потоков). Масштабирование по `--workers` на GPU: 2 → 27.0 с, 4 → 23.2, 8 → 23.7, 16 → 24.1, 20 → 24.1 с
(файл одинаков при любом числе). Когда все ядра заняты посторонним расчётом, запуск на GPU занимает на 5–7 с больше (первый, с холодным кэшем, — ещё дольше):
запас до порога 30 с ≈ 6 с есть только на свободной машине. **Результат не зависит от устройства**: GPU, CPU, Docker (Linux) и любой `--workers` — побайтно один файл
(UNet в fp32, TF32 выключен явно, TTA выключен).

**Чего ждать на сервере с H100.** GPU-часть (UNet) — ≈ 6.5 с на RTX 4070 и ≈ 2–3 с на H100 (оценка: сети маленькие, время упирается в передачу батчей);
CPU-часть (чтение deflate, признаки, стек-LGBM, постобработка, RLE) — ≈ 17 с при 8–16 ядрах уровня i5-13600KF и от числа ядер выше 8 почти не зависит.
Итого ожидание **≈ 19–22 с**; на серверных ядрах с меньшей частотой CPU-часть может вырасти до 20–25 с. H100 физически не проверялась —
проверено всё, что можно без неё: `reports/tasklog/h100_readiness.md`.
Подробности, профиль этапов и проверки детерминизма — `reports/speed.md`.

Необязательные аргументы: `--sample-submission CSV`, `--config configs/inference.yaml`, `--device auto|cpu|cuda`
(для нейросетевых бэкендов; LightGBM и правила всегда на CPU; `auto` при недоступной CUDA сам выбирает CPU), `--workers N` (потоки
чтения/признаков, по умолчанию `auto` = min(доступные ядра, 16) — на результат не влияет), `--profile` (таблица времени по этапам). Коды выхода: **0** — файл записан и прошёл валидацию;
**1** — записан, но не прошёл валидацию; **2** — нет входных данных/конфига (понятное сообщение вместо трассировки).

Воспроизводимость: повторный запуск даёт **побайтно тот же файл** при любом `--workers` и числе потоков LightGBM
(`tests/test_reproducibility.py`); результат в Docker (Linux) побайтно совпадает с Windows.

Проверка файла перед отправкой и оценка по известным маскам:

```
python scripts/check_submission.py out/submission.csv      # валидатор + сводка: пиксели по классам, пустые маски
python scripts/make_fake_test.py                           # 20 AF + 10 BS чипов train в data/test_fake
python inference.py --data-dir data/test_fake --output out/submission_fake.csv
python -m firemon.metrics --submission out/submission_fake.csv --masks-dir data/train --sample-submission data/test_fake/sample_submission.csv
```

### Как подключить модель

Модели подключаются в одном месте — реестрах `AF_BACKENDS` / `BS_BACKENDS` в `src/firemon/pipeline.py`; выбор — в
`configs/inference.yaml` (`af.model`, `bs.model`, запасной вариант `fallback`). Чтобы включить `BSPredictor` после слияния
`task/bs`: (1) добавить функцию-бэкенд в `BS_BACKENDS`, (2) положить веса в `weights/`, (3) выставить `bs.model`.
`inference.py`, Dockerfile и команды из этого README не меняются. Если бэкенд не поднялся (нет весов) — используется
`fallback` с ошибкой в логе: валидный submission важнее падения.

## 7. Docker

CPU-образ (проверен с обеими моделями: собран и запущен на реальном `data/test`, код 0, файл побайтно равен запуску `--device cpu` из venv):

```powershell
docker build -t firemon .
docker run --rm -v C:\path\to\test:/data:ro -v C:\path\to\out:/out firemon --data-dir /data --output /out/submission.csv
```

```bash
docker build -t firemon .
docker run --rm -v /path/to/test:/data:ro -v /path/to/out:/out firemon --data-dir /data --output /out/submission.csv
```

В образе: код, конфиги, веса из `weights/`, все зависимости (включая torch CPU для нейросетевого BS-бэкенда); сеть при
запуске не нужна (проверено с `--network none`; также с `LANG=C`, read-only корнем и данными, чужим рабочим каталогом, `--data-dir /data/`
со слэшем и несуществующим каталогом `--output`). `ENTRYPOINT ["python", "inference.py"]` — все аргументы те же, что в разделе 6. Размер образа ≈ 3.1 ГБ (из них веса 205 МБ),
сборка ≈ 5–6 мин (почти всё — загрузка колёс). Если в `/data` нет `sample_submission.csv`, передайте его явно:
`-v C:\path\to\sample_submission.csv:/sample.csv:ro … --sample-submission /sample.csv`.

**GPU-машина (Linux, драйвер NVIDIA ≥ 550).** Рекомендуемый путь — venv, он проверен (на Windows с RTX 4070; на Linux — разрешение
зависимостей для cp312):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                       # torch 2.6.0 + CUDA 12.4 внутри колеса
python inference.py --data-dir /path/to/test --output /path/to/out/submission.csv     # ≈ 23.5 с на RTX 4070; на H100 ожидаем ≈ 19–22 с
```

GPU-образ: `docker build -f Dockerfile.gpu -t firemon:gpu .` и
`docker run --rm --gpus all -v /path/to/test:/data:ro -v /path/to/out:/out firemon:gpu --data-dir /data --output /out/submission.csv`
(нужен NVIDIA Container Toolkit; ожидаемое время то же, ≈ 20–25 с; без `--gpus all` образ сам работает на CPU). **`Dockerfile.gpu` написан, но не собран из-за нехватки места на машине разработки; проверен CPU-образ**
(`reports/tasklog/h100_readiness.md`, п. 16); написан по тому же шаблону, что проверенный CPU-образ. Если нет Python 3.12 и
GPU-образ не собрался — CPU-образ выше даёт **побайтно тот же файл** за ≈ 69 с.

## 8. Веса

Веса лежат **в git** в `weights/` — 20 файлов (+ `.gitkeep`), всего **204.8 МБ**: AF 5.0 МБ, BS ≈ 200 МБ (четыре UNet по 49.3 МБ = 47.0 МиБ — каждый меньше лимитов 50 МиБ / 100 МБ; `bs_lgbm.txt` 4.8 МБ, три стека по 2.6 МБ, json-файлы — килобайты). При передаче архивами: `SAMARKAND_fire-monitoring_v1.0-final.zip` (`git archive` тега `v1.0-final`) уже содержит `weights/`; тот же каталог продублирован отдельным `SAMARKAND_fire-monitoring_weights_v1.0-final.zip` — если основной архив пришёл без весов (лимит размера), распакуйте его в корень репозитория: `Expand-Archive SAMARKAND_fire-monitoring_weights_v1.0-final.zip -DestinationPath .` / `unzip … -d .` (получится `weights/…`), проверка — `python inference.py …` не пишет в лог «веса не найдены». Отдельно скачивать ничего не нужно; после `git clone` файлы на диске совпадают с закоммиченными (сверено по blob-хэшам при слиянии):

```
weights/
  af_lgbm_fold0.txt … af_lgbm_fold4.txt   5 фолд-моделей LightGBM (текстовый формат)
  af_lgbm_meta.json                        список признаков, порог по OOF, параметры кандидатов
  bs_unet_r34_cedice.pt, bs_unet_r34_v4.pt  два UNet-resnet34 на полном разрешении, fp16 (в .gitignore стоит weights/*.pt → `git add -f`)
  bs_unet_r34_v4_s075.pt                   v5: UNet v4, обученный на масштабе 0.75× (третий член среднего)
  bs_unet_r34_v4_s050.pt                   v5: UNet v4 на масштабе 0.5× (его p_burn — признаки стека)
  bs_stack_ms.txt, bs_decode_ms.json       v5 (финал, `bs.variant: ms`): стек 2-го уровня, пороги и постобработка (event_zero_r 300)
  bs_stack.txt, bs_decode.json             v4 (быстрый режим, `bs.variant: base`)
  bs_stack_tta.txt, bs_decode_tta.json     вариант v4 с TTA (для сравнения)
  bs_lgbm.txt, bs_lc_medians.json          пиксельный LightGBM (режим use_lgbm, в финале выключен)
```

Нет весов AF → `inference.py` автоматически переходит на контекстное правило (OOF F1 0.8965) с ошибкой в логе.
Нет весов BS → бэкенд `unet` откатывается на `rules` (пороги dNBR) с ошибкой в логе.
Предобученные сторонние веса: энкодер resnet34 (ImageNet, torchvision BSD-3) через `segmentation_models_pytorch` (MIT) — только в BS-модели.

## 9. Обучение

```
python train.py --module af      # AF: признаки → data/cache/af_train.npz → 5 фолдов LightGBM → weights/af_lgbm_*
python train.py --module bs      # BS: правила → LGBM → 2×UNet (CV + full) → стек → декодирование (≈ 1.5 ч, GPU)
python train.py --module all     # af + bs
```

AF: ~1.5 мин с расчётом кэша признаков, ~30 с с готовым кэшем (8 потоков CPU, GPU не нужна). Печатает OOF F1
(GroupKFold по тайлу) и F1 на временном сплите (val = 2024), срезы по покрову/размеру очага; пишет
`reports/af_train_report.json` и строку в `reports/experiments.md` (`--no-log` — не писать; `--rebuild-cache` — пересчитать
кэш). **Детерминировано:** повторный запуск даёт OOF F1 0.9682 / temporal 0.9597 и веса, совпадающие по содержимому
байт-в-байт (проверено: `git diff` пуст; на Windows файлы пишутся с CRLF, в git хранятся с LF).

Валидация фиксирована: `configs/folds_af.csv` (5 групповых фолдов по тайлу), `configs/folds_bs.csv` (блоки 50×50 км),
колонка `temporal_split` (val = 2024). Пересоздание: `python -m firemon.splits --data-dir data/train` (seed 42).
Исследовательские скрипты AF (для воспроизведения сабмита не нужны): `scripts/af/*.py` — сетка правила, абляции,
постобработка, ансамбль, профиль, рисунки; EDA: `python eda/eda_af.py`, `python eda/eda_bs.py`, `python eda/test_vs_train.py`.

### Модуль BS (контур гари и степень поражения)

Схема: признаки (`src/firemon/features/bs_features.py`) → два UNet-resnet34 (среднее) выбирают регион гари → LightGBM 2-го уровня
уточняет пиксели и степень (по медианному dNBR, landcover, SCL и вероятностям UNet) → декодирование порогами →
постобработка. Подробности и все цифры — `reports/bs_module.md`.

```powershell
python train.py --module bs                          # всё: правила → LGBM → UNet (CV + full) → стек → декодирование
python train.py --module bs --bs-stages unet stack ensemble   # отдельные этапы: rules lgbm unet stack ensemble
python -m firemon.models.bs.lgbm --stage ablate      # абляции признаков (SAR / SCL / landcover / контекст / минимум / регрессия)
python scripts/bs_report_figures.py                  # рисунки и статистика для reports/bs_module.md
python scripts/bs_benchmark.py --device cuda         # время BS-инференса на 89 чипах (и --device cpu)
python scripts/bs_eda_checks.py --probs stack_avg14_event   # проверка фактов EDA об эталоне (буфер события, SCL, MMU, степень)
python scripts/bs_log_experiments.py                 # перенос результатов из data/cache/bs_oof/*.json в reports/experiments.md
```

Полный прогон на RTX 4070 ≈ 1.5 ч (UNet: 2 сети × (6 обучений × 5 мин + full); LGBM CV ≈ 15 мин; стек ≈ 15 мин; подбор
постобработки ≈ 10 мин). Промежуточные OOF-вероятности и выборки пикселей кэшируются в `data/cache/` (в git не входят,
нужно ≈ 8 ГБ диска и ≈ 16 ГБ RAM). Результат обучения — `weights/bs_unet_r34_cedice.pt` и `weights/bs_unet_r34_v4.pt`
(fp16, по 47 МБ; `encoder_weights="imagenet"` из smp — лицензия весов torchvision BSD-3; в git добавлены через `git add -f`),
`weights/bs_stack.txt`, `weights/bs_lgbm.txt`, `weights/bs_lc_medians.json`, `weights/bs_decode.json`.
**Кандидат v5 (многомасштабный BS, по умолчанию выключен):** `bs.variant: ms` в `configs/inference.yaml` или
`--config configs/inference_bs_ms.yaml` — плюс UNet, обученные на масштабах 0.75 и 0.5 (OOF BS-части 0.4965 / 0.5116 против
0.4912 / 0.5008; GPU 23.5 с, CPU 69 с; веса +97 МБ; обучение — этапы `ms_unet`, `ms_stack` в `train.py --module bs`;
подробности и оговорки — `reports/bs_module.md` §4.11).
Инференс настраивается в `configs/bs_inference.yaml` (`device: auto|cpu|cuda`, `use_stack`, `tta`, `tta_cpu`, `gpu_half`);
без GPU автоматически используется CPU-путь. **TTA выключен на всех устройствах, UNet на GPU считается в fp32**: прирост
от TTA на OOF всего +0.0043 / +0.0024 BS-части (< 0.005), зато результат не зависит от устройства — GPU- и CPU-файлы на
тесте отличаются 1 пикселем из 23.3 млн (`python scripts/compare_submissions.py a.csv b.csv`), повторные запуски на одном
устройстве побайтно одинаковы. Режим с TTA (чуть точнее, UNet ×4 по времени; результат GPU и CPU тогда различается
сильнее только если включить TTA не на обоих): `tta: true`, `tta_cpu: true`, `stack_weights: weights/bs_stack_tta.txt`,
`decode_file: weights/bs_decode_tta.json`. Проверка воспроизводимости обучения:
`python train.py --module bs --weights-dir weights_repro/` (свой кэш `data/cache/bs_weights_repro/`, основные веса не
затрагиваются) → `python scripts/bs_repro_compare.py` и инференс с `--config configs/inference_bs_repro.yaml`.

## 10. Сервис

Веб-карта + REST API: по полигону/bbox и интервалу дат — термоточки, контуры гарей (GeoJSON/Shapefile, атрибуты
id/класс/площадь) и справка по площадям (JSON/CSV). Подробности — `reports/service.md`, `service/README.md`.

```
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000      # или: python service/run.py
```

Открыть: http://localhost:8000 — карта (кнопка «Демо-запрос»), http://localhost:8000/docs — OpenAPI,
http://localhost:8000/health — состояние. Сервис стартует сразу после клонирования из снимка эталонных слоёв
`service/data/*.geojson.gz`. Слой предсказаний моделей строится командой (нужен `data/train`, ~2 мин):
`python scripts/build_service_data.py --source both` (какие модели использованы — `service/data/manifest.json`).

```bash
# 1) справка по площадям (JSON; format=csv — таблица)
curl "http://localhost:8000/api/v1/summary?bbox=39.25,47.95,39.50,48.10&date_from=2024-09-15&date_to=2024-10-10"
# 2) контуры гарей в Shapefile (zip: shp+shx+dbf+prj+cpg)
curl -o burn_scars_shp.zip "http://localhost:8000/api/v1/burn-scars?bbox=39.25,47.95,39.50,48.10&format=shp"
# 3) всё одним запросом: термоточки + гари + справка (область — bbox или GeoJSON-полигон в поле geometry)
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"bbox":[39.25,47.95,39.50,48.10],"date_from":"2024-09-15","date_to":"2024-10-10","source":"gt"}'
```

В PowerShell вместо `curl` — `curl.exe` (или `Invoke-RestMethod`).

## 11. Тесты

```
python -m pytest -q
```

Покрыто: I/O и обе раскладки meta, RLE по каждому правилу ТЗ и по примеру организаторов, метрика, сборка/валидация submission, фолды,
AF- и BS-модули, сервис, устойчивость инференса (NaN/нули/отсутствующие чипы, коды выхода), побайтная воспроизводимость, переносимость
(потоки, откат с CUDA, ASCII-консоль, отсутствие сетевых вызовов).

**Каким тестам нужны данные.** В чистом клоне без `data/` тесты не падают, а пропускаются (`pytest.skip` с пояснением):

| Нужно | Тесты | Без этого |
|---|---|---|
| ничего (только репозиторий) | `test_rle*.py`, `test_metrics.py`, `test_submission*.py` (пример организаторов — в `tests/fixtures/`), `test_portability.py`, `test_service.py` (снимок слоёв в git), `test_pipeline.py`, часть `test_bs.py`, `test_splits.py` | — |
| `data/train` | `test_io.py` (чтение train-чипов), `test_af.py`, большая часть `test_bs.py`, проверки фолдов по meta | skip: «нет data/train» |
| `data/test` | `test_inference_smoke.py::test_inference_on_real_test`, `test_reproducibility.py`, проверки тестовой раскладки | skip: «нет data/test» |
| `data/test_fake` (`python scripts/make_fake_test.py`, нужен `data/train`) | сквозной инференс с метрикой на 20 AF + 10 BS чипах | skip: «нет data/test_fake» |

С полными данными — 164 теста, ≈ 3–7 мин; в чистом клоне — см. `reports/tasklog/clone_check.md`.
На Windows тесты сами передают подпроцессам `PYTHONIOENCODING=utf-8`.

## 12. Состав репозитория

В git — **≈ 268 МБ** (≈ 360 файлов; каждый файл < 50 МБ, лимит GitHub 100 МБ не затронут — `reports/tasklog/release_check.md`,
проверка: `python scripts/check_release.py`). Данные (`data/`, ≈ 3 ГБ), файлы организаторов (`baseline/`), результаты запусков
(`out/`, `submissions/*.csv`) и `service/data/firemon.gpkg` (слой «модель» сервиса, 100 МБ, собирается командой) в git не входят.

```
                                                                                                            размер в git
inference.py            единственная точка входа инференса (CLI)                                                  7 КБ
train.py                обучение: python train.py --module af|bs|all
Dockerfile (CPU, проверен), Dockerfile.gpu (написан, не собран), .dockerignore
requirements.txt (GPU, Python 3.12) / requirements-cpu.txt / requirements-base.txt / requirements.lock.txt
requirements-py310.txt / requirements-py310-cpu.txt / requirements-py310-base.txt   страховка для Python 3.10–3.11
configs/                inference.yaml (v5 по умолчанию), inference_bs_base.yaml (быстрый режим v4), bs_inference.yaml,       < 0.1 МБ
                        af_lgbm.yaml, folds_af.csv, folds_bs.csv (фиксированная валидация)
weights/                веса моделей — 4 UNet по 47 МиБ, LightGBM AF ×5, стеки BS, json с порогами (раздел 8)            205 МБ
src/firemon/                                                                                                       0.2 МБ
  pipeline.py           конвейер инференса: реестры бэкендов AF/BS, потоки, профилировщик — МЕСТО ПОДКЛЮЧЕНИЯ МОДЕЛЕЙ
  io.py                 чтение чипов → dict массивов (C,H,W) + имена каналов, meta (обе раскладки), win_path
  rle.py                RLE encode / decode / validate (формат ТЗ)
  metrics.py            F1_af, IoU_burn, mIoU_sev, Score; evaluate_submission
  submission.py         сборка и валидация submission.csv по шаблону (текстовый формат — как пример организаторов)
  splits.py             групповые фолды (тайл / пространственный блок) + временной сплит
  utils.py              seed, логирование, auto_threads (ядра/cgroup), cuda_usable (откат на CPU)
  features/             af_features.py (110 признаков AF), bs_features.py (признаки BS, пороги правила степени)
  models/af/            predictor.py (AFPredictor), pipeline.py / train_lgbm.py / dataset.py (обучение), rules.py (правила)
  models/bs/            predictor.py (BSPredictor), infer.py, unet.py / lgbm.py / stack.py / ensemble.py / pipeline.py (обучение),
                        postprocess.py, rules.py (baseline dNBR), components.py (проверенная и не принятая гипотеза)
scripts/                check_data.py, check_submission.py, compare_submissions.py, check_release.py, make_fake_test.py,          0.2 МБ
                        build_service_data.py, make_demo_exports.py, make_report_figures.py, export_pdf.py, check_placeholders.py;
                        af/, bs_*.py, af_night_*.py — скрипты экспериментов (часть читает локальные каталоги весов экспериментов
                        `weights_repro/`, `weights_night_*/` и кэши `data/cache/` — в git их нет, воспроизводятся `train.py --weights-dir …`)
service/                FastAPI-приложение (app.py, …), static/ (интерфейс + vendor: Leaflet, Leaflet.draw, polygon-clipping,       17 МБ
                        Swagger UI), data/ — снимок эталонных слоёв *.geojson.gz + manifest.json
tests/                  pytest, 164 теста; fixtures/ — пример и шаблон submission организаторов (байт-в-байт)                    0.6 МБ
reports/                report.md / report.pdf, final_numbers.md, eda.md, baseline_review.md, af_module.md, bs_module.md,          30 МБ
                        service.md, experiments.md, submissions.md, speed.md, test_vs_train.md, figures/ (рисунки, 25 МБ),
                        tasklog/ (журналы задач: tz_compliance.md, release_check.md, h100_readiness.md, final_check.md, clone_check.md, merge_*.md)
presentation/           deck.pptx / deck.pdf (12 слайдов), build_deck.py (сборщик слайдов), qa.md (30 вопросов жюри)                  5 МБ
handoff/                HANDOFF.md — пакет для защиты: сценарий демо, чек-лист, копии PDF; demo_exports/ — выгрузки демо-запроса     6 МБ
eda/                    eda_af.py, eda_bs.py, test_vs_train.py, tables/ (46 csv)                                                 0.3 МБ
docs/                   ТЗ и критерии (pdf + txt)                                                                                  2 МБ
prompts/                постановки задач агентам (история разработки)                                                           0.1 МБ
data/, baseline/, out/, submissions/*.csv, service/data/firemon.gpkg, .venv/                                                  не в git
```

### Разработка в нескольких worktree

Модули делались параллельно в отдельных `git worktree` (`task/af`, `task/bs`, `task/eda`, `task/service`) рядом с основным:

- **Общий `.venv`** основного worktree, **без `pip install -e .`** — editable-установка привязала бы пакет `firemon` к одному
  worktree, и остальные молча импортировали бы чужой код.
- **`PYTHONPATH` = `<свой worktree>\src`** в каждой сессии: `$env:PYTHONPATH = "$PWD\src"` (bash: `export PYTHONPATH=$PWD/src`).
  Точки входа и тесты дополнительно сами добавляют свой `src/` в `sys.path`.
- **Junction-ы** `data`, `baseline`, `weights` в каждом worktree указывают на каталоги основного
  (`New-Item -ItemType Junction -Path data -Target ..\fire-monitoring\data`): данные не дублируются, веса сразу видны всем.
  Следствие: файл весов, записанный в одном worktree, появляется как неотслеживаемый во всех — перед слиянием ветки сверять
  его с веткой байт-в-байт.
- ⚠️ **Не удаляйте worktree с junction-ами через `Remove-Item -Recurse`** (и `rm -rf` из Git Bash): команда пройдёт по junction
  внутрь и **удалит общие `data/` и `weights/`**. Сначала снимите сами ссылки — `cmd /c rmdir data`, `cmd /c rmdir baseline`,
  `cmd /c rmdir weights` (удаляет только ссылку), затем `git worktree remove <путь>`.

## 13. Лицензии данных и моделей

- **Данные** предоставлены организаторами; источники открытые: VIIRS — общественное достояние (NASA), Sentinel-1/2 и Copernicus DEM GLO-30 —
  открытая лицензия Copernicus, ESA WorldCover v200 (CC BY 4.0), ERA5-Land (Copernicus C3S). Данные о территориях/датах теста, продукты
  FIRMS/VNP14/MCD64A1/FireCCI и ручная разметка теста **не используются**; модели работают только с содержимым чипа.
- **Предобученные модели:** AF — нет (LightGBM обучен с нуля). BS — энкодер resnet34 с весами ImageNet из torchvision:
  файл `resnet34-333f7ec4.pth` (https://download.pytorch.org/models/resnet34-333f7ec4.pth, `ResNet34_Weights.IMAGENET1K_V1`), репозиторий torchvision https://github.com/pytorch/vision (BSD-3-Clause);
  подключён через segmentation-models-pytorch 0.5.0 (https://github.com/qubvel-org/segmentation_models.pytorch, MIT); веса опубликованы задолго до старта
  и нужны только при обучении (на инференсе ничего не скачивается).
- **ПО** — только открытое: LightGBM (MIT), PyTorch (BSD-3), segmentation-models-pytorch (MIT), NumPy/SciPy/pandas (BSD),
  rasterio (BSD-3), GeoPandas/Shapely (BSD-3), FastAPI (MIT), Uvicorn (BSD-3), Leaflet (BSD-2), Leaflet.draw (MIT), polygon-clipping (MIT),
  Swagger UI (Apache-2.0); тексты лицензий — в `service/static/vendor/{leaflet,swagger,polygon-clipping}/LICENSE`. Подложки карты сервиса
  (Esri World Imagery с атрибуцией, OpenStreetMap, ODbL) — внешние тайлы, загружаются браузером, в репозиторий не входят; без сети
  интерфейс сам переходит в режим без подложки. Тайлы Esri — бесплатный, но не открытый сервис: на расчёты подложка не влияет и
  заменяется на OSM или «Нет» одной кнопкой.
- **Данные организаторов в репозитории**: снимок эталонных слоёв сервиса `service/data/*.geojson.gz` (производное от train-разметки) и
  `tests/fixtures/` (шаблон и пример submission) — выданы для участия в соревновании; репозиторий предполагается приватным / для жюри.

## 14. Команда

**SAMARKAND:** Кашин Глеб Игоревич, Закон Марк Михайлович, Панин Матвей Евгеньевич, Ожигов Егор Сергеевич, Ведерников Фёдор Яковлевич.
