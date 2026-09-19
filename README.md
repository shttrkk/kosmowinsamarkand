# Мониторинг природных пожаров — КосмоХакатон 2026, команда SAMARKAND

**Что это.** Решение кейса «Мониторинг природных пожаров»: модуль **AF** находит пиксели активного горения на чипах VIIRS
(256×256, 375 м), модуль **BS** строит контур гари и три степени поражения по Sentinel-2/1 (512×512, 20 м); одна команда
`inference.py` пишет `submission.csv` в формате RLE; веб-сервис с REST API показывает термоточки, контуры гарей и справку по площадям.

**Результат.** Public-лидерборд **0.8314** (`Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev`); локально 0.8353 на групповой
валидации и 0.8478 на 2024 годе. Полный инференс 269 тестовых чипов — **≈ 22–23 с на GPU**, ≈ 57 с на CPU.

**Воспроизводимость.** GPU, CPU и Docker дают **побайтно один и тот же файл** — он лежит в репозитории:
`submissions/submission_v5_final.csv`. Репозиторий: {{REPO_URL}}, релиз — тег `v1.0-final`.

![Схема пайплайна](reports/figures/pipeline.png)

Содержание: [1. Быстрый старт](#1-быстрый-старт--7-шагов) · [2. Сервис](#2-сервис--3-шага) · [3. Docker](#3-docker) ·
[4. Обучение](#4-обучение) · [5. Структура репозитория](#5-структура-репозитория) · [6. Результаты](#6-результаты) ·
[7. Воспроизводимость и ограничения](#7-воспроизводимость-и-ограничения) · [8. Лицензии](#8-лицензии) · [9. Команда](#9-команда)

---

## 1. Быстрый старт — 7 шагов

Нужно: Windows 10/11 или Linux x86_64 (glibc ≥ 2.28), **Python 3.12**, ≈ 6 ГБ диска под окружение (≈ 1.5 ГБ с CPU-сборкой torch),
8+ ГБ RAM. GPU не обязательна: с ней нужен драйвер NVIDIA ≥ 550 (CUDA 12.4 приходит внутри колеса torch, системный CUDA Toolkit
не нужен); без GPU или со старым драйвером инференс сам переходит на CPU с одной строкой предупреждения, код выхода 0.
Веса моделей уже лежат в `weights/` — скачивать ничего не нужно, **сеть при инференсе не используется**.

### Шаг 1. Получить код

Linux:

```bash
git clone {{REPO_URL}} fire-monitoring
cd fire-monitoring
```

Windows (PowerShell):

```powershell
git clone {{REPO_URL}} fire-monitoring
cd fire-monitoring
```

Если вместо адреса выдан архив — распакуйте его и перейдите в каталог, где лежит `inference.py`
(Linux: `unzip SAMARKAND_repo_v1.0-final.zip -d fire-monitoring`; Windows:
`Expand-Archive SAMARKAND_repo_v1.0-final.zip -DestinationPath fire-monitoring`). Веса внутри архива есть.

### Шаг 2. Окружение Python 3.12

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
$env:PYTHONIOENCODING = 'utf-8'
```

`PYTHONIOENCODING` нужен только для кириллицы в выводе вспомогательных скриптов; сам инференс работает и без него.

### Шаг 3. Установка зависимостей

Одинаково в Linux и Windows — выберите одну строку `pip install`:

```
pip install -r requirements.txt
python -m compileall -q src inference.py
```

| Ситуация | Файл зависимостей |
|---|---|
| есть GPU NVIDIA (драйвер ≥ 550), Python 3.12 | `requirements.txt` (torch 2.6.0 + CUDA 12.4, ≈ 2.5 ГБ загрузки) |
| нет GPU, Python 3.12 | `requirements-cpu.txt` (те же версии, torch CPU, ≈ 0.2 ГБ) |
| Python 3.10 или 3.11 | `requirements-py310.txt` (GPU) / `requirements-py310-cpu.txt` (CPU) |
| Python 3.13 или ставить пакеты нельзя | Docker, раздел 3 |

Версии зафиксированы точно. Основной набор (numpy 2.5, pandas 3.0 …) имеет колёса только для Python ≥ 3.12; набор `py310` —
ближайшие версии с колёсами под 3.10–3.11 (те же torch 2.6.0, LightGBM 4.7.0, smp 0.5.0), на Python 3.10.21 инференс даёт
побайтно тот же файл. `pip install -e .` **не нужен**: `inference.py`, `train.py`, скрипты и тесты сами добавляют `src/` в путь
импорта (переменная `PYTHONPATH=src` нужна только для команд вида `python -m firemon.…`).

### Шаг 4. Проверка установки

```
python -c "import torch, lightgbm, rasterio, segmentation_models_pytorch as smp; print(torch.__version__, torch.cuda.is_available())"
```

Должно напечатать `2.6.0+cu124 True` (на машине без GPU — `2.6.0+cpu False`). Заодно эта команда один раз загружает с диска
бинарные модули torch и LightGBM.

### Шаг 5. Положить тестовые данные в `data/test/`

Данные выдают организаторы (архив `fire-test-renamed.tar`, внутри уже лежит каталог `test/`); в репозиторий они не входят.

Linux:

```bash
mkdir -p data
tar -xf /path/to/fire-test-renamed.tar -C data
```

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force data | Out-Null
tar -xf C:\path\to\fire-test-renamed.tar -C data
```

Должно получиться:

```
data/test/
  meta.csv                   chip_id,kind,width,height,gsd,valid_frac,cloud_frac — 269 строк (180 af + 89 bs)
  sample_submission.csv      шаблон: 447 строк chip_id,class_id,rle — AF по 1 строке, BS по 3 (class_id 1, 2, 3)
  af/
    viirs/AF_te_000001_VIIRS_I1-I5.tif …     180 файлов, 8 каналов
    aux/AF_te_000001_AUX.tif …               180 файлов, 5 каналов
  bs/
    sentinel2_pre/BS_te_000001_Sentinel-2_pre.tif …      89 файлов
    sentinel2_post/BS_te_000001_Sentinel-2_post.tif …
    sentinel1_pre/BS_te_000001_Sentinel-1_pre.tif …
    sentinel1_post/BS_te_000001_Sentinel-1_post.tif …
    aux/BS_te_000001_AUX.tif …
```

Проверить комплектность: `python scripts/check_data.py --split test` → «ВСЁ ОК». Данные могут лежать где угодно — тогда на шаге 6
укажите свой путь в `--data-dir`. Шаблон строк ищется в порядке: `--sample-submission` → `<data-dir>/sample_submission.csv` →
`baseline/sample_submission.csv`; **каталог `baseline/` для инференса не нужен**. Чип из шаблона, которого нет на диске или который
не читается, получает пустую маску и предупреждение — пайплайн не падает.

> **Windows:** `aux` — зарезервированное имя устройства. Файлы внутри `aux/` читаются нормально, но сам каталог нельзя открыть в
> Проводнике или создать обычным `mkdir`. Код использует `firemon.io.win_path()` (префикс `\\?\`, на Linux — ничего не делает);
> вручную с этим каталогом ничего делать не нужно, `tar -xf` распаковывает его без проблем.

### Шаг 6. Инференс

Одинаково в Linux и Windows:

```
python inference.py --data-dir data/test --output submission.csv
```

Без интерактива; в конце печатается «Полное время инференса». Необязательные аргументы: `--sample-submission CSV`,
`--config configs/inference.yaml`, `--device auto|cpu|cuda` (`auto` при недоступной CUDA сам выбирает CPU), `--workers N`
(по умолчанию `auto` = min(доступные ядра, 16) с учётом квоты cgroup; на результат не влияет), `--profile` (таблица времени по этапам).

| Код выхода | Значение |
|---|---|
| 0 | файл записан и прошёл валидацию |
| 1 | файл записан, но не прошёл валидацию (ошибки — в логе) |
| 2 | нет входных данных или конфига (понятное сообщение вместо трассировки) |

### Шаг 7. Где файл и как его проверить

Файл — там, куда указывает `--output` (каталоги создаются сами); в примере это `submission.csv` в корне репозитория.

```
python scripts/check_submission.py submission.csv
```

Печатает «Формат: OK (447 строк)» и сводку пикселей по классам. **Эталон** — сданный нами файл
`submissions/submission_v5_final.csv`: ваш результат должен совпасть с ним побайтно, sha256
`784fe474040914ed561c29efb8f34c1115e07f422e575e45bcfbceb683523afc`.

Linux:

```bash
sha256sum submission.csv submissions/submission_v5_final.csv
```

Windows (PowerShell):

```powershell
Get-FileHash submission.csv, submissions\submission_v5_final.csv -Algorithm SHA256
```

Формат файла — **как в примере организаторов** `example_submission.csv`: заголовок `chip_id,class_id,rle`, строки в порядке шаблона,
RLE с нумерацией с 1 построчно (пары «start length» через пробел, по возрастанию, серии не соприкасаются), колонка `rle` без кавычек,
пустая маска — пустое поле, переводы строк CRLF. Копия примера лежит в `tests/fixtures/`; тест `tests/test_submission_example.py`
проверяет, что файл, собранный нами из масок примера, побайтно равен примеру.

**Ожидаемое время** (269 чипов, от старта процесса до провалидированного файла; i5-13600KF, 20 потоков):

| Конфигурация | Время |
|---|---|
| GPU RTX 4070, по умолчанию (v5) | **≈ 22–23 с** (AF ≈ 5 с, BS ≈ 16–17 с) |
| только CPU (`--device cpu`) | ≈ 57 с |
| Docker CPU-образ | ≈ 69 с |
| H100 | оценка ≈ 19–22 с — **не проверялось**: GPU-часть ≈ 6.5 с из 23, остальное — CPU-этапы (чтение, признаки, LightGBM, постобработка) |
| быстрый режим v4: `--config configs/inference_bs_base.yaml` | GPU ≈ 21 с, CPU ≈ 52 с; −0.005 Score |

**Замеряйте второй запуск.** Первый запуск в только что созданном окружении занимал у нас от 23 до 42 с: с диска впервые читаются
≈ 3 ГБ бинарных модулей torch / CUDA. Второй и следующие запуски стабильны — 22–23 с. Подробности — `reports/speed.md`.

---

## 2. Сервис — 3 шага

Веб-карта и REST API: по полигону или bbox и интервалу дат — термоточки, контуры гарей (GeoJSON / Shapefile с атрибутами
id, класс, площадь) и справка по площадям в гектарах (JSON / CSV). Работает на подготовленном наборе сцен — 644 чипа обучающей
выборки с геопривязкой; у тестовых чипов геопривязки нет, подключить их нельзя. Описание — `reports/service.md`, `service/README.md`.

### Шаг 1. Запуск

Окружение — то же, что для инференса (шаги 1–3 выше):

```
python -m uvicorn service.app:app --host 127.0.0.1 --port 8000
```

Если порт 8000 занят (uvicorn сообщит об ошибке привязки, а `/health` может ответить чужой процесс) — возьмите другой: `--port 8001`, и подставьте его в адреса ниже.

Старт ≈ 5–8 с, ждать строки «Application startup complete». Сервис поднимается сразу после клонирования: снимок эталонных слоёв
`service/data/*.geojson.gz` лежит в репозитории, `data/train` для этого не нужен.

### Шаг 2. Открыть

- http://localhost:8000 — карта;
- http://localhost:8000/docs — описание API (OpenAPI, работает без интернета);
- http://localhost:8000/health — состояние, должно быть `"status":"ok"`.

### Шаг 3. Демо-запрос

Кнопка **«Демо-запрос»** на карте — или ссылка, которая выполняет тот же запрос сама при открытии (удобно как закладка):

```
http://localhost:8000/?bbox=39.25,47.95,39.5,48.1&from=2024-09-15&to=2024-10-10&source=gt
```

Ожидаемый результат: **98 термоточек, 337 контуров, гарь 1 758.92 га** (слабая 115.0 / средняя 951.9 / сильная 692.0 га).
Термоточки раскрашены по температуре I4 (300 → 360 K), гари — по степени, поверх — контур события; наведение показывает
подсказку, клик — все атрибуты; слева справка и кнопки выгрузки.

**Слой «Модель».** Переключатель «Эталон / Модель» показывает предсказания наших AF- и BS-модулей. Этот слой в репозиторий не входит
(100 МБ) и собирается из обучающих данных — нужен `data/train` и ≈ 2 минуты на GPU:

```
python scripts/build_service_data.py --source both
```

Пока слой не собран, переключатель заблокирован, всё остальное работает. Какие модели использованы — `service/data/manifest.json`.

**Интернет.** Подложки «Спутник» (Esri World Imagery) и «Карта» (OpenStreetMap) — внешние тайлы, им нужен интернет. Без сети
интерфейс сам переходит в режим **«Нет»**: остаются граница территории и охваты сцен, все запросы, выгрузки и `/docs` работают офлайн.

**Примеры запросов к API** (Linux; в PowerShell вместо `curl` пишите `curl.exe`):

```bash
# 1) справка по площадям (JSON; format=csv — таблица)
curl "http://localhost:8000/api/v1/summary?bbox=39.25,47.95,39.50,48.10&date_from=2024-09-15&date_to=2024-10-10"
# 2) контуры гарей в Shapefile (zip: shp + shx + dbf + prj + cpg)
curl -o burn_scars_shp.zip "http://localhost:8000/api/v1/burn-scars?bbox=39.25,47.95,39.50,48.10&format=shp"
# 3) всё одним запросом: термоточки + гари + справка (область — bbox или GeoJSON-полигон в поле geometry)
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d '{"bbox":[39.25,47.95,39.50,48.10],"date_from":"2024-09-15","date_to":"2024-10-10","source":"gt"}'
```

Слои отдаются в EPSG:4326 и открываются в QGIS / ArcGIS как есть; площади считаются в UTM 37N / 38N, отклонение от эталонных
площадей на всех 224 чипах — не больше 0.025 %; ответ на весь архив за весь период — ≈ 0.4 с.

---

## 3. Docker

**CPU-образ — проверен**: собран и запущен на реальном тесте, код 0, файл побайтно равен прогонам в venv; проверен без сети
(`--network none`), с `LANG=C`, с read-only данными и корневой ФС, из чужого рабочего каталога, с `--cpus 2`.

Linux:

```bash
docker build -t firemon .
docker run --rm -v /path/to/test:/data:ro -v /path/to/out:/out firemon --data-dir /data --output /out/submission.csv
```

Windows (PowerShell):

```powershell
docker build -t firemon .
docker run --rm -v C:\path\to\test:/data:ro -v C:\path\to\out:/out firemon --data-dir /data --output /out/submission.csv
```

В образе — Python 3.12, код, конфиги, веса и все зависимости (torch CPU); размер ≈ 3.1 ГБ, сборка ≈ 5–6 мин, инференс ≈ 69 с
(с `--cpus 2` — ≈ 170 с). Аргументы те же, что на шаге 6. Если в `/data` нет `sample_submission.csv`, передайте его явно:
`-v /path/to/sample_submission.csv:/sample.csv:ro … --sample-submission /sample.csv`.

**GPU-образ — не проверен.** `Dockerfile.gpu` написан по тому же шаблону (база `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`),
но **не собран**: на машине разработки не хватило места на диске. Команды, которые должны работать (нужен NVIDIA Container Toolkit):

```bash
docker build -f Dockerfile.gpu -t firemon:gpu .
docker run --rm --gpus all -v /path/to/test:/data:ro -v /path/to/out:/out firemon:gpu --data-dir /data --output /out/submission.csv
```

Надёжный путь для машины с GPU — venv из раздела 1; если нет подходящего Python — CPU-образ даёт побайтно тот же файл.

---

## 4. Обучение

Нужны обучающие данные организаторов (`fire-train-renamed.tar` → `data/train/`; распаковка — как на шаге 5,
проверка: `python scripts/check_data.py`).

```
python train.py --module af
python train.py --module bs
python train.py --module all
```

| Модуль | Что делает | Время | Повторяемость |
|---|---|---|---|
| `af` | кандидаты день/ночь → 110 признаков → кэш `data/cache/af_train.npz` → 5 фолд-моделей LightGBM → `weights/af_lgbm_*` | ≈ 1.5 мин (≈ 30 с с готовым кэшем), GPU не нужна | **детерминировано**: OOF F1 0.9682 / 0.9597, веса совпадают по содержимому |
| `bs` | правила → пиксельный LightGBM → два UNet-resnet34 (CV + full) → стек 2-го уровня → подбор порогов и постобработки → `weights/bs_*` | ≈ 1 ч 45 мин на RTX 4070; ≈ 8 ГБ диска под `data/cache/`, ≈ 16 ГБ RAM | **нет** — см. ниже |

Отдельные этапы BS: `python train.py --module bs --bs-stages unet stack ensemble` (этапы: `rules lgbm unet stack ensemble`).
Финальная модель **v5** добавляет два UNet, обученных на пониженном разрешении (0.75× и 0.5×), и свой стек:

```
python train.py --module bs --bs-stages ms_unet ms_stack
```

Веса пишутся в `weights/bs_unet_r34_v4_s075.pt`, `bs_unet_r34_v4_s050.pt`, `bs_stack_ms.txt`, `bs_decode_ms.json`. Поставляемое
декодирование использует `event_zero_r: 300` — на нём измерен public-результат; при переобучении жадный подбор выбирает 350,
разница 0.0004 BS-части, это шум. В отдельный каталог, не трогая основные веса: `--weights-dir weights_repro/`.

**Честно про повторяемость обучения BS.** Контрольный полный повтор с тем же seed дал BS-часть 0.4718 / 0.4937 против
0.4912 / 0.5008 — разброс до 0.02 при допуске 0.005. Причина — недетерминированный backward взвешенного `F.cross_entropy` на CUDA.
Исправлено (one-hot CE в `unet.py`, два коротких обучения дают побитно равные веса), но **поставляемые веса обучены до
исправления**, полный повтор с новой функцией не делался. Правила, LightGBM и весь инференс повторяются точно.

Валидация фиксирована: `configs/folds_af.csv` (5 групповых фолдов по тайлу), `configs/folds_bs.csv` (блоки 50×50 км), колонка
`temporal_split` (val = 2024); seed 42. Прирост засчитывался только при улучшении на обоих сплитах; каждый эксперимент записан
в `reports/experiments.md`. Оценка файла по известным маскам:

```
python scripts/make_fake_test.py
python inference.py --data-dir data/test_fake --output out/submission_fake.csv
```

Linux: `PYTHONPATH=src python -m firemon.metrics --submission out/submission_fake.csv --masks-dir data/train --sample-submission data/test_fake/sample_submission.csv`;
Windows: сначала `$env:PYTHONPATH = "$PWD\src"`, затем та же команда без префикса.

### Как подключить свою модель

Модели подключаются в одном месте — реестрах `AF_BACKENDS` / `BS_BACKENDS` в `src/firemon/pipeline.py`; выбор — в
`configs/inference.yaml` (`af.model`, `bs.model`, запасной вариант `fallback`, вариант BS `bs.variant: ms | base`). Бэкенд — функция
`(data_dir, chip_ids, cfg, ctx) -> (preds, n_failed)`. Если бэкенд не поднялся (например, нет весов), используется `fallback`
(правила) с ошибкой в логе: валидный submission важнее падения. `inference.py` и Dockerfile при этом не меняются.

### Тесты

```
python -m pytest -q
```

170 тестов: I/O, RLE по каждому правилу ТЗ и по примеру организаторов, метрика, сборка и валидация submission, фолды, оба модуля,
сервис, устойчивость инференса (NaN, нули, отсутствующие чипы, коды выхода), побайтная воспроизводимость, переносимость (потоки,
откат с CUDA на CPU, ASCII-консоль, отсутствие сетевых вызовов). Без данных тесты не падают, а пропускаются с пояснением:

| Нужно | Тесты | Без этого |
|---|---|---|
| ничего (только репозиторий) | `test_rle*.py`, `test_metrics.py`, `test_submission*.py` (пример организаторов — в `tests/fixtures/`), `test_portability.py`, `test_service_snapshot.py` (снимок слоёв в git), `test_pipeline.py`, часть `test_bs.py` | — |
| `data/train` | `test_io.py`, `test_af.py`, `test_splits.py`, большая часть `test_bs.py` и `test_service.py` (пересобирает слои из train) | skip: «нет data/train» |
| `data/test` | `test_inference_smoke.py::test_inference_on_real_test`, `test_reproducibility.py`, проверки тестовой раскладки | skip: «нет data/test» |
| `data/test_fake` (`python scripts/make_fake_test.py`, нужен `data/train`) | сквозной инференс с метрикой на 20 AF + 10 BS чипах | skip: «нет data/test_fake» |

С полными данными — 169 passed, 1 skipped, ≈ 3–7 мин; в чистом клоне без данных — **130 passed, 40 skipped**, ≈ 35 с.

---

## 5. Структура репозитория

В git — **≈ 266 МБ**, каждый файл меньше 50 МБ. Проверка состава: `python scripts/check_release.py`. Не входят в git: данные
(`data/`, ≈ 3 ГБ), файлы организаторов (`baseline/`), результаты запусков (`out/`), слой «модель» сервиса
(`service/data/firemon.gpkg`), кэши и веса экспериментов.

```
                                                                                                                    размер
inference.py            единственная точка входа инференса (CLI)
train.py                обучение: python train.py --module af|bs|all
Dockerfile              CPU-образ (проверен);  Dockerfile.gpu — GPU-образ (написан, не собран);  .dockerignore
requirements.txt        Python 3.12, torch + CUDA 12.4;  requirements-cpu.txt — torch CPU;  requirements-base.txt — всё, кроме torch
requirements-py310*.txt набор для Python 3.10–3.11;  requirements.lock.txt — полный снимок окружения разработки
pyproject.toml, LICENSE (MIT), .gitignore, .gitattributes, CLAUDE.md (памятка для разработки: факты о данных, соглашения)
configs/                inference.yaml (v5 по умолчанию), inference_bs_base.yaml (быстрый режим v4), bs_inference.yaml,           < 0.1 МБ
                        af_lgbm.yaml, folds_af.csv, folds_bs.csv (фиксированная валидация)
weights/                4 UNet-resnet34 по 47 МиБ, 5 моделей LightGBM для AF, стеки BS, json с порогами и постобработкой            205 МБ
src/firemon/                                                                                                              0.2 МБ
  pipeline.py           конвейер инференса: реестры бэкендов AF/BS, потоки, профилировщик, фоновый прогрев torch
  io.py                 чтение чипов → массивы (C,H,W) + имена каналов, meta.csv в обеих раскладках, win_path
  rle.py                RLE: кодирование, декодирование, проверка правил ТЗ
  metrics.py            F1_af, IoU_burn, mIoU_sev, Score; оценка файла submission по маскам
  submission.py         сборка и валидация submission.csv по шаблону (текстовый формат — как пример организаторов)
  splits.py             групповые фолды (тайл / пространственный блок) и временной сплит
  utils.py              seed, логирование, выбор числа потоков (ядра, cgroup), проверка CUDA с откатом на CPU
  features/             af_features.py (110 признаков AF), bs_features.py (признаки BS, пороги правила степени)
  models/af/            predictor.py (AFPredictor), обучение LightGBM, контекстное правило день/ночь
  models/bs/            predictor.py (BSPredictor), infer.py, unet.py / lgbm.py / stack.py / ensemble.py (обучение),
                        postprocess.py, rules.py (baseline dNBR), components.py (проверенная и не принятая гипотеза)
scripts/                check_data.py, check_submission.py, compare_submissions.py, check_release.py, make_fake_test.py,          0.2 МБ
                        build_service_data.py, make_report_figures.py; af/, bs_*.py, af_night_*.py — скрипты экспериментов
                        (часть читает локальные каталоги весов экспериментов и кэши data/cache/ — для инференса не нужны)
service/                FastAPI-приложение, static/ (интерфейс + локальные Leaflet, Leaflet.draw, polygon-clipping, Swagger UI),    18 МБ
                        data/ — снимок эталонных слоёв *.geojson.gz и manifest.json
tests/                  pytest, 170 тестов; fixtures/ — пример и шаблон submission организаторов, байт-в-байт                      0.5 МБ
eda/                    eda_af.py, eda_bs.py, test_vs_train.py, tables/ (46 csv)                                                  0.4 МБ
reports/                report.md (полный отчёт), final_numbers.md (единый источник чисел), experiments.md, submissions.md,          28 МБ
                        af_module.md, bs_module.md, eda.md, baseline_review.md, service.md, speed.md;
                        json с сырыми числами экспериментов; figures/ — рисунки (27 МБ)
submissions/            submission_v5_final.csv — сданный файл, эталон результата инференса                                          3 МБ
```

Веса (`weights/`): `af_lgbm_fold0..4.txt` + `af_lgbm_meta.json` (список признаков, порог по OOF); `bs_unet_r34_cedice.pt` и
`bs_unet_r34_v4.pt` — UNet на полном разрешении; `bs_unet_r34_v4_s075.pt` — UNet, обученный на масштабе 0.75× (третий член
среднего); `bs_unet_r34_v4_s050.pt` — UNet на 0.5× (его вероятности — признаки стека); `bs_stack_ms.txt` + `bs_decode_ms.json` —
стек и пороги финала v5; `bs_stack.txt` + `bs_decode.json` — быстрый режим v4; `bs_stack_tta.txt` + `bs_decode_tta.json` —
вариант с TTA для сравнения; `bs_lgbm.txt`, `bs_lc_medians.json` — пиксельный LightGBM и медианы по покрову.
Нет весов AF → автоматический переход на контекстное правило (OOF F1 0.8965); нет весов BS → на пороги dNBR; в обоих случаях
ошибка в логе и валидный файл.

---

## 6. Результаты

Метрика: `Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev`, микро-усреднение по пикселям всех чипов. Две локальные оценки:
групповая валидация GroupKFold-5 (AF — по тайлу, BS — по блоку 50×50 км) и временной сплит (val = 2024). Единый источник всех
чисел — `reports/final_numbers.md`, полный отчёт — `reports/report.md`.

| Модуль | Модель | Групповая валидация (OOF) | Временной сплит 2024 |
|---|---|---|---|
| AF | baseline-правило `I4 > 320 K & I4−I5 > 15 K` | F1 0.2435 | — |
| AF | **LightGBM по пикселям-кандидатам (финал)** | **F1 0.9682** | **F1 0.9597** |
| BS | baseline-правило dNBR 0.10 / 0.27 / 0.44 | IoU_burn 0.3706, mIoU_sev 0.3129 | 0.3571, 0.2967 |
| BS | v4: 2 UNet → стек-LGBM → постобработка (быстрый режим) | 0.7560, 0.7554 | 0.7723, 0.7684 |
| BS | **v5: UNet v1, v4 и v4@0.75× → среднее → стек с признаками UNet 0.5×, ворота 0.01 → пороги → постобработка (финал)** | **0.7645, 0.7632** | **0.7904, 0.7831** |

Лестница отправок:

| № | Отправка | OOF | 2024 год | Public |
|---|---|---|---|---|
| 1 | оба baseline-правила | 0.3088 | — | 0.3117 |
| 2 | + AF LightGBM | 0.5625 | 0.5499 | 0.6035 |
| 3 | + BS 2 UNet → стек, вариант с TTA и fp16 | 0.8344 | 0.8392 | 0.8267 |
| 4 | v4: без TTA, fp32 — результат не зависит от устройства | 0.8300 | 0.8367 | 0.8262 |
| 5 | **v5, финал: многомасштабный ансамбль (`submissions/submission_v5_final.csv`)** | **0.8353** | **0.8478** | **0.8314** |

Public финала на 0.004–0.016 ниже локальных оценок — переобучения нет: в тесте облачных BS-чипов 25.8 % против 10.7 % в обучающей
выборке. Разница −0.0005 между отправками 3 и 4 — сознательная цена детерминизма (отказ от TTA и fp16). v5 даёт +0.0052 на public
к v4 ценой ≈ 2.5 с на GPU: сети, обученные на пониженном разрешении, видят больший контекст и лучше выбирают «своё» событие.

Главная мысль решения BS: степень поражения в эталоне — это пороги по медианному dNBR 3×3, свои для каждого типа покрова (правило
воспроизводит 97 % степеней), а вся трудность — выбрать «свой» пожар внутри буфера события ≈ 4 км, не центрированного по чипу.
UNet выбирает регион, бустинг восстанавливает границы и степень. Подробно — `reports/bs_module.md`, `reports/af_module.md`.

---

## 7. Воспроизводимость и ограничения

**Что воспроизводится точно.**
- Инференс: GPU, CPU, Docker (Linux), Windows, любое `--workers`, Python 3.10 и 3.12 — **побайтно один файл**, sha256 `784fe474…`
  (UNet в fp32, TF32 и автоподбор алгоритмов cuDNN выключены, TTA выключен). Проверено и в чистом клоне репозитория.
- Обучение AF — те же метрики и веса при повторном запуске.
- Сеть при инференсе не используется (веса энкодеров не скачиваются), данные открываются только на чтение.

**Ограничения — честно.**
- **Обучение BS не повторяется в пределах ±0.005** (до 0.02): причина найдена и исправлена в коде, но поставляемые веса обучены
  до исправления (раздел 4).
- **H100 не проверялась**, оценка ≈ 19–22 с — расчётная. **GPU-образ Docker не собран.** Проверены venv на GPU и CPU и CPU-образ.
- **Порог 30 с** выдерживается на свободной машине с GPU с запасом ≈ 7 с; под посторонней нагрузкой и в самом первом запуске в
  свежем окружении (23–42 с) — не гарантирован. Запасной вариант — быстрый режим v4.
- Официальный скрипт метрики и baseline организаторы не выдавали: метрика реализована по тексту ТЗ (`tests/test_metrics.py`),
  в отчёте разбирается наш baseline-правило (`reports/baseline_review.md`).
- Формат колонки `rle` — по примеру организаторов (без кавычек, пустая маска — пустое поле); в тексте ТЗ упомянуты кавычки.
  CSV-парсеры читают оба варианта одинаково, отправки приняты лидербордом.
- Слой «Модель» сервиса построен на обучающих сценах — это демонстрация, а не оценка качества (площадь 237 231.6 га против
  236 359.3 га в эталоне, +0.37 %).
- Матрица ошибок и срезы BS по облачности в отчёте посчитаны на варианте с TTA; для финала порядок величин тот же.
- В репозитории лежат **производные от данных организаторов**: снимок эталонных слоёв сервиса (`service/data/*.geojson.gz`),
  копии шаблона и примера submission (`tests/fixtures/`), сданный файл. Данные выданы для участия в соревновании — репозиторий
  предполагается закрытым, с доступом для жюри.
- Правила соревнования: данные о территориях и датах теста, продукты FIRMS / VNP14 / MCD64A1 / FireCCI и ручная разметка теста не
  используются; геопривязка и даты тестовых чипов не восстанавливаются; ни один признак не использует географические координаты,
  даты, `chip_id` или порядок чипов (положение пикселя внутри чипа — используется).

---

## 8. Лицензии

- **Наш код — MIT**, файл `LICENSE`.
- **Данные** предоставлены организаторами; источники открытые: VIIRS — общественное достояние (NASA); Sentinel-1/2 и
  Copernicus DEM GLO-30 — открытая лицензия Copernicus; ESA WorldCover v200 — CC BY 4.0; ERA5-Land — Copernicus C3S.
- **Предобученные веса** — только в BS: энкодер resnet34 с весами ImageNet из torchvision, файл `resnet34-333f7ec4.pth`
  (https://download.pytorch.org/models/resnet34-333f7ec4.pth, `ResNet34_Weights.IMAGENET1K_V1`), репозиторий torchvision
  https://github.com/pytorch/vision (BSD-3-Clause); подключён через segmentation-models-pytorch 0.5.0
  (https://github.com/qubvel-org/segmentation_models.pytorch, MIT). Веса опубликованы задолго до старта соревнования и нужны
  только при обучении — на инференсе ничего не скачивается. AF обучен с нуля.
- **ПО — только открытое:** LightGBM (MIT), PyTorch (BSD-3), segmentation-models-pytorch (MIT), NumPy / SciPy / pandas (BSD),
  scikit-learn (BSD-3), rasterio (BSD-3), GeoPandas / Shapely (BSD-3), FastAPI (MIT), Uvicorn (BSD-3), Leaflet (BSD-2),
  Leaflet.draw (MIT), polygon-clipping (MIT), Swagger UI (Apache-2.0); тексты лицензий встроенных библиотек —
  `service/static/vendor/{leaflet,swagger,polygon-clipping}/LICENSE`.
- **Подложки карты** — внешние тайлы, загружаются браузером, в репозиторий и в расчёты не входят: Esri World Imagery (атрибуция
  «Tiles © Esri — Esri, Maxar, Earthstar Geographics, and the GIS User Community»; бесплатный, но не открытый сервис) и
  OpenStreetMap (ODbL). Одной кнопкой заменяются друг на друга или на режим без подложки.

---

## 9. Команда

**SAMARKAND** — Кашин Глеб Игоревич, Закон Марк Михайлович, Панин Матвей Евгеньевич, Ожигов Егор Сергеевич,
Ведерников Фёдор Яковлевич.
