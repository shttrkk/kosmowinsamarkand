# Промпт 1 — Инфраструктура: репозиторий, данные, I/O, RLE, метрика, сплиты, тесты

> Запускать ПЕРВЫМ, один агент. Все остальные задачи зависят от него.
> Перед началом прочитай `CLAUDE.md` целиком.

## Контекст
Мы делаем решение кейса «Мониторинг природных пожаров» (КосмоХакатон 2026). Исходники лежат в `C:\Users\User\Downloads\hackatonchik\`:
- `Постановка_кейса_—_…pdf`, `Критерии_оценки_—_…pdf` — ТЗ и критерии;
- `Мониторинг DATA\Мониторинг DATA\fire-train-renamed.tar` — обучающий набор (2.3 ГБ, внутри `train/af/…`, `train/bs/…`);
- `Мониторинг DATA\Мониторинг DATA\fire-aoi\` — граница территории (geojson/gpkg/shp);
- `_work\` — черновики, можно игнорировать;
- baseline и тестовый набор организаторы добавят позже (ожидается ~через 2 часа) — их надо будет положить в `baseline/` и `data/test/`.

## Что сделать

### 1. Репозиторий
1. Создай папку `C:\Users\User\hack\fire-monitoring\`, `git init`, скопируй туда `CLAUDE.md` (лежит рядом с этим промптом: `prompts\00_CLAUDE.md`), положи PDF ТЗ в `docs/` и сделай их текстовые версии `docs/case.txt`, `docs/criteria.txt` (pdftotext / pypdf — любой способ).
2. Создай структуру каталогов из раздела 8 CLAUDE.md, `.gitignore` (data/, baseline/, weights/*.pt больше 50 МБ, __pycache__, .venv, *.tif, service/data/*.gpkg и т.п.), `pyproject.toml` (пакет `firemon` в `src/`, установка `pip install -e .`).
3. Окружение: создай `.venv` (Python 3.10/3.11), установи и зафиксируй в `requirements.txt` с точными версиями: `numpy, pandas, rasterio, tifffile, scipy, scikit-image, scikit-learn, lightgbm, torch (CUDA-сборка, проверь `torch.cuda.is_available()` и выведи имя GPU), segmentation-models-pytorch, albumentations, pyyaml, tqdm, matplotlib, geopandas, shapely, pyproj, fastapi, uvicorn, pytest, httpx`. Если rasterio на Windows ставится плохо — ставь wheel из pip (обычно ок), в крайнем случае fallback на tifffile + ручной парс GeoTIFF-тегов (ModelPixelScaleTag, ModelTiepointTag, GDAL_METADATA).
4. Распакуй tar в `data/train/` так, чтобы получились `data/train/af/{meta.csv,viirs,aux,masks}` и `data/train/bs/{meta.csv,sentinel2_pre,sentinel2_post,sentinel1_pre,sentinel1_post,aux,masks}`. Проверь: 420 AF-чипов × 3 файла, 224 BS-чипа × 6 файлов, все chip_id из meta.csv имеют все файлы. Напиши `scripts/check_data.py`, который это проверяет и печатает сводку.

### 2. `src/firemon/io.py`
- `read_af_chip(data_dir, chip_id, with_mask=True) -> dict` с ключами: `viirs` (8,256,256) float32, `aux` (5,256,256) float32, `mask` (256,256) uint8 или None, `band_names` (список для каждого файла из GDAL_METADATA), `transform`/`crs` (если есть), `meta` (строка meta.csv как dict).
- `read_bs_chip(data_dir, chip_id, with_mask=True) -> dict`: `s2_pre`, `s2_post` (10,512,512) uint16, `s1_pre`, `s1_post` (2,512,512) int16, `aux` (3,512,512) int16, `mask` (512,512) uint8|None, `band_names`, `transform`, `crs`, `meta`.
- Всегда возвращать (C,H,W). NaN в VIIRS оставлять как есть, но дополнительно отдавать `valid` маску. Маска: значение 255 (nodata) → отдельно `mask_valid`.
- Список чипов: `list_chips(data_dir, kind) -> list[str]` из meta.csv (тестовый meta без масок — учесть, что `masks/` может отсутствовать).
- Функции должны быть быстрыми: открывать файл один раз, читать все каналы одним вызовом. Замерь время чтения одного AF и одного BS чипа, запиши в отчёт задачи.

### 3. `src/firemon/rle.py`
- `rle_encode(mask: np.ndarray[bool|uint8]) -> str` по правилам CLAUDE.md §5: нумерация с 1, row-major, пары `start length`, смежные серии слиты, пустая маска → `""`.
- `rle_decode(rle: str, shape: tuple[int,int]) -> np.ndarray[uint8]`.
- `validate_rle(rle: str, shape)` — проверяет возрастание, непересечение, несоприкосновение, границы; кидает `ValueError` с внятным текстом.
- Реализация векторизованная (numpy diff по flatten), без питоновских циклов по пикселям.
- **Когда появится `baseline/`** с официальным RLE-скриптом — добавь тест, что наш encode совпадает посимвольно с официальным на 50 случайных масках и что decode(official_encode(m)) == m. Пока baseline нет — сделай тест-заглушку с `pytest.skip` и пометку в отчёте.

### 4. `src/firemon/metrics.py`
- `af_f1(preds: dict[chip_id -> mask], gts: dict) -> dict(tp, fp, fn, precision, recall, f1)` — микро по всем пикселям всех чипов, с учётом валидности (пиксели nodata=255 в эталоне исключаются — уточни, как делает официальный скрипт, когда появится; по умолчанию исключать).
- `bs_metrics(preds, gts) -> dict(iou_burn, iou_1, iou_2, iou_3, miou_sev)` — микро, правило 0/0 → 1.
- `score(af_f1, iou_burn, miou_sev)` = 0.35/0.35/0.30.
- `evaluate_submission(submission_csv, data_dir_with_masks, sample_submission_csv) -> dict` — декодирует RLE и считает всё выше; используется для OOF-оценок и для проверки, что цепочка «маска → RLE → csv → декод → метрика» не теряет ничего.
- Тесты: ручные примеры 3×3 с известными TP/FP/FN; случай пустого класса; сверка с официальным скриптом (skip, пока нет).

### 5. `src/firemon/submission.py`
- `build_submission(af_preds: dict[str, np.ndarray], bs_preds: dict[str, np.ndarray], sample_submission_csv, out_csv)` — формирует файл строго по шаблону (тот же набор пар chip_id/class_id, для BS разбивает маску 0..3 на три бинарные), QUOTE_ALL для колонки rle.
- `validate_submission(csv_path, sample_submission_csv, shapes: dict[chip_id -> (H,W)])` — все проверки из ТЗ: 447 строк, набор пар совпадает, нет NaN, RLE валиден, классы одного чипа не пересекаются. Возвращает список ошибок; `python -m firemon.submission --validate …` как CLI.
- Пока sample_submission.csv нет — сгенерируй `baseline/sample_submission_FAKE.csv` из train chip_id (только для тестов, с пометкой), чтобы вся цепочка была протестирована; потом заменить на настоящий.

### 6. `src/firemon/splits.py`
Public/private тест разделён по территории И по времени, поэтому обычный KFold завысит оценку.
- AF: `make_af_folds(meta, n_folds=5, seed=42)` — GroupKFold, группа = тайл `(epsg, x_min, y_min)` (70 уникальных). Дополнительно `make_af_temporal_split(meta)`: train = годы 2019–2023, val = 2024 (проверь, что в val есть и позитивные, и негативные чипы, и ночные пролёты).
- BS: `make_bs_folds(meta, n_folds=5, seed=42)` — группа = пространственный блок: округли `x_min`, `y_min` до 50 км внутри epsg → block_id; проверь, что 224 fire_event_id распределены по фолдам без утечек (один пожар — один фолд, это гарантирует блок). Дополнительно `make_bs_temporal_split`: val = 2024, train = остальное.
- Сохрани фолды в `configs/folds_af.csv`, `configs/folds_bs.csv` (chip_id, fold, temporal_split) — все агенты моделей ОБЯЗАНЫ использовать эти файлы, чтобы метрики были сравнимы.

### 7. `src/firemon/utils.py`
`seed_everything(seed)`, `Timer` контекст-менеджер, `get_logger`, `load_config(yaml)`.

### 8. Baseline-заглушки для сквозной проверки
- `inference.py` — уже сейчас должен работать сквозно на «фейковом» тесте: аргументы `--data-dir`, `--output`, опционально `--sample-submission` (по умолчанию `<data-dir>/sample_submission.csv`, fallback `baseline/sample_submission.csv`), `--config configs/inference.yaml`. Пока модели нет — простейшие правила: AF: `I4 > 320 K и (I4−I5) > 15 K` (грубо), BS: `dNBR > 0.1` → класс 1/2/3 по порогам 0.1/0.27/0.44 (USGS). Это временно; агенты 3 и 4 заменят. Скрипт должен обрабатывать любые чипы (в т.ч. с NaN и облаками) без падения, печатать время работы и завершаться с кодом 0.
- `scripts/make_fake_test.py` — копирует 20 AF + 10 BS train-чипов в `data/test_fake/` в тестовой структуре (без масок, meta только с тестовыми колонками) и создаёт `data/test_fake/sample_submission.csv`. На нём прогони `inference.py` → `validate_submission` → `evaluate_submission` (маски берутся из train). Зафиксируй время инференса.

### 9. Тесты (`tests/`)
`test_io.py` (формы, dtype, band_names, NaN), `test_rle.py` (roundtrip на 200 случайных масках разных размеров + крайние случаи: пустая, полная, одна строка, последняя ячейка), `test_metrics.py`, `test_submission.py` (валидатор ловит каждую ошибку из ТЗ: лишняя строка, пропущенная пара, пересечение классов, выход за границы, NaN), `test_splits.py` (нет утечки групп между фолдами), `test_inference_smoke.py` (на 2 чипах из test_fake, <30 с).

### 10. README.md (черновик)
Разделы: описание, структура, установка (PowerShell-команды), подготовка данных, запуск инференса, запуск обучения (заглушка), сервис (заглушка), тесты. Будет дополняться другими агентами.

## Критерии готовности
- `python scripts/check_data.py` — ок; `python -m pytest -q` — зелёный;
- `python inference.py --data-dir data/test_fake --output out/submission_fake.csv` → код 0, валидный файл, время напечатано;
- `configs/folds_af.csv`, `configs/folds_bs.csv` существуют;
- первый коммит сделан; отчёт `reports/tasklog/01_infrastructure.md` с: временем чтения чипов, временем инференса заглушки, версией torch/CUDA и именем GPU, открытыми вопросами.

## Когда появятся baseline и тест (сделать в этой же задаче, если успеют прийти, иначе — отдельным коротким промптом 1b)
1. Положить в `baseline/` и `data/test/`; `scripts/check_data.py --split test` (269 чипов, 447 строк в sample_submission).
2. Прочитать baseline-код целиком, написать `reports/baseline_review.md`: что делает пайплайн AF и BS, какие пороги, что не учитывает (техноген, тип покрова, облака), запустить baseline на train с нашей метрикой (OOF не нужен — просто прогон) и записать его Score, F1_af, IoU_burn, mIoU_sev в `reports/experiments.md` первой строкой как точку отсчёта.
3. Снять `pytest.skip` с тестов совместимости RLE и метрики, добиться совпадения.
4. Заменить фейковый шаблон на настоящий, прогнать `inference.py` на реальном `data/test`, провалидировать.
