# Задача 01 — Инфраструктура (2026-09-18)

## Что сделано
- Репозиторий, структура каталогов по CLAUDE.md §8, `.gitignore`, `pyproject.toml` (пакет `firemon`, `pip install -e .`).
- `docs/case.pdf`, `docs/criteria.pdf` + текстовые версии (pypdf); граница AOI → `service/data/aoi/`.
- Окружение `.venv`: Python 3.12.7, **torch 2.6.0+cu124, CUDA 12.4, `cuda.is_available() = True`, GPU NVIDIA GeForce RTX 4070 (12 ГБ)**; rasterio 1.5.1 (GDAL 3.12.4) поставился из pip без проблем. `requirements.txt` (прямые зависимости, точные версии) + `requirements.lock.txt` (полный freeze).
- Данные распакованы в `data/train`; `scripts/check_data.py` → «ВСЁ ОК»: 420 AF × 3 файла, 224 BS × 6 файлов, все chip_id из meta.csv укомплектованы, лишних файлов нет.
- `src/firemon`: `io.py`, `rle.py`, `metrics.py`, `submission.py` (+CLI), `splits.py` (+CLI), `utils.py`, заглушки `models/af/rules.py`, `models/bs/rules.py`.
- `configs/folds_af.csv`, `configs/folds_bs.csv`, `configs/inference.yaml`.
- `inference.py` (сквозной, с автовалидацией результата), `train.py` (заглушка), `scripts/make_fake_test.py`.
- Тесты: 56 passed, 2 skipped (совместимость с официальными RLE/метрикой — ждут `baseline/`).
- README (черновик), `reports/experiments.md` (строка №0 — заглушка как точка отсчёта), дополнен CLAUDE.md.

## Как запустить
```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe scripts\check_data.py
.\.venv\Scripts\python.exe scripts\make_fake_test.py
.\.venv\Scripts\python.exe inference.py --data-dir data/test_fake --output out/submission_fake.csv
.\.venv\Scripts\python.exe -m firemon.submission --validate out/submission_fake.csv --sample data/test_fake/sample_submission.csv --data-dir data/test_fake
.\.venv\Scripts\python.exe -m firemon.metrics --submission out/submission_fake.csv --masks-dir data/train --sample-submission data/test_fake/sample_submission.csv
.\.venv\Scripts\python.exe -m pytest -q
```

## Замеры
| Что | Время |
|---|---|
| Чтение AF-чипа (viirs + aux + mask, rasterio) | 27 мс холодное / **17 мс** тёплое |
| Чтение BS-чипа (5 слоёв + mask) | 96 мс холодное / **70 мс** тёплое |
| `inference.py` (заглушка) на test_fake, 20 AF + 10 BS | **1.26 с** полное (AF 0.17 с, BS 0.54 с, сборка+валидация 0.13 с) |
| `inference.py` (заглушка) на всём train, 644 чипа | **21.6 с** (AF 5.0 с, BS 13.4 с, сборка+валидация 2.8 с) → ≈9 с на 269 тестовых чипов |

Запас по лимиту 30 с есть, но основное время — чтение/распаковка TIFF; для моделей, которым нужны все слои BS, стоит читать в пуле потоков.

## Метрики заглушки (не модель — точка отсчёта)
| Выборка | F1_af | IoU_burn | mIoU_sev | Score |
|---|---|---|---|---|
| весь train (420 AF + 224 BS) | 0.2435 (P 0.141, R 0.911) | 0.3706 | 0.3129 | **0.3088** |
| test_fake (20 + 10) | 0.3296 | 0.4226 | 0.2984 | 0.3528 |

Цепочка «маска → RLE → csv → декод → метрика» проверена тестами (roundtrip побитовый).

## Фолды
- AF: 70 тайлов → 5 фолдов по 84 чипа, 13–15 тайлов, 1920–1953 px огня в каждом (баланс по числу чипов и пикселей огня); temporal: train 342 (2019–2023) / val 78 (2024: 53 позитивных, 25 негативных, 32 ночных).
- BS: 96 блоков 50×50 км → 5 фолдов по 44–45 чипов, один пожар — один фолд; temporal: train 142 / val 82 (2024).
- Вместо `sklearn.GroupKFold` (не принимает seed и не балансирует класс) — собственное жадное детерминированное распределение групп; утечек нет (тест).

## Находки / отклонения от промпта
1. **`aux` — зарезервированное имя в Windows 10.** tar распаковал, файлы читаются, но каталог не открывается обычными средствами (PowerShell, `os.listdir`, `abspath`). Решение: `firemon.io.win_path()` (`\\?\`-префикс) в `list_chips`/`check_data`/`make_fake_test`. Внесено в CLAUDE.md и README.
2. Python 3.10/3.11 на машине нет → venv на 3.12.7 (удовлетворяет «3.10+»).
3. Стандартный модуль `csv` падал на длинных RLE (>131072 символов) — лимит поднят в `submission.py`.
4. `seed_everything(with_torch=False)` в инференсе: импорт torch стоит секунды, заглушкам он не нужен.
5. В архиве есть пустые лишние каталоги (`af/images`, `bs/pre`, `bs/post`, `bs/sar_pre`, `bs/sar_post`) — игнорируются.
6. В train годы 2019–2024, в README к AOI заявлены сезоны 2019–2025 → private, вероятно, включает 2025; временной сплит важен.
7. `baseline/sample_submission_FAKE.csv` создан (1092 строки по train) — только для тестов/OOF; сам каталог в .gitignore.

## Открытые вопросы (закрыть в задаче 1b, когда придут baseline и тест)
- Как официальная метрика обращается с nodata=255 в эталоне (сейчас исключаем; переключатель `ignore_nodata`).
- Побитовая совместимость RLE и метрики с официальными скриптами — тесты-заглушки `test_official_rle_compat`, `test_official_metric_compat` сейчас skip и **намеренно упадут**, как только в `baseline/` появятся `.py` — чтобы их не забыли подключить.
- Формат chip_id в тесте (предполагаем префиксы `AF`/`BS` — на них опирается `chip_kind`) и имена файлов `{chip_id}_{суффикс}.tif`.
- `check_data.py --split test`, `reports/baseline_review.md`, строка baseline в `experiments.md`, замена фейкового шаблона.

## Что изменено в CLAUDE.md
Добавлен подраздел «Что уже есть (задача 01)» в §8: окружение, ловушка `aux`, API пакета, времена чтения, наблюдение про 2025 год, лимит csv, фейковый тест.
