# Аудит готовности к запуску на инфраструктуре организаторов (Linux + H100)

Дата: 2026-09-19, перед тегом `v1.0-final`. Машина проверки: Windows 10, i5-13600KF (20 потоков), RTX 4070 12 ГБ,
Docker Desktop 29.2 (WSL2). **Самой H100 и «голого» Linux-хоста с GPU у нас нет** — проверено всё, что воспроизводится без них: Linux —
в Docker (`python:3.12-slim`), поведение на Ampere/Hopper — по коду (TF32, cuBLAS, cuDNN) и по побайтному совпадению GPU/CPU.

**Итог: `submissions/submission_v5_final.csv` побайтно НЕ изменился** (sha256 `784fe474…3afc`) — после каждого исправления
`inference.py` давал тот же файл на GPU, CPU и в Docker. Переотправка на лидерборд не нужна. Добавлено 9 тестов
(`tests/test_portability.py`).

## Таблица «риск → проверка → результат → что исправлено»

| # | Риск | Проверка | Результат | Что исправлено |
|---|---|---|---|---|
| 1 | **Сеть**: smp/timm качают веса энкодера (`encoder_weights="imagenet"`) и падают офлайн | код: `BSPredictor._load_unets` → `build_unet(..., pretrained=False)` → `encoder_weights=None`; grep по пути инференса на `torch.hub / urllib / requests / http / subprocess / socket`; прогон `docker run --network none` | сетевых вызовов нет; офлайн-прогон: код 0, файл = финал | ничего; добавлен тест `test_inference_path_has_no_network_or_pretrained_downloads` |
| 2 | `torch.use_deterministic_algorithms(True)` + cuBLAS ≥ 10.2 без `CUBLAS_WORKSPACE_CONFIG` → RuntimeError | grep: на пути инференса `use_deterministic_algorithms` не вызывается | риска на инференсе нет | профилактически `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` в `_load_unets` до импорта torch / первого CUDA-вызова; значение пользователя не перетирается |
| 3 | **TF32 на H100**: включён по умолчанию в cuDNN-свёртках → другие вероятности → другой файл | код | раньше TF32 выключался только в ветке `use_cuda and not half` | теперь выключается **безусловно**: `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`; дополнительно `cudnn.benchmark = False`. На RTX 4070 файл не изменился |
| 4 | `torch.load(..., weights_only=True)` не читает наши чекпойнты | загрузка всех 4 `weights/*.pt` с `weights_only=True` | OK: в чекпойнтах только тензоры, str, list, float, None | `weights_only=False` → `True` (код из файла весов не исполняется) |
| 5 | CUDA недоступна / старый драйвер → падение | `CUDA_VISIBLE_DEVICES=-1` (в PowerShell пустая строка переменную удаляет, поэтому −1); unit-тест с `is_available() = True` и падающим первым CUDA-вызовом | код 0, одна строка `WARNING … CUDA недоступна (нет GPU или старый драйвер) — … на CPU`; файл на `data/test_fake` побайтно равен GPU-прогону | новая `firemon.utils.cuda_usable()` (is_available + пробное выделение, любая ошибка → False) в `RunContext.torch_device()` и `BSPredictor` |
| 6 | **Потоки — константа 8**: на 64 ядрах простой, на 2 — лишние переключения | `--workers 2/4/8/16/20` на GPU; CPU-прогон | GPU: 27.0 / 23.2 / 23.7 / 24.1 / 24.1 с — плато с 4 потоков, деградации на 16–20 нет. CPU: 68.8–69.2 с (8 потоков) → **56.3–57.7 с** (16). Файл одинаков при любом числе | `workers: auto`, `num_threads: auto`, `lgbm_threads: auto` → `auto_threads()` = min(доступные ядра, 16) |
| 7 | **Docker `--cpus 2`: `os.cpu_count()` = 20 (ядра хоста)** → 16 потоков на 2 ядрах | прогон в контейнере с `--cpus 2` | до исправления **не завершился за 11 мин** (остановлен); после — **169.9 с**, код 0, файл = финал | `available_cpus()`: `os.cpu_count()` ∩ `sched_getaffinity` ∩ квота cgroup v2 `cpu.max` / v1 `cfs_quota_us`; вне Linux — no-op (тест `test_cgroup_quota_limits_threads`) |
| 8 | Пиковая RAM ≥ 16 ГБ, память GPU ≥ 8 ГБ | опрос дерева процессов и `nvidia-smi` каждые 250 мс во время полного прогона, 16 потоков | RAM: пик рабочего набора **8.3 ГБ** (GPU-режим) / 7.9 ГБ (CPU-режим), private 12.8 / 9.4 ГБ; **GPU 3.05 ГБ** (1374 → 4425 МиБ) | в README было «RAM ≤ 2 ГБ» (цифра времён BS-заглушки) — исправлено |
| 9 | `win_path()` и прочие Windows-ветки на Linux | тесты `tests/test_io.py` (monkeypatch `os.name`) + все прогоны в Docker | no-op на Linux, OK | — |
| 10 | Регистр имён каталогов/файлов (`sentinel2_pre`, `VIIRS_I1-I5`, `Sentinel-2_pre`), переводы строк в `weights/*.txt`, `*.json` | прогон в Linux-контейнере на реальном `data/test` | 269 чипов прочитаны, 0 пустых масок из-за ошибок чтения; веса читаются; файл = финал | — |
| 11 | `LANG=C`: кириллица в логах → `UnicodeEncodeError` | контейнер с `-e LANG=C -e LC_ALL=C -e PYTHONIOENCODING=ascii` (жёстче, чем просто LANG=C: образ сам ставит utf-8) | код 0, кириллица заменена на `?`, файл = финал | `get_logger` → `stdout/stderr.reconfigure(errors="replace")`; тест `test_cyrillic_log_survives_ascii_console` |
| 12 | Чужой cwd и абсолютные пути; каталог `--output` не существует; `--data-dir` со слэшем; данные read-only; запись рядом с данными/кодом | один прогон: `-w /tmp --entrypoint python … /app/inference.py --data-dir /data/ --output /out/new/sub/dir/b.csv`, том данных `:ro`, **корневая ФС контейнера `--read-only`** (+ tmpfs `/tmp`), `--network none` | код 0, каталоги созданы, файл = финал; кроме `--output`, ничего не пишется | — |
| 13 | Поиск `sample_submission.csv` | код `find_sample_submission`: `--sample-submission` → `<data-dir>/` → `baseline/`; в Docker `baseline/` нет — берётся из `/data` | OK | — |
| 14 | **Пины не ставятся на Python организаторов** | `pip install --dry-run --ignore-installed --python-version 3.10 / 3.11 / 3.12 / 3.13 --platform linux_x86_64 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --only-binary=:all: -r requirements.txt` (dry-run вместо `pip download`: то же разрешение зависимостей без скачивания 2.5 ГБ torch ×4) | **3.12 — OK, 3.13 — OK, 3.10 и 3.11 — НЕ ставятся** (`numpy==2.5.2`, `pandas==3.0.6` требуют ≥ 3.12). README и `pyproject.toml` утверждали 3.10–3.12 | основные пины не менялись; для 3.10 / 3.11 добавлен отдельный проверенный набор — см. п. 18; README §3: «3.12 — основной, 3.10 / 3.11 — `requirements-py310.txt`, 3.13 — Docker» |
| 15 | torch/CUDA на Linux, драйвер | `requirements.txt`: `torch==2.6.0+cu124` из `https://download.pytorch.org/whl/cu124` (колёса `linux_x86_64`, CUDA-библиотеки внутри) | разрешается для cp312 | README §3: драйвер ≥ 550, системный CUDA Toolkit не нужен, PyPI-torch на Linux уже с CUDA 12.4, автоматический откат на CPU |
| 16 | **Docker GPU** (`Dockerfile.gpu`) | `docker build -f Dockerfile.gpu -t firemon:gpu .` — две попытки | **`Dockerfile.gpu` написан, но не собран из-за нехватки места на машине разработки; проверен CPU-образ.** Первая попытка оборвалась на загрузке базового образа `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (`error reading from server: EOF`), вторая заполнила диск C: до 100 % (виртуальный диск Docker вырос до 66 ГБ), демон Docker перестал отвечать | сборка остановлена и больше не запускалась; демон не чинился, в WSL/Docker ничего не удалялось. Все шаги с CPU-образом (п. 1, 7, 10–13) выполнены до зависания демона — пропущенных Docker-шагов, кроме GPU-образа, нет. Рекомендованный путь для GPU-машины — venv (`requirements.txt`) |
| 17 | Скорость на H100: узкое место — CPU | профиль `--profile` на GPU-прогоне 23.67 с | UNet (GPU) 6.1–6.8 с + загрузка моделей 2.2 с; CPU-этапы (чтение deflate, признаки, стек-LGBM, постобработка, AF, RLE, валидация) ≈ 16.5 с ≈ **70 %** | README §6: ожидание на H100 ≈ 19–22 с (GPU-часть ≈ 2–3 с — оценка, CPU-часть ≈ 17 с при 8–16 ядрах); быстрый режим v4 — запасной вариант |
| 18 | **Python 3.10 / 3.11 без Docker** (следствие п. 14) | `uv python install 3.10` → venv на **3.10.21** → `pip install -r requirements-py310-cpu.txt` (numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-image 0.25.2, scikit-learn 1.7.2, rasterio 1.4.4, tifffile 2025.5.10, imagecodecs 2025.3.30, pyproj 3.7.1, matplotlib 3.10.9; torch 2.6.0, LightGBM 4.7.0, smp 0.5.0, geopandas, shapely — те же) → `python inference.py --data-dir data/test --output out/py310.csv --device cpu` → `compare_submissions.py` → `python -m pytest -q`; разрешение зависимостей `uv pip compile --python-version 3.10 / 3.11 --python-platform linux` для обоих файлов | установка — код 0; инференс — код 0, файл **побайтно равен финалу** (`784fe474…`, 0 отличий; два прогона: 112.0 с холодный, 54.8 с повторный); pytest — код 0 (423 с); Linux 3.10 и 3.11, cu124 и cpu — разрешаются. Вариант cu124 и Python 3.11 проверены только разрешением зависимостей, установка и прогон — на 3.10 + torch CPU (Windows) | добавлены `requirements-py310-base.txt`, `requirements-py310.txt`, `requirements-py310-cpu.txt`; README §3–4; `pyproject.toml` возвращён к `>=3.10,<3.13`; venv удалён |
| 19 | **Холодный старт**: первый запуск после установки медленнее порога 30 с | новый venv ×4, `python -X importtime`, профиль первого и повторных запусков, удаление `__pycache__`, `compileall` (`reports/speed.md` §1d) | старый код: первый запуск **33.87 с**, повторные 23.6 с. Импорты — 4.3 с (torch 1.3, smp + timm 1.4, LightGBM + sklearn 1.0); geopandas / shapely / pyproj / matplotlib на пути инференса не импортируются; байткод ни при чём (pip компилирует site-packages сам, свой код — 0 с разницы); причина — первая загрузка ≈ 3 ГБ бинарных модулей torch / CUDA, целиком в этапе BS | `pipeline.prewarm_torch`: импорт torch / smp и CUDA-контекст в фоне, пока считается AF → первый запуск **27.90 с**, повторные 21.8–23.2 с; после шага «проверка установки» из README первый запуск **23.30 с**; `compileall` — в README и в обоих Dockerfile (образы не пересобирались: демон Docker не работает); файл побайтно тот же; тесты `test_inference_imports_stay_lean`, `test_prewarm_is_optional_and_harmless` |

## Что изменено в коде (все правки минимальные)

- `src/firemon/utils.py`: `auto_threads()`, `available_cpus()`, `_cgroup_cpu_quota()`, `cuda_usable()`, `_safe_stdout()` (вызов в `get_logger`).
- `src/firemon/models/bs/predictor.py`: `CUBLAS_WORKSPACE_CONFIG` через `setdefault`, TF32 и `cudnn.benchmark` выключены безусловно,
  `weights_only=True`, `cuda_usable`, потоки `auto`.
- `src/firemon/models/af/predictor.py`, `src/firemon/pipeline.py`, `inference.py`: потоки через `auto_threads`, устройство через `cuda_usable`.
- `configs/*.yaml`: `workers: auto`, `num_threads: auto`.
- `tests/test_portability.py` — 9 тестов. `pyproject.toml`, `README.md` (§3, §6, §7, §8), `CLAUDE.md`, `reports/speed.md` (§1c).

## Проверка неизменности файла после каждого шага

| Шаг | GPU | CPU | Docker CPU |
|---|---|---|---|
| до аудита (релиз v5) | `784fe474…` ×3 | `784fe474…` ×2 | `784fe474…` |
| TF32 / cuDNN / cuBLAS / `weights_only` / `cuda_usable` + потоки auto | `784fe474…` (+ `--workers` 2/4/8/16/20) | `784fe474…` ×2 | `784fe474…` (обычный; офлайн + LANG=C + read-only) |
| квота cgroup | — (на Windows no-op) | — | `784fe474…` (обычный офлайн; `--cpus 2`) |

## Чего проверить не удалось

- Реальная H100 и Linux-хост с GPU вне Docker: оценки времени на H100 — расчётные.
- GPU-образ Docker (п. 16): `Dockerfile.gpu` написан, но не собран из-за нехватки места на машине разработки; проверен CPU-образ.
- Побайтное совпадение файла между **разными GPU** (RTX 4070 против H100) не проверено; довод в пользу совпадения — fp32 без TF32 и
  автоподбора алгоритмов и то, что GPU-файл побайтно равен CPU-файлу. Если на H100 файл всё же отличится, ожидаемый масштаб — единицы
  пикселей из 23.3 млн (< 1e-6 метрики).
