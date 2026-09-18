# Задача 01c — доработки перед параллельными задачами (2026-09-18)

## Что сделано
1. **`firemon.io.win_path()`** — no-op вне Windows (`os.name != "nt"` → `str(path)`). На Windows: путь приводится к
   абсолютному через `ntpath` (префикс `\\?\` допустим только для абсолютных путей; «/» заменяются на «\»), префикс
   не дублируется, UNC-пути не трогаются. Тесты `test_win_path_is_noop_off_windows`, `test_win_path_prefix_on_windows`
   (monkeypatch `os.name`; благодаря `ntpath` оба теста проходят на любой ОС).
   Grep по `src/`, `scripts/`, `inference.py` (`\\?\`, `os.name`, `sys.platform`, `platform.`): других OS-специфичных
   мест нет; `win_path` вызывается в `io.list_chips`, `io.has_masks`, `scripts/check_data.py`, `scripts/make_fake_test.py`.
2. **`pyproject.toml`**: `requires-python = ">=3.10,<3.13"`. Проверка Linux-колёс — ниже, замен версий не потребовалось.
3. **`metrics.py`**: `ignore_nodata` оставлен (default True); в docstring и CLAUDE.md §3 — «default будет выставлен по
   официальному скрипту метрики после его появления (задача 1b)».
4. **`reports/experiments.md`**, строка №0 — дополнен вывод по AF-заглушке (см. «Отклонение» ниже).
5. **CLAUDE.md**: §4 «Тест» — политика сплитов (temporal val=2024 — основной индикатор private, GroupKFold —
   вспомогательный, улучшение = прирост на обоих, отчитывать обе цифры); §8 — `win_path()` и правило «любая
   OS-специфичная ветка обязана иметь no-op для Linux».

## Отклонение от формулировки задания (п. 4)
В задании вывод содержал «(горячая голая почва, застройка)». Перед записью в оцениваемый журнал проверил на train
(54 201 ложный пиксель заглушки):

| landcover | доля ложных | доля площади |
|---|---|---|
| 40 пашня | 58.5 % | 52.7 % |
| 30 трава | 35.4 % | 20.3 % |
| 10 лес | 2.9 % | 15.3 % |
| 60 голая почва | 1.7 % | 0.1 % (обогащение ×17) |
| 50 застройка | 0.7 % | 1.3 % |
| 90 болото / 80 вода | 0.6 % / 0.3 % | 2.0 % / 7.8 % |

100 % ложных срабатываний — дневные пролёты. То есть это прогретый солнцем открытый грунт полей и степи, застройка
почти не участвует. В журнал записаны измеренные цифры; числа recall 0.91 / precision 0.14 / 86 % — как в задании.
Полезно агенту AF: раздельная логика день/ночь и контекстный фон важнее фильтра по застройке.

## Проверка колёс: Linux x86_64, CPython 3.12
Команда: `pip download -r <requirements без torch/torchvision> --no-deps --platform manylinux2014_x86_64
--platform manylinux_2_28_x86_64 --python-version 3.12 --implementation cp --only-binary=:all:` → exit 0, 22/22.

| Пакет | Колесо | ОК |
|---|---|---|
| numpy 2.5.2 | cp312 manylinux_2_27/2_28 | да |
| pandas 3.0.6 | cp312 manylinux_2_24/2_28 | да |
| scipy 1.18.1 | cp312 manylinux_2_27/2_28 | да |
| scikit-image 0.26.0 | cp312 manylinux_2_24/2_28 | да |
| scikit-learn 1.9.1 | cp312 manylinux_2_27/2_28 | да |
| lightgbm 4.7.0 | py3 manylinux_2_27/2_28 | да |
| segmentation-models-pytorch 0.5.0 | py3-none-any | да |
| albumentations 2.0.8 | py3-none-any | да |
| rasterio 1.5.1 | cp312 manylinux_2_28 | да |
| tifffile 2026.9.15 | py3-none-any | да |
| imagecodecs 2026.8.16 | cp312-abi3 manylinux_2_28 | да |
| geopandas 1.1.4 | py3-none-any | да |
| shapely 2.1.2 | cp312 manylinux2014 | да |
| pyproj 3.8.0 | cp312 manylinux_2_28 | да |
| fastapi 0.141.1 / uvicorn 0.53.0 / httpx 0.28.1 | py3-none-any | да |
| PyYAML 6.0.3 | cp312 manylinux2014/2_28 | да |
| tqdm 4.70.1 / pypdf 6.19.0 / pytest 9.1.1 | py3-none-any | да |
| matplotlib 3.11.2 | cp312 manylinux2014 | да |
| torch 2.6.0 / torchvision 0.21.0 | исключены из проверки (ставятся с индекса PyTorch CUDA/CPU) | — |

Замечания: (а) проверялись прямые зависимости (`--no-deps`), транзитивные резолвятся pip на месте; (б) 9 пакетов
имеют только теги manylinux_2_24…2_28 → нужен glibc ≥ 2.28 (Debian 10+/Ubuntu 20.04+; `python:3.12-slim` подходит,
CentOS 7 — нет). Для агента Docker: базовый образ не старше этого.

## Тесты
`python -m pytest -q` → 58 passed, 2 skipped (совместимость с официальными RLE/метрикой — ждут baseline).
`python scripts/check_data.py` → «ВСЁ ОК».

## Что менять в CLAUDE.md
Уже внесено (§3, §4, §8). Открытых вопросов по этой задаче нет; вопросы задачи 1b остаются.
