# Проверка релиза (`python scripts/check_release.py`)

Сформировано: 2026-09-19 16:23 UTC · коммит `7816dc8` · ветка `main`

## 1. Файлы, на которые ссылаются документы

| Документ | путей | в git | вне git по замыслу | НЕ НАЙДЕНО |
|---|---|---|---|---|
| `README.md` | 29 | 26 | 3 | 0 |
| `reports/report.md` | 44 | 44 | 0 | 0 |
| `reports/service.md` | 14 | 14 | 0 | 0 |
| `reports/bs_module.md` | 46 | 46 | 0 | 0 |
| `reports/af_module.md` | 10 | 10 | 0 | 0 |

Все пути из документов либо есть в git, либо относятся к тому, что в git не входит по замыслу (`data/`, `baseline/`, `out/`, `submissions/`, каталоги весов экспериментов, `service/data/firemon.gpkg`).

## 2. Обязательные файлы (инференс, сервис, отчётность)

Проверено 31 файлов: **все в git**.
Снимки слоёв сервиса в git: 3 (`burn_scars.geojson.gz`, `chips.geojson.gz`, `hotspots.geojson.gz`).

## 3. Размеры

- отслеживаемых файлов: **318**, суммарно **253.9 МиБ**;
- файлов ≥ 100 МБ (жёсткий лимит git-хостингов): **0**; файлов 50–100 МБ (хостинги предупреждают при push): 0;
- самый большой файл: `weights/bs_unet_r34_v4_s075.pt` — 47.0 МиБ.

| Файл | МиБ |
|---|---|
| `weights/bs_unet_r34_v4_s075.pt` | 47.0 |
| `weights/bs_unet_r34_v4_s050.pt` | 47.0 |
| `weights/bs_unet_r34_v4.pt` | 47.0 |
| `weights/bs_unet_r34_cedice.pt` | 47.0 |
| `service/data/burn_scars.geojson.gz` | 15.0 |
| `reports/figures/eda_bs_04_false_positives.png` | 5.5 |
| `weights/bs_lgbm.txt` | 4.6 |
| `submissions/submission_v5_final.csv` | 2.9 |
| `reports/figures/bs_error_cases.png` | 2.6 |
| `weights/bs_stack_ms.txt` | 2.5 |

| Каталог | МиБ |
|---|---|
| `weights` | 204.8 |
| `reports` | 27.4 |
| `service` | 17.4 |
| `submissions` | 2.9 |
| `tests` | 0.5 |
| `eda` | 0.3 |
| `src` | 0.2 |
| `scripts` | 0.2 |
| `(корень)` | 0.1 |
| `configs` | 0.0 |

Каталог `.git`: **267.5 МиБ** (после `git gc`).

## 3a. Состав: репозиторий = код + документация к коду

Презентация, пакет защиты, журналы задач, ТЗ организаторов, PDF — вне git (пакет артефактов): **в git их нет**.

## 4. Рабочая копия

`git status` — **чисто**.

Теги на HEAD: `v1.0-final`.

## 5. Как выложить (удалённый репозиторий не настроен — площадка выбирается вручную)

```
git remote add origin <url>
git push -u origin main --tags
```

Все 318 файлов меньше 100 МБ — Git LFS не нужен. Файлы 50–100 МБ хостинги обычно принимают с предупреждением; у нас таких 0 (веса UNet по 47 МиБ = 49.3 МБ — ниже и этого порога). После push проверить на сайте, что четыре `weights/*.pt` открываются как бинарные файлы по ~47 МБ, а не как указатели LFS.

## Итог

**Проблем нет.**
