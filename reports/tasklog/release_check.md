# Проверка релиза (`python scripts/check_release.py`)

Сформировано: 2026-09-19 11:58 UTC · коммит `3a860a0` · ветка `main`

## 1. Файлы, на которые ссылаются документы

| Документ | путей | в git | вне git по замыслу | НЕ НАЙДЕНО |
|---|---|---|---|---|
| `README.md` | 50 | 46 | 4 | 0 |
| `handoff/HANDOFF.md` | 30 | 30 | 0 | 0 |
| `reports/report.md` | 40 | 40 | 0 | 0 |

Все пути из документов либо есть в git, либо относятся к тому, что в git не входит по замыслу (`data/`, `baseline/`, `out/`, `submissions/`, каталоги весов экспериментов, `service/data/firemon.gpkg`).

## 2. Обязательные файлы (инференс, сервис, отчётность)

Проверено 30 файлов: **все в git**.
Снимки слоёв сервиса в git: 3 (`burn_scars.geojson.gz`, `chips.geojson.gz`, `hotspots.geojson.gz`).

## 3. Размеры

- отслеживаемых файлов: **376**, суммарно **269.2 МиБ**;
- файлов ≥ 100 МБ (жёсткий лимит GitHub): **0**; файлов 50–100 МБ (GitHub предупреждает при push): 0;
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
| `presentation/deck.pptx` | 3.6 |
| `reports/report.pdf` | 3.1 |
| `handoff/report.pdf` | 3.1 |

| Каталог | МиБ |
|---|---|
| `weights` | 204.8 |
| `reports` | 29.6 |
| `service` | 17.4 |
| `handoff` | 5.9 |
| `presentation` | 5.2 |
| `submissions` | 2.9 |
| `docs` | 1.9 |
| `tests` | 0.5 |
| `eda` | 0.3 |
| `src` | 0.2 |
| `scripts` | 0.2 |
| `(корень)` | 0.1 |
| `prompts` | 0.1 |
| `configs` | 0.0 |

Каталог `.git`: **251.2 МиБ** (после `git gc`).

## 4. Рабочая копия

`git status` — **чисто**.

Теги на HEAD: —.

## 5. Как выложить (удалённого репозитория пока нет)

```
git remote add origin <url>
git push -u origin main --tags
```

Все 376 файлов меньше 100 МБ — Git LFS не нужен. Файлы 50–100 МБ GitHub принимает с предупреждением; у нас таких 0 (веса UNet по 47 МиБ = 49.3 МБ — ниже и этого порога). После push проверить на сайте, что четыре `weights/*.pt` открываются как бинарные файлы по ~47 МБ, а не как указатели LFS.

## Итог

**Проблем нет.**
