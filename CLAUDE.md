# CLAUDE.md — КосмоХакатон 2026, кейс «Мониторинг природных пожаров»

Этот файл читает КАЖДЫЙ агент Claude Code, работающий в репозитории. Прочитай целиком перед любой задачей.
Полное ТЗ и критерии лежат в `docs/case.pdf` и `docs/criteria.pdf` (текстовые версии `docs/case.txt`, `docs/criteria.txt`).

## 1. Цель проекта

Сдать организаторам за 24 часа полностью рабочее решение из 5 элементов:
1. **Модуль AF (Active Fire)** — по AF-чипу VIIRS 256×256 @375 м выдаёт бинарную маску «пиксель природного горения» (1) / всё остальное, включая техногенные термоаномалии (0).
2. **Модуль BS (Burn Severity)** — по BS-чипу 512×512 @20 м (Sentinel-2 pre/post, Sentinel-1 pre/post, DEM/slope/landcover) выдаёт маску 0 (фон) / 1 (слабая) / 2 (средняя) / 3 (сильная степень поражения).
3. **submission.csv** в формате RLE, формируемый командой  
   `python inference.py --data-dir /path/to/test --output /path/to/submission.csv` (код выхода 0, без интерактива).
4. **Информационно-аналитический сервис** — веб-приложение + REST API: по полигону/bbox и интервалу дат возвращает карту термоточек, векторные контуры гарей (GeoJSON/Shapefile) и справку с площадью в га по трём степеням.
5. **Репозиторий, отчёт (`reports/report.md`), презентация (8–12 слайдов)**.

## 2. Как оценивают (100 баллов) — на что тратить время

| Раздел | Баллы | Замечание |
|---|---|---|
| Метрика на private-тесте | 40 | `Score = 0.35·F1_af + 0.35·IoU_burn + 0.30·mIoU_sev`, микро-усреднение по пулу пикселей всех чипов. Балл = линейная нормализация между худшей и лучшей командой. |
| Отчёт и исследование | 20 | EDA с выводами «наблюдение → решение» (4), разбор baseline (4) — организаторы baseline не выдали, поэтому разбирается НАШ baseline-правило (заглушка AF `I4>320 K & I4−I5>15 K`, BS dNBR 0.10/0.27/0.44; `reports/baseline_review.md`), эксперименты/абляции с фиксированной валидацией (4), инженерные решения (4), выводы (4). |
| Код, воспроизводимость | 12 | README с точными командами (5), структура/читаемость (4), корректный submission (3). Повторный запуск — метрика ±0.005. |
| Сервис | 12 | Работает + REST (4), карта + выгрузка векторов с атрибутами id/класс/площадь (4), справка (4). |
| Презентация | 8 | |
| Скорость инференса | 8 | **<30 с — 8 баллов**, ≤5 мин — 6, ≤12 мин — 4, ≤30 мин — 2. Замер от старта `inference.py` до готового файла, включая загрузку весов. |

Следствия: (а) 60 баллов не про модель — отчёт, код и сервис делаются с тем же приоритетом; (б) инференс должен быть быстрым — лёгкие модели, векторизованный препроцессинг, никакого лишнего I/O; (в) любой эксперимент фиксируется в `reports/experiments.md` (конфиг, seed, схема валидации, OOF-метрики) — это прямые баллы отчёта.

## 3. Метрика — точные формулы

Пиксели всех чипов объединяются в общий пул (микро-усреднение):
```
precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R)
IoU(k)   = TP(k)/(TP(k)+FP(k)+FN(k))
F1_af    — F1 по пикселям класса 1 на AF-чипах
IoU_burn — IoU бинарной маски (pred>0 vs gt>0) на BS-чипах
mIoU_sev = (IoU(1)+IoU(2)+IoU(3))/3 по BS-чипам
Score    = 0.35*F1_af + 0.35*IoU_burn + 0.30*mIoU_sev
```
Если знаменатель 0 (класса нет ни в эталоне, ни в предсказании) — величина = 1. Если класс есть в эталоне, но не в предсказании — IoU = 0.
Пиксели эталона с nodata=255: в `src/firemon/metrics.py` есть переключатель `ignore_nodata` (default True — исключаются).
Официальный скрипт метрики организаторы НЕ выдают: реализация `src/firemon/metrics.py` — строго по тексту ТЗ и является окончательной (ручные примеры в `tests/test_metrics.py`); расхождения можно заметить только по лидерборду.

## 4. Данные

Расположение: `data/train/` (распакованный `fire-train-renamed.tar`), `data/test/` (распакованный `fire-test-renamed.tar`, раскладка отличается — см. «Тест»), `baseline/` (только копия `sample_submission.csv` из теста; baseline-пайплайн и скрипты RLE/метрики организаторы НЕ выдают). Данные в git НЕ коммитить (`.gitignore`).

```
data/train/
  af/ meta.csv  viirs/AF_tr_000001_VIIRS_I1-I5.tif  aux/AF_tr_000001_AUX.tif  masks/AF_tr_000001_MASK.tif   (420 чипов)
  bs/ meta.csv  sentinel2_pre/BS_tr_000001_Sentinel-2_pre.tif  sentinel2_post/...  sentinel1_pre/BS_tr_000001_Sentinel-1_pre.tif
      sentinel1_post/...  aux/BS_tr_000001_AUX.tif  masks/BS_tr_000001_MASK.tif   (224 чипа)
```
Все GeoTIFF, компрессия deflate, каналы описаны в GDAL_METADATA (band descriptions). Читать через `rasterio` (предпочтительно) или `tifffile` (внимание: `tifffile.asarray()` для многоканальных файлов может вернуть (H, W, C) — всегда приводить к (C, H, W)).

### AF-чип, 256×256, 375 м, EPSG 32637/32638, тайлы 96×96 км
- `VIIRS_I1-I5.tif` — 8 каналов float32, nodata=NaN: `I1` (640 нм, отражение 0–1), `I2` (865 нм), `I3` (1610 нм, SWIR, признак солнечного блика), `I4` (3.74 мкм, яркостная температура K — ГЛАВНЫЙ канал; номинальное насыщение ~367 K в данных НЕ встречается: max I4 в train 358.2 K), `I5` (11.45 мкм, яркостная T K, фон), `solar_zenith` (°; >90 = ночь; ночью I1–I3 — полностью NaN, а не «шум»: признаки по ним ночью недоступны), `sensor_zenith` (°), `valid` (0/1).
- `AUX.tif` — 5 каналов float32: `landcover` (ESA WorldCover v200: 10 деревья, 20 кустарник, 30 трава, 40 пашня, 50 застройка, 60 голая почва, 70 снег, 80 вода, 90 болото, 95 мангры, 100 мох), `dem` (м), `t2m` (**в кельвинах**, 276–309 K в train, а не °C; ERA5-Land), `rh2m` (%), `wind_speed` (м/с).
- `MASK.tif` — uint8, 0/1, nodata 255.
- Статистика train: 420 чипов (296 с огнём, 124 без), 9 725 пикселей огня = 0.035 % площади, медиана 21 px на позитивный чип, max 243. Спутники SNPP 280 / NOAA20 123 / NOAA21 17. 293 дневных (8–11 UTC) и 127 ночных (21–00 UTC) пролётов. 70 уникальных тайлов, каждый снят многократно. Месяцы: апрель 128, август 111, сентябрь 94, октябрь 47, остальные мало. 14 чипов с valid_frac<1 (min 0.46). **NaN bow-tie:** кроме того, на 223 чипах (53 %) есть NaN в I4/I5 при `valid=1` и `valid_frac=1.0` — 7.65 % пикселей train, строки bow-tie deletion (0 % при sensor_zenith < 36°, дальше плато ~8 % и ~22 %); 11 % пикселей огня — прямые соседи NaN. Правила: на NaN предсказывать 0 (`chip["valid"]` их уже исключает); любые свёртки/оконные статистики нормировать на число валидных соседей (не заполнять NaN нулём и не давать NaN расползаться по окну).

### BS-чип, 512×512, 20 м, EPSG 32637/32638, покрытие 10.24 км
- `Sentinel-2_pre.tif` / `_post.tif` — 10 каналов uint16, отражение L2A ×10000: `B2,B3,B4` (видимые), `B5,B6,B7` (red edge), `B8A` (NIR 865), `B11` (SWIR 1610), `B12` (SWIR 2190), `SCL` (0 nodata, 1 saturated, 2 dark, 3 тень облака, 4 растительность, 5 голая почва, 6 вода, 7 unclassified, 8 облако средн., 9 облако выс., 10 cirrus, 11 снег).
- `Sentinel-1_pre.tif` / `_post.tif` — 2 канала int16: `VV`, `VH` (σ⁰ в дБ ×100). **12.8 % пикселей S1 — нули = нет съёмки** (31 чип пуст более чем наполовину): 0 — это nodata, а не 0 дБ; нужен признак «S1 есть/нет», разности считать только там, где есть pre и post.
- `AUX.tif` — 3 канала int16: `dem`, `slope`, `landcover` (те же коды WorldCover). Экспозиции склона в файле НЕТ, хотя ТЗ её упоминает. `slope` целочисленный и **константа на 175 из 224 чипов** (равнина) — как признак почти бесполезен.
- `MASK.tif` — uint8 0..3; nodata 255 объявлен, но в train-масках BS значения 255 **нет ни в одном пикселе** (переключатель `ignore_nodata` на BS-метрику train не влияет).
- Статистика train: 224 чипа = 224 разных `fire_event_id`. Гарь занимает в среднем 10 % чипа (медиана 7.5 %, max 56 %), нет чипов без гари. Классы по пикселям: 40.0 / 37.1 / 22.9 %. Разрыв pre→post 10–50 дней (медиана 15). cloud_frac: медиана 1.6 %, 24 чипа >25 %, max 48.5 %. valid_frac min 0.52. Годы: 2020 (71) и 2024 (82) доминируют, месяцы июль–сентябрь.

### meta.csv
Train: `chip_id, kind, fire_event_id, region(пусто), epsg, x_min, y_min, x_max, y_max, width, height, gsd, acq_datetime, satellite, date_pre, date_post, s1_date_pre, s1_date_post, valid_frac, cloud_frac, landcover_top(пусто), n_fire_px, burn_area_ha, sev1_px, sev2_px, sev3_px`.
Test: только `chip_id, kind, width, height, gsd, valid_frac, cloud_frac` (для AF cloud_frac = nan). Геопривязки и дат в тесте НЕТ — модели не должны от них зависеть.

### Тест
Раскладка `data/test/` ОТЛИЧАЕТСЯ от train — один общий `meta.csv` и шаблон в корне, масок нет:
```
data/test/
  meta.csv                 # 269 строк (180 af + 89 bs): chip_id,kind,width,height,gsd,valid_frac,cloud_frac
  sample_submission.csv    # 447 строк; колонка rle пустая и БЕЗ кавычек: `AF_te_000001,1,`
  af/ viirs/AF_te_000001_VIIRS_I1-I5.tif  aux/AF_te_000001_AUX.tif                                  (180 чипов)
  bs/ sentinel2_pre/BS_te_000001_Sentinel-2_pre.tif  sentinel2_post/  sentinel1_pre/  sentinel1_post/  aux/   (89 чипов)
```
`firemon.io` (`read_meta`, `list_chips`, `read_*_chip`) поддерживает обе раскладки: сначала `<data-dir>/<kind>/meta.csv`, иначе `<data-dir>/meta.csv` с фильтром по `kind`. `inference.py` ищет шаблон в порядке: `--sample-submission` → `<data-dir>/sample_submission.csv` → `baseline/sample_submission.csv`. Формы, dtype и имена каналов тестовых файлов идентичны train, но **геопривязка из тестовых GeoTIFF вырезана** (`crs=None`, transform единичный) — `chip["crs"]`/`chip["transform"]` на тесте бесполезны. В архиве те же пустые каталоги-пустышки (`af/images`, `bs/pre`, …) — при распаковке пропущены. Сдвиг test vs train — `reports/test_vs_train.md` (тест: днём I4 теплее на ~6 K в хвосте, больше ночных чипов, больше облачных BS-чипов, меньше пашни — больше травы/болота).

AF: 90 public + 90 private; BS: 44 + 45. Итого 269 чипов, 447 строк submission (269 строк class 1, 89 class 2, 89 class 3). Разделение public/private одновременно по территории и по времени → валидация должна быть групповой (по тайлу/пожару) и по возможности временной.

В train только 2019–2024, а AOI-README говорит о сезонах 2019–2025 → private-тест почти наверняка содержит 2025 год. Временной сплит (`temporal_split`, val = 2024) считается основным индикатором private, групповой GroupKFold — вспомогательным. Улучшение засчитывается только при приросте на ОБОИХ сплитах; все агенты моделей отчитывают обе цифры.

## 5. Формат submission.csv

UTF-8, разделитель запятая, заголовок `chip_id,class_id,rle`. Ровно строки шаблона `data/test/sample_submission.csv` (копии — `baseline/sample_submission.csv`, `baseline/sample_submission_1.csv`): по одной на AF-чип (`class_id=1`) и по три на BS-чип (`class_id` 1,2,3), всего 447. Порядок строк — как в шаблоне.
RLE: пиксели нумеруются **с 1**, построчно слева направо, сверху вниз (row-major). Строка — пары `start length` через пробел, по возрастанию `start`; серии не пересекаются и **не соприкасаются** (сливать смежные); выход за пределы `H*W` запрещён. NaN/пропуски запрещены.
**Текстовый формат — как в примере организаторов `baseline/example_submission.csv`** (прислан после старта; `tests/test_submission_example.py`): колонка `rle` БЕЗ кавычек (`csv.QUOTE_MINIMAL`), пустая маска — пустое поле (`AF_te_000001,1,`), порядок строк строго как в шаблоне (не сортировать), переводы строк CRLF. `write_submission` пишет именно так; `read_submission` (валидатор, `evaluate_submission`, `compare_submissions.py`) читает оба стиля — с кавычками (старые файлы `submissions/*.csv`) и без. Внутри BS-чипа маски классов 1/2/3 не пересекаются. Официального скрипта RLE нет: правила — по тексту ТЗ (`tests/test_rle_spec.py` проверяет каждое на ручных примерах) и по примеру организаторов (декодирование → кодирование совпадает посимвольно), проверены отправками на лидерборд (`submissions/submission_stub.csv`). Перед любой отправкой: `python scripts/check_submission.py <csv>`.

## 6. Предметные факты (использовать в признаках и в отчёте)

**AF.** Очаг 600–1200 K светит в MIR: I4 поднимается на сотни K при доле очага в пиксели <5 %, I5 — на десятки. Признак горения — I4 и разность I4−I5 относительно **фоновой статистики соседей в скользящем окне** (контекстный алгоритм, как в VNP14). Ложные срабатывания: (1) газовые факелы — по одному снимку неотличимы, в эталоне размечены 0 по признаку повторяемости зимой; (2) промышленные площадки/металлургия/ТЭЦ — landcover 50, стабильно тёплые; (3) солнечный блик на воде/крышах/теплицах — только днём, высокий I3 и I1–I2, отношение I4/I5 иное; (4) облака и края сцены. Ночью (solar_zenith>90) фон холоднее, контраст выше, бликов нет. **Эталон AF (EDA)** ≈ VNP14-подобные пороги, раздельные для дня и ночи: правило `день I4>329 K & I4−I5>25 K; ночь I4>295 K & I4−I5>10 K` даёт F1 0.921 на train (заглушка 0.244) — это опорная точка и источник признаков; абсолютные значения на тест переносить осторожно (дневной фон теста теплее на ~6 K, `reports/test_vs_train.md`). Ночные ложные срабатывания — техноген (52 % на застройке против 20 % у огня); карту повторяемости аномалий на тест переносить нельзя (нет геопривязки и запрещено правилами) — только признаки из чипа.

**BS.** `NBR = (B8A−B12)/(B8A+B12)`, `dNBR = NBR_pre − NBR_post`, `RdNBR = dNBR/sqrt(|NBR_pre|)`. Пороги степени в эталонной разметке **зависят от типа покрова**: для леса/кустарника — шкала USGS, для степи/пашни/поймы — свои, ниже. Значит один глобальный порог заведомо плох, landcover обязан быть признаком. Ложные контуры: убранные/вспаханные поля (положительный dNBR без огня), облака и тени (SCL ненадёжен — светлую почву принимает за облако), дым (давит NIR сильнее SWIR), зарастание травяной гари за 2–4 недели. Sentinel-1: в лесу после пожара падает VH, в степи меняется VV; спекл-шум → медианный фильтр 3×3/5×5 перед использованием.

**BS — как устроен эталон (EDA, `reports/eda.md`, `eda/tables/`).** (1) Эталон гари **обрезан круглым буфером события, r ≈ 4 км**: гарь вне круга в чипе размечена 0. Оракул-окружность поднимает IoU_burn правила 0.355 → 0.619 (→ 0.673 с маской SCL) — попиксельных признаков мало, нужна пространственная модель/локализация события (`eda/tables/bs_b4b_event_zone*.csv`). (2) **MMU 25 px (1 га)**: компонент гари меньше 25 px в эталоне нет → в постобработке удалять компоненты < 25 px. (3) Под `SCL_post` ∈ {0 nodata, 3 тень, 9 облако выс., 11 снег} гари в эталоне **0 %** → там всегда предсказывать 0. (4) Степень = пороги dNBR **по landcover** (t1/t2/t3, `eda/tables/bs_b3_thresholds_by_landcover.csv`): лес (10) 0.16/0.32/0.60; трава (30) 0.08/0.22/0.40; пашня (40) 0.08/0.18/0.38; болото (90) 0.14/0.34/0.62; кустарник, застройка, голая почва, вода, прочие — глобальные 0.08/0.20/0.40 (гари там < 0.1 %). При известном контуре эти пороги дают mIoU_sev 0.877 — т.е. степень почти решена, главный резерв — контур (IoU_burn). (5) **Не обрезать предсказание по центру чипа; расстояние до центра — максимум мягкий признак** (EDA B4c, `reports/eda_event_circle.md`): центр события смещён от центра чипа (медиана модуля смещения центроида гари 159 px), у 87 % чипов эталонная гарь касается края, обрезка кругом 280 px даёт IoU_burn лишь 0.418 → 0.424; событие локализовать по содержимому чипа.

## 7. Правила соревнования (нарушение = дисквалификация)

- Только открытое и бесплатное ПО; предобученные модели — только публичные до старта, с указанием лицензии в отчёте.
- Нельзя использовать данные о территориях/датах тестовой выборки, готовые продукты FIRMS/VNP14/MCD64A1/FireCCI и т.п. для ответов по тесту.
- Нельзя восстанавливать геопривязку/даты тестовых чипов и размечать тест вручную. Модели работают ТОЛЬКО с содержимым чипа и полями тестового meta.
- Организаторы могут перезапустить инференс по нашей инструкции — README должен быть исчерпывающим.

## 8. Соглашения по коду

- Python 3.12 — основной (для 3.10/3.11 — страховочный `requirements-py310.txt`, см. ниже), окружение Windows + PowerShell (без bash-измов в скриптах; пути через `pathlib`; никаких абсолютных путей с машины разработчика). GPU (CUDA) доступна для обучения; инференс должен работать и на CPU.
- Структура репозитория (не менять без необходимости):
```
fire-monitoring/
  CLAUDE.md  README.md  requirements.txt  Dockerfile  .gitignore  pyproject.toml
  inference.py         # единственная точка входа инференса (ТЗ) — только CLI
  train.py             # `python train.py --module af|bs|all`
  configs/             # yaml-конфиги моделей и путей
  src/firemon/
    io.py              # чтение чипов → dict[np.ndarray (C,H,W)] + описания каналов
    rle.py             # encode/decode + валидация
    metrics.py         # F1_af, IoU_burn, mIoU_sev, Score — совместимо со скриптом организаторов
    splits.py          # групповые фолды
    submission.py      # сборка/валидация submission.csv по шаблону
    pipeline.py        # конвейер инференса: реестры AF_BACKENDS/BS_BACKENDS — место подключения моделей
    features/af_features.py  bs_features.py
    models/af/…  models/bs/…
    postprocess.py
    utils.py           # seed_everything, логирование, таймеры
  service/             # FastAPI app + static/ (Leaflet), data/ (предрассчитанные слои)
  eda/                 # скрипты EDA, figures/
  reports/  report.md  experiments.md  figures/
  tests/               # pytest
  weights/             # веса моделей (маленькие — в git, большие — по ссылке в README)
  docs/                # case.pdf, criteria.pdf, txt
  data/  baseline/     # gitignored
```
- Детерминизм: `seed_everything(42)` везде (random, numpy, torch, cudnn.deterministic), `n_jobs`/`num_threads` фиксированы, LightGBM `deterministic=True`.
- Каждый модуль имеет тесты в `tests/` (pytest), запуск `python -m pytest -q` должен быть зелёным перед завершением задачи.
- Логи экспериментов: каждая обученная модель/вариант → строка в `reports/experiments.md`: дата, модуль, описание, конфиг, схема валидации, seed, OOF-метрики (F1_af или IoU_burn/mIoU_sev/Score), время инференса, вывод. Без этого эксперимент считается несделанным.
- Не трогать `data/` и `baseline/` (только чтение). Не коммитить данные и большие веса (>50 МБ).
- Коммиты небольшие, осмысленные сообщения. Перед завершением задачи: `python -m pytest -q`, `git status` чистый.
- Если что-то в этом файле противоречит выданному baseline или уточнениям организаторов — правы организаторы; исправь CLAUDE.md и сообщи в итоговом отчёте задачи.

### Что уже есть (задача 01) — использовать, не дублировать
- Окружение: `.venv` на **Python 3.12.7** (3.10/3.11 на машине нет), torch 2.6.0+cu124, RTX 4070 12 ГБ. Запуск: `.\.venv\Scripts\python.exe …`. В PowerShell перед запуском скриптов с кириллическим выводом: `$env:PYTHONIOENCODING='utf-8'`.
- **`aux` — зарезервированное имя устройства Windows.** Файлы `…/aux/X_AUX.tif` открываются обычным путём, но сам каталог нельзя перечислить/создать (`os.listdir`, `mkdir`, `Path.resolve`, `os.path.abspath`, Проводник, `Get-ChildItem` — ломаются). Для операций с каталогом — `firemon.io.win_path(path)` (префикс `\\?\`). Никогда не вызывать `abspath/resolve` на пути, оканчивающемся на `aux`.
- `win_path()` — no-op на любой ОС, кроме Windows (`os.name != "nt"` → `str(path)`); на Windows делает путь абсолютным и добавляет `\\?\` без дублирования. **Любые OS-специфичные ветки обязаны иметь no-op для Linux** (организаторы и Docker — Linux): никаких `\\?\`, `os.name == "nt"`-логики и Windows-путей без такой защиты и без теста с `monkeypatch.setattr(os, "name", …)`. Python: `>=3.10,<3.13` (`pyproject.toml`): пины `requirements.txt` (numpy 2.5, pandas 3.0) на 3.10/3.11 НЕ ставятся — для них `requirements-py310.txt` / `-cpu.txt` ← `requirements-py310-base.txt` (numpy 2.2.6, pandas 2.2.3 …; на 3.10.21 инференс побайтно равен финалу, pytest зелёный — `h100_readiness.md` п. 18); новый пакет добавлять в ОБА base-файла; Python 3.13 — Docker; все пакеты имеют колёса manylinux x86_64 / cp312 (нужен glibc ≥ 2.28: Debian 11+/Ubuntu 20.04+, `python:3.12-slim` подходит).
- В `data/train` есть пустые каталоги-пустышки из архива (`af/images`, `bs/pre`, `bs/post`, `bs/sar_pre`, `bs/sar_post`) — игнорировать.
- API: `firemon.io.read_af_chip / read_bs_chip(data_dir, chip_id, with_mask, layers=…)` (есть ключи `valid`, `mask_valid`; `layers` — читать только нужные слои), `list_chips`, `read_meta`, `read_mask`; `firemon.rle`; `firemon.metrics.evaluate(af_preds, af_gts, bs_preds, bs_gts)` и `evaluate_submission`; `firemon.submission.build_submission / validate_submission`; `firemon.splits.load_folds("af"|"bs")`.
- Чтение: AF ≈ 17 мс/чип (viirs+aux+mask), BS ≈ 70 мс/чип (все 5 слоёв + маска); узкое место — распаковка deflate, rasterio отпускает GIL → при необходимости читать в `ThreadPoolExecutor`.
- В train годы 2019–2024, а README к AOI говорит о сезонах 2019–2025 → private-тест, вероятно, содержит 2025 год; временной сплит (val=2024) — самая честная оценка.
- RLE шумных BS-масок длиннее стандартного лимита поля `csv` (131072) — читать submission только через `firemon.submission.read_submission` (лимит поднят) или pandas.
- Фейковый тест (train-чипы с масками, для сквозной проверки с метрикой): `python scripts/make_fake_test.py` → `data/test_fake/` (20 AF + 10 BS, свой `sample_submission.csv`). Шаблона `sample_submission_FAKE.csv` больше нет; для OOF-оценки `evaluate_submission(csv, data/train)` шаблон не обязателен.
- Реальный тест: `python scripts/check_data.py --split test` (комплектность, chip_id meta == шаблон == файлы, 5+5 случайных чипов против train), `python eda/test_vs_train.py` (сдвиг), `python scripts/check_submission.py <csv>` (валидатор + сводка пикселей/пустых масок). Заглушка на реальном тесте: 10–13 с, код 0.
- В AF-чипах NaN в I4/I5 встречаются и при `valid=1` и `valid_frac=1.0` (bow-tie, см. §4; в тесте доля чуть выше: 9.4 % против 8.3 % по среднему на чип) — `valid_frac` из meta их не отражает; пользоваться ключом `chip["valid"]`.
- EDA (задача 02): `eda/eda_af.py`, `eda/eda_bs.py` (кэш `eda/cache/`, gitignored), отчёт `reports/eda.md`, числа для слайдов `reports/eda_key_numbers.md`, 46 таблиц `eda/tables/*.csv`.

### Модуль AF (задача 03) — использовать, не дублировать
- `firemon.models.af.predictor.AFPredictor(cfg).load()` → `predict(chip)` (маска 0/1), `predict_proba(chip)`, `predict_many(chips)`; функция `predict_af(chip)`. Веса: `weights/af_lgbm_fold{0..4}.txt` + `af_lgbm_meta.json` (порог, список признаков). Нет весов → `inference.py` откатывается на `rules.predict_af_context` (контекстное правило, OOF F1 0.8965).
- Признаки: `firemon.features.af_features` — `compute_af_features(chip)` (F,H,W), `af_feature_rows(chip, mask)` (n,F), `candidate_mask`, `FEATURE_NAMES` (110). Модель обучается и применяется ТОЛЬКО на кандидатах (recall кандидатов на train 100 %).
- Метрики AF (финал, AF-17): OOF F1 0.9682 (GroupKFold), 0.9597 (temporal 2024). Постобработка (гистерезис и др.) проверена и отклонена. Детали — `reports/af_module.md`.
- Кэш пиксельного датасета — `data/cache/af_train.npz` (npz, не parquet: в окружении нет `pyarrow`).
- Кандидаты AF (AF-17): день `I4>314 K & I4−I5>17 K`, ночь `I4>290 K & I4−I5>5 K` — запас ≥ 5 K от минимума огня train. Sanity на реальном тесте — `reports/af_test_sanity.md` (дневные сцены теста теплее train).

### Сервис (задача 06) — использовать, не дублировать
- `service/` (FastAPI + Leaflet), запуск `python -m uvicorn service.app:app --port 8000`; данные — `python scripts/build_service_data.py --source both` (перезапустить после появления финальных `AFPredictor`/`BSPredictor`: адаптер `service/predictors.py` ждёт `Predictor(config).load()/.predict(chip)` и опционально `.predict_proba(chip)`). В git — только снимок gt `service/data/*.geojson.gz`; `firemon.gpkg` не коммитить. Описание — `reports/service.md`. Интерфейс (`service/static/`, без изменений API): подложки Спутник (Esri) / Карта (OSM) / Нет с автооткатом офлайн, термоточки по шкале I4 300→360 K, контур события считается в браузере объединением полигонов `fire_event_id` (vendor `polygon-clipping`, до 3 000 контуров), состояние запроса в URL (`?bbox=…&from=…&to=…&source=gt|pred` выполняется при открытии). Гари, контур и термоточки держать на ОДНОМ canvas-рендерере: отдельные canvas-панели ломают hover у нижних слоёв.

### Инференс, Docker, скорость (задача 05) — использовать, не дублировать
- `inference.py` — тонкий CLI (`--data-dir --output [--sample-submission --config --device auto|cpu|cuda --workers N --profile]`), коды выхода 0 / 1 (файл не прошёл валидацию) / 2 (нет входных данных). Вся работа с моделями — в `src/firemon/pipeline.py`: бэкенд = функция `(data_dir, chip_ids, cfg, ctx) -> (preds, n_failed)`, реестры `AF_BACKENDS` / `BS_BACKENDS`, выбор в `configs/inference.yaml` (`<kind>.model`, `<kind>.fallback`). **Подключение BSPredictor = одна запись в `BS_BACKENDS` + веса в `weights/` + `bs.model` в конфиге**; `inference.py`, Dockerfile, README не трогать. Для бэкенда есть готовое: `map_chips` (потоки, пропуск отсутствующих/битых чипов → None), `empty_mask`, `ctx.torch_device()` (ленивый импорт torch), `ctx.profiler.stage(...)`.
- Скорость (AF LightGBM + BS-заглушка, CPU, 269 чипов): ~10–11 с, Docker 11.1–11.5 с; бюджет BS-модели на CPU ≈ 16–19 с до порога 30 с. Потоки = процессы по скорости → оставлены потоки. **Финал v5: GPU 23.2–24.1 с, CPU 56–58 с, Docker CPU 68.7 с** (`reports/speed.md` §1c). Детали и таблица для дополнения — `reports/speed.md`.
- Холодный старт (`reports/speed.md` §1d): `pipeline.prewarm_torch` в фоне импортирует torch/smp и создаёт CUDA-контекст, пока считается AF (первый запуск в свежем venv 33.9 → 27.9 с; после README-шага «проверка установки» — 23.3 с; тёплый ≈ 23.0 с). Байткод ни при чём (pip компилирует site-packages сам); geopandas/shapely/pyproj/matplotlib на пути инференса не импортируются — не добавлять их в top-level импорты `firemon` (тест `test_inference_imports_stay_lean`). Замеры времени делать, записывая загрузку GPU/CPU перед стартом: на машине разработки GPU периодически занят посторонними приложениями (выбросы 30–43 с).
- Потоки: `workers: auto`, `num_threads: auto` → `firemon.utils.auto_threads()` = min(доступные ядра, 16) с учётом affinity и квоты cgroup (`docker run --cpus N`); константы в конфиги не возвращать. CUDA: `firemon.utils.cuda_usable()` (is_available + пробное выделение) → при старом драйвере/без GPU автоматический откат на CPU с одной строкой WARNING; TF32 и `cudnn.benchmark` выключены явно, `CUBLAS_WORKSPACE_CONFIG` выставляется через `setdefault`, чекпойнты грузятся с `weights_only=True`, сеть на инференсе не нужна (`tests/test_portability.py`, `reports/tasklog/h100_readiness.md`). Логи: `get_logger` делает `stdout.reconfigure(errors="replace")` — кириллица не роняет процесс при `LANG=C`.
- Детерминизм: submission побайтно одинаков при любом `--workers`, числе потоков LightGBM, в venv cu124 / venv CPU / Docker (`tests/test_reproducibility.py`); у v5 — и GPU против CPU (sha256 `784fe474…`). Переобучение AF даёт те же веса по содержимому; на Windows LightGBM пишет CRLF, в git LF — «M» в `git status` после `train.py` при пустом `git diff` — это только EOL.
- Зависимости: `requirements-base.txt` (всё без torch) ← `requirements.txt` (torch cu124) и `requirements-cpu.txt` (torch CPU, Docker). Новый пакет добавлять в `requirements-base.txt`. В `python:3.12-slim` колесу rasterio нужен `libexpat1`, LightGBM — `libgomp1` (уже в Dockerfile).
- Docker: `Dockerfile` (CPU, собран и проверен на реальном тесте), `Dockerfile.gpu` (не тестировался). В образ копируются только `inference.py`, `train.py`, `configs/`, `src/`, `weights/` — новый каталог, нужный инференсу, добавить в `COPY`.
- Отправки на лидерборд — `reports/submissions.md` (stub 0.3117; AF LightGBM + BS-заглушка 0.6035 → F1_af на public ≈ 0.93–0.99; v3 с TTA 0.8267; v4 0.8262; **v5 финал 0.8314**).

### Разработка в нескольких worktree
- Worktree `C:\Users\User\hack\wt-{af,bs,eda,svc}` (ветки `task/*`) рядом с основным `fire-monitoring` (`main`). **Общий `.venv` основного worktree, `pip install -e .` НЕ делать**: editable-установка привязала бы `firemon` к одному worktree, остальные молча импортировали бы чужой код.
- В каждой сессии `PYTHONPATH` = `<свой worktree>\src` (`$env:PYTHONPATH = "$PWD\src"`); точки входа и тесты сами добавляют свой `src/` в `sys.path`.
- В каждом worktree `data`, `baseline`, `weights` — **junction-ы** на каталоги основного. Следствия: (1) веса, записанные агентом в своём worktree, сразу появляются в `main` как неотслеживаемые файлы — перед слиянием ветки сверять с веткой по blob-хэшу (`git hash-object` против `git rev-parse <ветка>:<путь>`), совпавшие копии удалять, иначе merge откажется; (2) ветка может «случайно» закоммитить чужие веса (так было с `task/eda`) — безвредно, если файлы совпадают байт-в-байт.
- ⚠️ **Никогда не удалять worktree с junction-ами через `Remove-Item -Recurse` / `rm -rf`** — команда пройдёт внутрь junction и удалит общие `data/` и `weights/`. Сначала снять ссылки: `cmd /c rmdir data`, `cmd /c rmdir baseline`, `cmd /c rmdir weights` (удаляет только ссылку), затем `git worktree remove <путь>`.

### Что уже есть (задача 04, модуль BS) — использовать, не дублировать
- Инференс: `firemon.models.bs.predictor.BSPredictor(cfg).load().predict(chip)` / `predict_bs(chip)`; батчевый конвейер для
  `inference.py` — `firemon.models.bs.infer.run_bs_inference(data_dir, chip_ids, load_bs_config(cfg))`. Конфиг —
  `configs/bs_inference.yaml`; пороги декодирования и постобработка — `weights/bs_decode.json` (пишет `ensemble.py`).
  Нет весов → аварийный режим на правилах dNBR (код выхода 0 сохраняется). Предиктор не читает геопривязку/даты/meta.
- Схема: 2×UNet-resnet34 (каналы v1 и v4, среднее) → LightGBM 2-го уровня на «воротах» p_burn>0.03 → декодирование →
  постобработка. OOF BS-часть 0.4955 (group) / 0.5033 (temporal); детали и все абляции — `reports/bs_module.md`.
- Факты об эталоне BS (проверены на train): степень = пороги по **медианному dNBR 3×3**, свои для landcover (`RULE_THR` в
  `bs_features.py`, признак `rule_sev`); эталон ⊂ маски правила (97.9 %) и обрезан буфером события ~4 км, **не
  центрированным по чипу** (резать по центру чипа нельзя; центр события оценивается по содержимому); под SCL pre {0,3,6,11} /
  post {0,3,9,11} гари нет; MMU = 25 px; облака/тени НЕ обнулять (−0.07…−0.09 BS-части).
- Постобработка BS живёт в `src/firemon/models/bs/postprocess.py` (общий `src/firemon/postprocess.py` не тронут).
- BSPredictor подключён как бэкенд `"unet"` в `BS_BACKENDS` (`src/firemon/pipeline.py`), `bs.model: unet`, `fallback: rules`.
  Полный `inference.py` на реальном тесте (269 чипов): GPU 21.1–21.5 с (BS ≈ 16 с), `--device cpu` 52.4 с (BS 47 с).
- BS по умолчанию БЕЗ TTA и с UNet в fp32 на GPU (`tta: false`, `gpu_half: false`): прирост TTA < 0.005 BS-части, а так GPU- и
  CPU-файлы отличаются 1 пикселем из 23.3 млн. Финальный OOF BS-части 0.4912 / 0.5008 (с TTA было 0.4955 / 0.5033; TTA-вариант —
  `weights/bs_stack_tta.txt` + `bs_decode_tta.json`). Приор «в каждом BS-чипе есть гарь» — `nonempty_min_px: 25` в постобработке.
- Обучение UNet: взвешенный `F.cross_entropy` на CUDA недетерминирован (OOF двух прогонов расходился до 0.02) → в `unet.py`
  one-hot CE (`ce_loss`), веса двух прогонов побитно равны. `train.py --module bs --weights-dir X/` пишет веса/кэш/итоги отдельно.
- В окружении нет `pyarrow`, `onnxruntime`, `psutil` — кэши BS лежат в `data/cache/*.npy` (не parquet).
- **Финал — v5 (`bs.variant: ms`, по умолчанию в `configs/inference.yaml` и `configs/bs_inference.yaml`)**: среднее трёх UNet (v1, v4 на 1× и v4, обученный на масштабе 0.75×: `bs_unet_r34_v4_s075.pt`) → стек `bs_stack_ms.txt` с признаками UNet 0.5× (`bs_unet_r34_v4_s050.pt`) и воротами 0.01 → `bs_decode_ms.json` (event_zero_r 300; жадный подбор при переобучении даёт 350, Δ 0.0004 — шум). OOF 0.7645 / 0.7632 / BS-часть 0.4965 (Score 0.8353), temporal 0.7904 / 0.7831 / 0.5116 (0.8478), public 0.8314. v4 = `bs.variant: base` (или `--config configs/inference_bs_base.yaml`) — быстрый режим: GPU 21 с / CPU 52 с, −0.005 Score. Обучение v5 — этапы `ms_unet`, `ms_stack` в `train.py --module bs`. Проверено и НЕ принято (не повторять): ens4 из четырёх UNet, классификатор компонент (AUC 0.7; потолок оракула +0.025/+0.034), Lovász, SWA, многомасштабный инференс тех же сетей, рамочный приор центра — `reports/tasklog/04_bs_night.md`, `reports/bs_module.md` §4.9–4.11.
- **Чистый клон должен проходить README** (`reports/tasklog/clone_check.md`): тесты не имеют права зависеть от `baseline/` и `data/` — файлы организаторов для тестов лежат в `tests/fixtures/` (байт-в-байт, `.gitattributes`: `-text`, иначе `eol=lf` ломает CRLF примера), тесты на данных — только через фикстуры `train_dir` / `test_dir` / `fake_test_dir` из `conftest.py` (дают `pytest.skip`); сервис в клоне проверяет `tests/test_service_snapshot.py`. Перед релизом: `python scripts/check_release.py` (ссылки из README/HANDOFF/отчёта есть в git, размеры < 100 МБ, `git status` чистый → `reports/tasklog/release_check.md`). Сверка с ТЗ — `reports/tasklog/tz_compliance.md`. Выгрузки демо-запроса для защиты — `python scripts/make_demo_exports.py` → `handoff/demo_exports/`.
- В PowerShell-инструменте here-string с двойными кавычками внутри `python -c` ломается — патчи/скрипты писать в файл. То же в Git Bash: heredoc превращает `\r\n` внутри python-строк в реальные переводы строк — скрипты с escape-последовательностями писать инструментом Write.

## 9. Формат отчёта о выполненной задаче

В конце каждой задачи агент пишет краткий отчёт (в ответе и в `reports/tasklog/<номер>_<имя>.md`): что сделано, какие файлы, как запустить, какие тесты прошли, метрики (если есть), что не получилось/открытые вопросы, что менять в CLAUDE.md.
