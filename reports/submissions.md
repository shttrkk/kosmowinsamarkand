# Журнал отправок на лидерборд

Каждая отправка = строка: файл, коммит, какие модели/веса/конфиг, локальные метрики, public Score (вписывает Марк после отправки),
вывод. Файлы лежат в `submissions/` (не в git — воспроизводятся командой из колонки «Как получен»). Перед отправкой обязательно:
`python scripts/check_submission.py <csv>`.

| # | Дата | Файл | Коммит main | Модели | Как получен | Локально (train) | Public Score | Вывод |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-09-18 | `submission_stub.csv` | `c66d481` | AF: правило `I4>320 K & I4−I5>15 K`; BS: dNBR 0.10/0.27/0.44 (наш baseline, строка 0 `experiments.md`) | `python inference.py --data-dir data/test --output out/submission_stub.csv` | Score 0.3088 (F1_af 0.2435, IoU_burn 0.3706, mIoU_sev 0.3129) на всём train | **0.3117** | Формат принят с первой попытки: кавычки `""`, RLE с 1 row-major, порядок строк — всё верно, реализация по тексту ТЗ подтверждена. Public 0.3117 ≈ train 0.3088 — baseline переносится на тест без сюрпризов. |
| 2 | 2026-09-18 | `submission_af.csv` | `3e243ae` | AF: LightGBM AF-17 (5 фолдов, `weights/af_lgbm_*`); BS: та же заглушка dNBR | `python inference.py --data-dir data/test --output out/submission_af.csv` | F1_af OOF 0.9682 / temporal 0.9597; BS как в №1 | **0.6035** | ΔScore = +0.2918 при неизменном BS → всё изменение даёт AF: ΔScore = 0.35·ΔF1_af ⇒ ΔF1_af ≈ 0.834. F1_af заглушки на public неизвестен; при F1_af(stub) от 0.10 до 0.16 (train 0.24; на тесте дневной фон теплее и заглушка даёт ×5 срабатываний, т.е. ниже train) получается **F1_af(public) ≈ 0.93–0.99** — согласуется с OOF 0.968 / temporal 0.960. AF подтверждён, переобучения на train не видно; дальнейший резерв Score — BS (вес 0.65). |
| 3 | 2026-09-18 | `submission_v3_af_bs.csv` | `9b575aa` | AF: LightGBM AF-17; BS: BSPredictor — 2×UNet-resnet34 (fp16, TTA) → стек-LGBM → декодирование порогами → постобработка (`weights/bs_*`) | `python inference.py --data-dir data/test --output submissions/submission_v3_af_bs.csv` (GPU; sha256 `7e3045fb…`) | OOF: F1_af 0.9682, IoU_burn 0.7631, mIoU_sev 0.7614 → Score **0.8344**; temporal 2024: 0.9597 / 0.7761 / 0.7720 → **0.8392** | _впишет Марк после отправки_ | Первый полный пайплайн. Прогноз public не даём: AF на public подтверждён, но BS-часть теста облачнее train (25.8 % против 10.7 % чипов). Файл получен на GPU; прогон `--device cpu` даёт другой файл (`6b92fb65…`, без TTA). |
| 4 | 2026-09-19 | `submission_v4_final.csv` | `6b15df2` | AF: LightGBM AF-17; BS: BSPredictor **без TTA, UNet в fp32** (`weights/bs_stack.txt`, `bs_decode.json`, t_burn 0.46), приор «в каждом чипе есть событие» | `python inference.py --data-dir data/test --output submissions/submission_v4_final.csv` (конфиг по умолчанию; sha256 `543c3e62…`, побайтно равен прогону автора BS-модуля; `--device cpu` и Docker — `7c5db547…`, отличие 1 пиксель) | OOF: 0.9682 / 0.7560 / 0.7554 → Score **0.8300**; temporal 2024: 0.9597 / 0.7723 / 0.7684 → **0.8367** | _ожидается (public v4)_ | **Финальная конфигурация.** Против №3: −0.0044 OOF / −0.0025 temporal (цена отказа от TTA), зато файл не зависит от устройства и воспроизводится организаторами без GPU. BS: 1 322 448 / 1 112 061 / 596 540 px, пустых масок 1 / 4 / 10. |

## Что следует из двух отправок

- BS-заглушка на public: `0.35·IoU_burn + 0.30·mIoU_sev = 0.3117 − 0.35·F1_af(stub)` ≈ 0.26–0.28, т.е. на уровне train
  (0.35·0.3706 + 0.30·0.3129 = 0.2236) или немного выше — оценка грубая, зависит от неизвестного F1_af заглушки.
- Потолок текущей связки: даже при F1_af = 1.0 Score ≤ 0.35 + вклад BS ≈ 0.61–0.63. Весь дальнейший рост — за счёт BS-модели.
- Следующая отправка — после слияния `task/bs` (BSPredictor): ожидаемый файл `submissions/submission_<дата>_<коммит>.csv`,
  копия для отправки — `submissions/final_submission.csv`.

## Состояние после задачи 05 (рефакторинг inference.py, Docker)

`python inference.py --data-dir data/test --output out/submission.csv` на текущем `main` даёт файл, **побайтно совпадающий**
с отправкой №2 (`sha256 69c22e9fd6ddb3a5…`) — в venv с torch cu124, в чистом venv с torch CPU и в Docker-образе `firemon`.
Поэтому новая отправка не формировалась; `submissions/final_submission.csv` будет собран после подключения BSPredictor
(порядок: слияние `task/bs` → запись в `BS_BACKENDS` → `inference.py` → `scripts/check_submission.py` → строка в этой таблице).
