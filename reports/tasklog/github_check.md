# Публикация на GitHub и контрольный клон (2026-09-19)

Удалённый репозиторий: **https://github.com/shttrkk/kosmowinsamarkand** (приватный). Публикация подтверждена пользователем.

## Публикация

| Шаг | Результат |
|---|---|
| Финальный файл сабмита в git | `submissions/submission_v5_final.csv` (3.0 МБ, 447 строк) — исключение `!/submissions/submission_v5_final.csv` в `.gitignore`; `.gitattributes`: `submissions/*.csv -text` (файл с CRLF, как пример организаторов; без этого `eol=lf` изменил бы байты и sha256). sha256 блоба в git = sha256 сданного файла: `784fe474040914ed561c29efb8f34c1115e07f422e575e45bcfbceb683523afc` |
| README | адрес репозитория (первый экран и §1), §6 — «эталонный результат инференса» с sha256 и командами проверки хеша |
| `python scripts/check_release.py` | «Проблем нет»: 374 файла, самый большой — 47.0 МиБ (`weights/*.pt`), файлов ≥ 100 МБ — 0, 50–100 МБ — 0 |
| Проверка на секреты перед push | `git grep` и `git log -p -G` по шаблонам токенов GitHub / ключей AWS / приватных ключей / паролей — пусто |
| `git remote add origin …` → `git ls-remote` | доступ без интерактива (сохранённые учётные данные), удалённый репозиторий был пуст |
| `git push -u origin main --tags` | код 0, ≈ 11 мин (243 МБ, канал на отдачу); `main` и `v1.0-final` → `a1d88fa`; ветки `task/*` не публиковались |

## Контрольный клон с GitHub (`C:\Users\User\hack\gh_check`, коммит `a1d88fa`)

| Проверка | Результат |
|---|---|
| `git clone https://github.com/shttrkk/kosmowinsamarkand.git` | код 0, 37 с; рабочая копия 269 МБ, `.git` 243 МБ |
| `git describe --tags` | `v1.0-final` |
| Состав | `weights/` — **21 файл**, четыре `.pt` по 47 МиБ (настоящие бинарные файлы, не указатели LFS); `service/data/` — `burn_scars`, `chips`, `hotspots` `.geojson.gz`; `reports/report.pdf` 3.1 МБ; `presentation/deck.pdf` 1.5 МБ; `handoff/HANDOFF.md`, `handoff/demo_exports/*` (включая zip с Shapefile); `tests/fixtures/example_submission.csv` 420 КБ (CRLF сохранён); `submissions/submission_v5_final.csv` — sha256 `784fe474…` |
| venv по README (`py -3.12 -m venv`, `pip install -r requirements.txt`, `compileall`, проверка установки) | код 0, 188 с; `torch 2.6.0+cu124, cuda True` |
| `python inference.py --data-dir <data/test основного репо> --output out/sub.csv` | код 0; `out/sub.csv` **побайтно равен `submissions/submission_v5_final.csv` из клона** (sha256 совпали). Время: **первый запуск 41.7 с**, три повторных — 22.24 / 22.26 / 21.97 с (AF 4.7–4.9, BS 16.1–16.4) |
| `python -m uvicorn service.app:app --port 8041` | `/health` 200 через 2 с; `/` 200; демо-запрос `source=gt`: 98 термоточек, 337 контуров, 1 758.92 га |
| `python -m pytest -q` | код 0: **130 passed, 40 skipped**; все 34 строки пропусков — из-за отсутствия `data/train`, `data/test`, `data/test_fake` или `baseline/` |
| `python scripts/check_release.py --no-write` | «Проблем нет» |

`gh_check` после проверки удалён.

## Замечания

- **Первый запуск в свежем окружении — 41.7 с**, хотя шаг «проверка установки» был выполнен. Вместе с замерами аудита холодного старта
  (`reports/speed.md` §1d) первый запуск в шести свежих окружениях занимал от 23 до 42 с; стабильно быстрым (22–23 с) становится второй.
  Формулировки в `speed.md`, README, `h100_readiness.md` и CLAUDE.md исправлены: шаг проверки холодный старт не гарантированно убирает.
  Для критерия «< 30 с» это главный остаточный риск, кодом он не закрывается.
- Размер репозитория на GitHub через API получить не удалось (`gh` не установлен, репозиторий приватный — анонимный API отдаёт 404);
  ориентир — размер pack-файла, который GitHub получил: **≈ 243 МБ** (рабочая копия 269 МБ). Точное число — на странице репозитория → Settings.
- В репозитории лежат производные от выданных организаторами данных (снимок эталонных слоёв сервиса, `tests/fixtures/`, файл сабмита) —
  репозиторий должен оставаться приватным; доступ жюри — через приглашение коллаборатором.
