# Промпт 5 — inference.py, воспроизводимость, Docker, скорость, финальный submission

> После промптов 3 и 4 (нужны предикторы и веса). Один агент. Прочитай `CLAUDE.md`. Это раздел критериев «Код и воспроизводимость» (12 баллов) + «Скорость» (8 баллов) + валидность сабмита (без него 0 по метрике).

## Цель
Организатор на своей машине (возможно без GPU) выполняет РОВНО то, что написано в README, и за минимальное время получает валидный `submission.csv`, а при повторе — тот же файл. Целевое время полного инференса на 269 чипов: **< 30 с на GPU** и как можно ближе к этому на CPU (≤ 5 мин обязательно).

## Задачи

### 1. `inference.py` — финальная версия
- CLI: `python inference.py --data-dir <dir> --output <file.csv> [--sample-submission <csv>] [--config configs/inference.yaml] [--device auto|cpu|cuda] [--workers N]`. `--data-dir` — каталог теста в структуре организаторов (`af/`, `bs/`, `meta.csv` в каждом, `sample_submission.csv` — проверь, где он лежит в реальном тесте, и поддержи оба варианта: в `--data-dir` и в `baseline/`). Ничего интерактивного, код выхода 0 при успехе, ненулевой с понятным сообщением при отсутствии файлов.
- Порядок: загрузка весов один раз → список чипов из sample_submission (не из meta — обрабатываем ровно то, что требует шаблон; если чип из шаблона отсутствует на диске — пишем пустую маску и warning, не падаем) → AF-чипы → BS-чипы → `build_submission` → `validate_submission` (внутренняя проверка, при ошибке — ненулевой код) → лог времени по этапам.
- Устойчивость: NaN, чипы с высокой облачностью, отсутствующий `masks/`, лишние файлы — всё обрабатывается. Тест на «сломанных» чипах (NaN везде, нули везде) — предсказание пустое, без исключений.
- Параллелизм: чтение и признаки в `concurrent.futures` (процессы на Windows — `spawn`; проверь, что это не медленнее из-за старта процессов; возможно, лучше потоки + numpy). LightGBM `num_threads` = число ядер. UNet — батчами по 8–16 чипов, `torch.inference_mode()`, fp16 на GPU, `torch.set_num_threads` на CPU, ONNX Runtime на CPU, если быстрее.
- Профилирование: `--profile` печатает таблицу (чтение, признаки, predict AF, predict BS, RLE, запись). Оптимизировать самое дорогое. Результаты в `reports/speed.md`: таблица «конфигурация → время» для GPU и CPU, холодный старт vs тёплый.

### 2. Детерминизм
- Два запуска `inference.py` подряд → `submission.csv` побайтно одинаковые (тест `tests/test_reproducibility.py` на `data/test_fake`). Если UNet на GPU недетерминирован — включить `torch.use_deterministic_algorithms(True)` / `CUBLAS_WORKSPACE_CONFIG=:4096:8` или на крайний случай усреднять и округлять вероятности до 1e-4 перед порогом. Метрика при повторе должна отличаться ≤ 0.005 — у нас цель 0.000.
- `train.py` — два запуска дают OOF-метрики в пределах 0.005 (задокументировать фактические числа).

### 3. Окружение и Docker
- `requirements.txt` с точными версиями (`pip freeze` отфильтрованный) + отдельный `requirements-cpu.txt` (torch CPU). Python версия зафиксирована в README и `pyproject.toml`.
- `Dockerfile` (base `python:3.11-slim` или `pytorch/pytorch:*-cuda*` — два варианта: `Dockerfile` (CPU, гарантированно работает везде) и `Dockerfile.gpu`). Внутри: код, веса, зависимости; `ENTRYPOINT ["python", "inference.py"]`. Пример запуска в README:
  ```
  docker build -t firemon .
  docker run --rm -v C:\path\to\test:/data -v C:\path\to\out:/out firemon --data-dir /data --output /out/submission.csv
  ```
  Проверь сборку и запуск на `data/test_fake` реально (Docker Desktop). Если Docker на машине недоступен — напиши Dockerfile максимально стандартно и честно отметь в README, что образ не тестировался, но описан fallback-путь через venv.
- Веса: если суммарно ≤ 100 МБ — в git (`weights/`), иначе Git LFS или архив на облаке + `scripts/download_weights.py` с проверкой sha256. В README — точная инструкция и ожидаемое дерево `weights/`.

### 4. README.md — итоговый (критерий «Воспроизводимость» 5 баллов)
Обязательные разделы (в этом порядке): 1) Кратко о решении + схема пайплайна (картинка `reports/figures/pipeline.png` — сделай через matplotlib/graphviz/mermaid-рендер); 2) Результаты (таблица метрик OOF/temporal/public LB, если будет); 3) Требования к окружению (ОС, Python, CUDA опционально, RAM); 4) Установка (PowerShell и bash варианты, venv, pip); 5) Структура входных данных (дерево каталогов теста, где sample_submission); 6) Запуск инференса — одна команда, где появляется submission.csv, ожидаемое время на CPU/GPU; 7) Docker; 8) Веса — где взять, куда положить; 9) Обучение — `python train.py --module all`, время, что появится; 10) Сервис — команда запуска, URL, примеры curl (из промпта 6); 11) Тесты; 12) Структура репозитория; 13) Лицензии данных/моделей; 14) Команда.

### 5. Финальный сабмит
- Когда `data/test` и `baseline/sample_submission.csv` на месте: `python inference.py --data-dir data/test --output submissions/submission_<дата>_<коммит>.csv`, валидация, замер времени; лог в `reports/speed.md`. Файл для отправки — копия `submissions/final_submission.csv` + запись в `reports/submissions.md` (какие веса/конфиг/коммит, локальные OOF, public-скор после отправки — заполняет Марк).
- Прогони `scripts/check_submission.py <csv>` — обёртка над `validate_submission`, печатает «OK» и сводку (сколько пикселей на класс, сколько пустых масок — sanity: AF с пустыми масками должно быть ~25–30 %, BS-чипы без гари — 0).

### 6. Чистка репозитория (критерий «Структура» 4 балла)
- Удалить мёртвый код, `print`-отладку, дубли; docstring у публичных функций; параметры в yaml, не в коде; нет абсолютных путей (grep по `C:\\` и `/home/`); `ruff`/`black` прогон; `python -m pytest -q` зелёный; `.gitignore` корректный; `git log` осмысленный.

## Критерии готовности
- `inference.py` на реальном тесте → валидный файл, время задокументировано (GPU и CPU), два запуска побайтно равны;
- Dockerfile собран и запущен хотя бы на fake-тесте (или честная пометка);
- README полный, все команды из него выполнены буквально в чистом venv (создай второй venv и проверь!);
- `reports/speed.md`, `reports/submissions.md`, `reports/tasklog/05_inference.md`, коммит.
