"""Единственная точка входа инференса (ТЗ):

    python inference.py --data-dir /path/to/test --output /path/to/submission.csv

Необязательные аргументы: --sample-submission CSV, --config YAML, --device auto|cpu|cuda, --workers N, --profile.

Порядок работы: конфиг → шаблон sample_submission.csv (он определяет состав строк) → AF-чипы → BS-чипы →
сборка submission → валидация записанного файла → времена этапов. Без интерактива. Коды выхода:
0 — файл записан и прошёл валидацию; 1 — файл записан, но не прошёл валидацию; 2 — нет входных данных/конфига.
Модели подключаются в `src/firemon/pipeline.py` (реестры AF_BACKENDS / BS_BACKENDS), выбор — в configs/inference.yaml.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

T_START = time.perf_counter()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # работает и без `pip install -e .`

from firemon.io import LAYERS, list_chips  # noqa: E402
from firemon.pipeline import BACKENDS, Profiler, RunContext, missing_layers, prewarm_torch, run_backend  # noqa: E402
from firemon.submission import (build_submission, chip_kind, read_submission, shapes_from_meta,  # noqa: E402
                                submission_rows, validate_submission, write_submission)
from firemon.utils import auto_threads, get_logger, load_config, seed_everything  # noqa: E402

log = get_logger("inference")
EXIT_OK, EXIT_INVALID, EXIT_NO_INPUT = 0, 1, 2


class InputError(Exception):
    """Нет входных данных/конфига — понятное сообщение и код выхода 2 вместо трассировки."""


def find_sample_submission(data_dir: Path, explicit: str | None) -> Path | None:
    """Шаблон: --sample-submission → <data-dir>/sample_submission.csv → baseline/sample_submission.csv."""
    if explicit:
        if not Path(explicit).is_file():
            raise InputError(f"--sample-submission: файл не найден: {explicit}")
        return Path(explicit)
    for c in (data_dir / "sample_submission.csv", ROOT / "baseline" / "sample_submission.csv"):
        if c.is_file():
            return c
    return None


def chips_by_kind(data_dir: Path, sample: Path | None) -> dict[str, list[str]]:
    """Чипы к обработке: ровно те, что в шаблоне (порядок шаблона); без шаблона — из meta.csv/имён файлов."""
    if sample is not None:
        ids = list(dict.fromkeys(read_submission(sample)["chip_id"]))
        return {kind: [c for c in ids if chip_kind(c) == kind] for kind in BACKENDS}
    log.warning("sample_submission.csv не найден — строки формируются по списку чипов из meta.csv/каталогов")
    out = {}
    for kind in BACKENDS:
        try:
            out[kind] = list_chips(data_dir, kind)
        except (FileNotFoundError, OSError):
            out[kind] = []
    return out


def run(data_dir: Path, output: Path, sample: Path | None, cfg: dict, ctx: RunContext) -> int:
    prof = ctx.profiler
    with prof.stage("шаблон и meta.csv"):
        chips = chips_by_kind(data_dir, sample)
        ctx.shapes = shapes_from_meta(data_dir)
    if not any(chips.values()):
        raise InputError(f"в {data_dir} нет ни шаблона sample_submission.csv, ни чипов af/ и bs/")
    # отдельные отсутствующие чипы — пустая маска и warning; но если на диске нет НИ ОДНОГО чипа из шаблона,
    # это почти наверняка неверный --data-dir: лучше понятная ошибка, чем «успешный» пустой submission
    if not any(not missing_layers(data_dir, kind, cid, layers=(next(iter(LAYERS[kind])),))
               for kind, ids in chips.items() for cid in ids):
        raise InputError(f"в {data_dir} не найдено ни одного чипа из шаблона {sample} — проверьте --data-dir "
                         f"(ожидаются подкаталоги af/viirs, bs/sentinel2_pre, …)")

    if chips.get("bs"):
        prewarm_torch(cfg, ctx)  # импорт torch/smp и CUDA-контекст — в фоне, пока считается AF
    preds, n_failed = {}, 0
    for kind in BACKENDS:
        with prof.stage(f"{kind}: всего"):
            preds[kind], failed, used = run_backend(kind, data_dir, chips[kind], cfg.get(kind) or {}, ctx)
        n_failed += failed
        log.info("%s: %d чипов, бэкенд %s, %.2f с%s", kind.upper(), len(chips[kind]), used,
                 prof.seconds(f"{kind}: всего"), f", не прочитано: {failed}" if failed else "")

    with prof.stage("RLE + запись csv"):
        if sample is not None:
            n_rows = build_submission(preds["af"], preds["bs"], sample, output, strict=False)
        else:
            rows = submission_rows(preds["af"], preds["bs"])
            write_submission(rows, output)
            n_rows = len(rows)
    with prof.stage("валидация записанного файла"):
        errors = validate_submission(output, sample or output, ctx.shapes)

    log.info("submission: %s (%d строк), чипов с пустой маской из-за ошибок чтения: %d", output, n_rows, n_failed)
    for e in errors[:20]:
        log.error("валидация: %s", e)
    if errors:
        log.error("submission НЕ прошёл валидацию (%d ошибок)", len(errors))
    return EXIT_INVALID if errors else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Инференс AF + BS → submission.csv")
    ap.add_argument("--data-dir", required=True, type=Path, help="каталог теста: af/, bs/, meta.csv, sample_submission.csv")
    ap.add_argument("--output", required=True, type=Path, help="куда записать submission.csv (каталоги создаются)")
    ap.add_argument("--sample-submission", default=None,
                    help="по умолчанию <data-dir>/sample_submission.csv, затем baseline/sample_submission.csv")
    ap.add_argument("--config", default=str(ROOT / "configs" / "inference.yaml"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="устройство для нейросетевых бэкендов (LightGBM и правила всегда на CPU)")
    ap.add_argument("--workers", type=int, default=None,
                    help="потоки чтения/признаков (по умолчанию из конфига: auto = min(число ядер, 16))")
    ap.add_argument("--profile", action="store_true", help="напечатать таблицу времени по этапам")
    args = ap.parse_args(argv)

    prof = Profiler()
    prof.add("старт Python + импорты", time.perf_counter() - T_START)
    try:
        if not args.data_dir.is_dir():
            raise InputError(f"--data-dir: каталог не найден: {args.data_dir}")
        if not Path(args.config).is_file():
            raise InputError(f"--config: файл не найден: {args.config}")
        cfg = load_config(args.config)
        seed_everything(int(cfg.get("seed", 42)), with_torch=False)  # torch сидируют сами torch-бэкенды
        workers = args.workers or auto_threads(cfg.get("workers"))  # auto → min(число ядер, 16)
        ctx = RunContext(workers=max(1, workers), device=args.device, profiler=prof)
        sample = find_sample_submission(args.data_dir, args.sample_submission)
        code = run(args.data_dir, args.output, sample, cfg, ctx)
    except InputError as e:
        log.error("%s", e)
        return EXIT_NO_INPUT
    total = time.perf_counter() - T_START
    if args.profile:
        log.info("профиль (этапы внутри пула потоков — сумма по %d потокам):\n%s", ctx.workers, prof.table(total))
    log.info("Полное время инференса: %.2f с", total)
    return code


if __name__ == "__main__":
    sys.exit(main())
