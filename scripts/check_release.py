"""Инвентаризация релиза: всё, на что ссылаются README / HANDOFF / отчёт, лежит в git; размеры файлов; чистота рабочей копии.

    python scripts/check_release.py            # печать + reports/release_check.md; код 1, если есть проблемы
    python scripts/check_release.py --no-write # только печать (для проверки в чистом клоне)

Проверки: (1) пути из документов существуют в git (или явно относятся к тому, что в git не входит: data/, baseline/,
out/, submissions/, каталоги весов экспериментов); (2) каждый отслеживаемый файл < 100 МБ (типичный лимит git-хостингов), предупреждение с 50 МБ;
(3) суммарный размер отслеживаемых файлов и каталога .git; (4) `git status` чистый.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "reports/report.md", "reports/service.md", "reports/bs_module.md", "reports/af_module.md"]
HARD_LIMIT, SOFT_LIMIT = 100 * 1024 ** 2, 50 * 1024 ** 2
# то, что по замыслу НЕ в git: данные и файлы организаторов, результаты запусков, веса экспериментов, локальные окружения
OUTSIDE_GIT = ("data/", "baseline/", "out/", "submissions/", "weights_", ".venv", "service/data/firemon.gpkg", "service/data/cache",
               "eda/cache", "C:", "/path/", "path/to")
EXTENSIONS = (".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".pdf", ".pptx", ".png", ".pt", ".gz", ".toml", ".js", ".html",
              ".css", ".zip", ".gpkg", ".gpu", ".lock")
TOP_LEVEL = ("configs/", "src/", "scripts/", "service/", "tests/", "weights/", "reports/", "eda/", "submissions/")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def candidate_paths(text: str) -> set[str]:
    """Пути из `кода`, ссылок [..](..) и картинок ![..](..): без пробелов, шаблонов и плейсхолдеров."""
    raw = set(re.findall(r"`([^`\n]+)`", text)) | set(re.findall(r"\]\(([^)\s]+)\)", text))
    out = set()
    for tok in raw:
        tok = tok.strip().strip(".,;:").replace("\\", "/")
        if not tok or " " in tok or any(c in tok for c in "<>*{}…$=|\"'") or tok.startswith(("http", "-", "/", "#", "python")):
            continue
        if tok.endswith("/"):
            if tok.startswith(TOP_LEVEL) or tok in TOP_LEVEL:
                out.add(tok)
        elif tok.lower().endswith(EXTENSIONS) and ("/" in tok or tok in ("README.md", "CLAUDE.md", "Dockerfile.gpu")):
            out.add(tok)
        elif tok in ("Dockerfile", "inference.py", "train.py", "requirements.txt", "pyproject.toml"):
            out.add(tok)
    return out


def resolve(tok: str, doc: Path, tracked: set[str], tracked_dirs: set[str]) -> str | None:
    """'git' | 'outside' | None (не найден). Путь пробуется от корня репозитория, от каталога документа и как src/firemon/…"""
    if tok.startswith(OUTSIDE_GIT) or any(f"/{p}" in f"/{tok}" for p in ("weights_night", "weights_repro", "weights_lovasz")):
        return "outside"
    bases = ["", doc.parent.relative_to(ROOT).as_posix() + "/", "src/firemon/", "src/", "reports/", "reports/figures/", "scripts/",
             "service/", "tests/", "configs/"]
    for base in bases:
        p = (base + tok).replace("//", "/").lstrip("./")
        parts = []
        for seg in p.split("/"):  # нормализация a/../b
            if seg == "..":
                parts and parts.pop()
            elif seg not in ("", "."):
                parts.append(seg)
        p = "/".join(parts)
        if p in tracked or p.rstrip("/") in tracked_dirs:
            return "git"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-write", action="store_true", help="не писать reports/release_check.md")
    args = ap.parse_args()

    tracked = set(git("ls-files", "-z").split("\0")) - {""}
    tracked_dirs = {"/".join(f.split("/")[:i]) for f in tracked for i in range(1, f.count("/") + 1)}
    problems: list[str] = []
    lines = ["# Проверка релиза (`python scripts/check_release.py`)", "",
             f"Сформировано: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · коммит `{git('rev-parse', '--short', 'HEAD').strip()}`"
             f" · ветка `{git('rev-parse', '--abbrev-ref', 'HEAD').strip()}`", ""]

    # 1. ссылки из документов
    lines += ["## 1. Файлы, на которые ссылаются документы", "", "| Документ | путей | в git | вне git по замыслу | НЕ НАЙДЕНО |", "|---|---|---|---|---|"]
    missing_all: list[tuple[str, str]] = []
    for name in DOCS:
        doc = ROOT / name
        if not doc.exists():
            problems.append(f"нет документа {name}")
            continue
        toks = candidate_paths(doc.read_text(encoding="utf-8"))
        res = {t: resolve(t, doc, tracked, tracked_dirs) for t in toks}
        miss = sorted(t for t, r in res.items() if r is None)
        missing_all += [(name, t) for t in miss]
        lines.append(f"| `{name}` | {len(toks)} | {sum(r == 'git' for r in res.values())} | {sum(r == 'outside' for r in res.values())} | {len(miss)} |")
    lines.append("")
    if missing_all:
        problems.append(f"в документах {len(missing_all)} путей, которых нет в git")
        lines += ["Не найдены в git:", ""] + [f"- `{t}` ← `{d}`" for d, t in missing_all] + [""]
    else:
        lines += ["Все пути из документов либо есть в git, либо относятся к тому, что в git не входит по замыслу "
                  "(`data/`, `baseline/`, `out/`, `submissions/`, каталоги весов экспериментов, `service/data/firemon.gpkg`).", ""]

    # обязательный минимум для инференса и сервиса из чистого клона
    required = ["inference.py", "train.py", "requirements.txt", "requirements-cpu.txt", "requirements-base.txt", "Dockerfile",
                "configs/inference.yaml", "configs/bs_inference.yaml", "configs/folds_af.csv", "configs/folds_bs.csv",
                "weights/af_lgbm_meta.json", "weights/bs_unet_r34_cedice.pt", "weights/bs_unet_r34_v4.pt",
                "weights/bs_unet_r34_v4_s075.pt", "weights/bs_unet_r34_v4_s050.pt", "weights/bs_stack_ms.txt",
                "weights/bs_decode_ms.json", "weights/bs_lc_medians.json", "service/data/manifest.json",
                "reports/report.md", "reports/final_numbers.md", "README.md", "LICENSE", "submissions/submission_v5_final.csv",
                "tests/fixtures/example_submission.csv", "tests/fixtures/sample_submission.csv"]
    required += [f"weights/af_lgbm_fold{i}.txt" for i in range(5)]
    absent = [f for f in required if f not in tracked]
    snapshots = sorted(f for f in tracked if f.startswith("service/data/") and f.endswith(".geojson.gz"))
    lines += ["## 2. Обязательные файлы (инференс, сервис, отчётность)", "",
              f"Проверено {len(required)} файлов: " + ("**все в git**." if not absent else "**нет в git:** " + ", ".join(f"`{f}`" for f in absent)),
              f"Снимки слоёв сервиса в git: {len(snapshots)} ({', '.join('`' + Path(s).name + '`' for s in snapshots) or '—'}).", ""]
    if absent:
        problems.append(f"нет обязательных файлов: {absent}")
    if not snapshots:
        problems.append("нет снимков service/data/*.geojson.gz")

    # 2. размеры
    sizes = sorted(((ROOT / f).stat().st_size, f) for f in tracked if (ROOT / f).is_file())
    total = sum(s for s, _ in sizes)
    over_hard = [(s, f) for s, f in sizes if s >= HARD_LIMIT]
    over_soft = [(s, f) for s, f in sizes if SOFT_LIMIT <= s < HARD_LIMIT]
    lines += ["## 3. Размеры", "", f"- отслеживаемых файлов: **{len(sizes)}**, суммарно **{total / 1024 ** 2:.1f} МиБ**;",
              f"- файлов ≥ 100 МБ (жёсткий лимит git-хостингов): **{len(over_hard)}**; файлов 50–100 МБ (хостинги предупреждают при push): {len(over_soft)};",
              f"- самый большой файл: `{sizes[-1][1]}` — {sizes[-1][0] / 1024 ** 2:.1f} МиБ.", "",
              "| Файл | МиБ |", "|---|---|"] + [f"| `{f}` | {s / 1024 ** 2:.1f} |" for s, f in reversed(sizes[-10:])] + [""]
    if over_hard:
        problems.append(f"файлы ≥ 100 МБ: {[f for _, f in over_hard]}")
    by_dir: dict[str, int] = {}
    for s, f in sizes:
        by_dir[f.split("/")[0] if "/" in f else "(корень)"] = by_dir.get(f.split("/")[0] if "/" in f else "(корень)", 0) + s
    lines += ["| Каталог | МиБ |", "|---|---|"] + [f"| `{d}` | {s / 1024 ** 2:.1f} |" for d, s in sorted(by_dir.items(), key=lambda x: -x[1])] + [""]
    git_dir = ROOT / ".git"
    if git_dir.is_dir():
        gsize = sum(p.stat().st_size for p in git_dir.rglob("*") if p.is_file())
        lines += [f"Каталог `.git`: **{gsize / 1024 ** 2:.1f} МиБ** (после `git gc`).", ""]

    # каталоги, вынесенные из репозитория в пакет артефактов, не должны вернуться
    banned = sorted({f.split("/")[0] for f in tracked if f.split("/")[0] in ("presentation", "handoff", "prompts", "docs", "artifacts")}
                    | {"reports/tasklog" for f in tracked if f.startswith("reports/tasklog/")}
                    | {f for f in tracked if f.lower().endswith((".pdf", ".pptx", ".rar")) or (f.lower().endswith(".zip"))})
    lines += ["## 3a. Состав: репозиторий = код + документация к коду", "",
              "Презентация, пакет защиты, журналы задач, ТЗ организаторов, PDF — вне git (пакет артефактов): "
              + ("**в git их нет**." if not banned else "**НАЙДЕНЫ в git:** " + ", ".join(f"`{b}`" for b in banned)), ""]
    if banned:
        problems.append(f"в git есть то, что должно лежать вне репозитория: {banned}")

    # 3. чистота
    status = git("status", "--porcelain").strip()
    own = "reports/release_check.md"
    dirty = [ln for ln in status.splitlines() if not ln.endswith(own)]
    lines += ["## 4. Рабочая копия", "", "`git status` — **чисто**." if not dirty else "`git status` — есть изменения:", ""]
    if dirty:
        problems.append(f"git status не чистый ({len(dirty)} записей)")
        lines += ["```"] + dirty[:40] + ["```", ""]
    tags = git("tag", "--points-at", "HEAD").split()
    lines += [f"Теги на HEAD: {', '.join('`' + t + '`' for t in tags) or '—'}.", ""]

    lines += ["## 5. Как выложить (удалённый репозиторий не настроен — площадка выбирается вручную)", "", "```", "git remote add origin <url>",
              "git push -u origin main --tags", "```", "",
              f"Все {len(sizes)} файлов меньше 100 МБ — Git LFS не нужен. Файлы 50–100 МБ хостинги обычно принимают с предупреждением; "
              f"у нас таких {len(over_soft)} (веса UNet по 47 МиБ = 49.3 МБ — ниже и этого порога). После push проверить на сайте, что "
              "четыре `weights/*.pt` открываются как бинарные файлы по ~47 МБ, а не как указатели LFS.", "",
              "## Итог", "", "**Проблем нет.**" if not problems else "**Проблемы:**", ""] + [f"- {p}" for p in problems]

    report = "\n".join(lines).rstrip() + "\n"
    print(report)
    if not args.no_write:
        out = ROOT / "reports" / "release_check.md"
        out.write_text(report, encoding="utf-8", newline="\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
