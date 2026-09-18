"""Проверка, что в итоговых документах не осталось незаполненных плейсхолдеров `{{…}}`.

    python scripts/check_placeholders.py                 # код 1, если что-то осталось
    python scripts/check_placeholders.py --allow FINAL:SCORE_PUBLIC

Проверяются отчёт, README, Q&A, сборщик слайдов, пакет передачи и текст слайдов deck.pptx (если установлен python-pptx).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "reports/report.md", "reports/final_numbers.md", "reports/submissions.md", "presentation/qa.md",
         "presentation/build_deck.py", "handoff/HANDOFF.md", "handoff/qa.md", "handoff/final_numbers.md"]
PATTERN = re.compile(r"\{\{([A-Z]+(?::[A-Z0-9_]+)?)\}\}")


def deck_text(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        return ""
    chunks = []
    for slide in Presentation(str(path)).slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                chunks.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                chunks += [c.text for r in shape.table.rows for c in r.cells]
        if slide.has_notes_slide:
            chunks.append(slide.notes_slide.notes_text_frame.text)
    return "\n".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", nargs="*", default=[], help="плейсхолдеры, которые пока разрешено оставить")
    args = ap.parse_args()
    found: dict[str, dict[str, int]] = {}
    texts = {f: (ROOT / f).read_text(encoding="utf-8") for f in FILES if (ROOT / f).exists()}
    deck = ROOT / "presentation" / "deck.pptx"
    if deck.exists():
        texts["presentation/deck.pptx"] = deck_text(deck)
    for name, txt in texts.items():
        for key in PATTERN.findall(txt):
            found.setdefault(key, {}).setdefault(name, 0)
            found[key][name] += 1
    bad = {k: v for k, v in found.items() if k not in args.allow}
    for key, where in sorted(found.items()):
        mark = "разрешён" if key in args.allow else "ОСТАЛСЯ"
        print(f"{{{{{key}}}}} — {mark}: " + ", ".join(f"{f} ×{n}" for f, n in where.items()))
    print(f"проверено файлов: {len(texts)}; незаполненных плейсхолдеров: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
