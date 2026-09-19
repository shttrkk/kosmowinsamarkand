"""Экспорт отчёта и презентации в PDF через LibreOffice.

    pip install markdown               # только для экспорта, в requirements проекта не входит
    python scripts/export_pdf.py       # → reports/report.pdf, presentation/deck.pdf

report.md → HTML (python-markdown, таблицы и блоки кода) → PDF; deck.pptx → PDF. Путь к LibreOffice: переменная окружения
SOFFICE или стандартные места установки.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = """
body { font-family: 'Liberation Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.35; color: #111; }
h1 { font-size: 19pt; margin: 0 0 8pt 0; } h2 { font-size: 14pt; margin: 16pt 0 6pt 0; color: #1d4f91; }
h3 { font-size: 11.5pt; margin: 11pt 0 4pt 0; } p { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.5pt; }
th { background: #2a78d6; color: #fff; padding: 3pt 4pt; border: 0.5pt solid #9ab; text-align: left; }
td { padding: 2.5pt 4pt; border: 0.5pt solid #bbb; vertical-align: top; }
pre { background: #f3f4f6; padding: 5pt; font-size: 8pt; font-family: 'Liberation Mono', Consolas, monospace; }
code { font-family: 'Liberation Mono', Consolas, monospace; font-size: 8.5pt; }
blockquote { margin: 6pt 0; padding: 2pt 8pt; border-left: 2pt solid #2a78d6; color: #333; }
em { color: #444; } img { display: block; margin: 6pt auto 2pt auto; }
"""


def find_soffice() -> str:
    candidates = [os.environ.get("SOFFICE"), shutil.which("soffice"), shutil.which("libreoffice"),
                  r"C:\Program Files\LibreOffice\program\soffice.exe", "/usr/bin/soffice",
                  "/Applications/LibreOffice.app/Contents/MacOS/soffice"]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("LibreOffice не найден: установите его или задайте переменную окружения SOFFICE")


def convert(soffice: str, src: Path, out_dir: Path, fmt: str = "pdf", infilter: str | None = None) -> Path:
    cmd = [soffice, "--headless"] + ([f"--infilter={infilter}"] if infilter else [])
    subprocess.run(cmd + ["--convert-to", fmt, "--outdir", str(out_dir), str(src)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    out = out_dir / (src.stem + ".pdf")
    if not out.exists():
        raise RuntimeError(f"LibreOffice не создал {out}")
    return out


def report_pdf(soffice: str) -> Path:
    import markdown
    from PIL import Image

    md = (ROOT / "reports" / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])

    def size_img(m: re.Match) -> str:  # LibreOffice не понимает max-width: задаём размер явно, высота — не больше страницы
        src = m.group(2)
        w, h = Image.open(ROOT / "reports" / src).size
        width = min(560, int(700 * w / h)) if h else 560  # поле A4 ≈ 600 px при 96 dpi; высота ≤ 700 px
        return f'<img {m.group(1)}src="{src}" width="{width}" height="{round(width * h / w)}"'

    body, n_img = re.subn(r'<img ((?:alt="[^"]*" )?)src="([^"]+)"', size_img, body)  # python-markdown пишет alt перед src
    if n_img != md.count("]("):
        print(f"внимание: размеры заданы {n_img} рисункам из {md.count('](')} ссылок")
    html = ROOT / "reports" / "_report_export.html"  # рядом с figures/, чтобы работали относительные пути
    html.write_text(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>",
                    encoding="utf-8")
    try:
        # импорт HTML в Writer (а не в Writer/Web): страницы A4, нормальная вёрстка таблиц
        pdf = convert(soffice, html, ROOT / "reports", "pdf:writer_pdf_Export", infilter="HTML (StarWriter)")
    finally:
        html.unlink(missing_ok=True)
    final = ROOT / "reports" / "report.pdf"
    pdf.replace(final)
    return final


def main() -> int:
    soffice = find_soffice()
    outs = [report_pdf(soffice), convert(soffice, ROOT / "presentation" / "deck.pptx", ROOT / "presentation")]
    for out in outs:
        print(f"записано: {out} ({out.stat().st_size / 1e6:.1f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
