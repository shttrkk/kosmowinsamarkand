"""Схема пайплайна для README/отчёта → reports/figures/pipeline.png.

    python scripts/make_pipeline_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "pipeline.png"

INK, MUTED = "#1f2933", "#52606d"
FILL = {"data": "#e8eef5", "af": "#fde8d7", "bs": "#dcefe2", "out": "#ece6f5", "svc": "#f3f0e6"}
EDGE = {"data": "#5b7a99", "af": "#c8642a", "bs": "#2f855a", "out": "#6b4fa0", "svc": "#8a7a3d"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.4,
                                facecolor=FILL[kind], edgecolor=EDGE[kind]))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center", va="top", fontsize=8.6, color=MUTED, linespacing=1.35)


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color=MUTED,
                                 shrinkA=2, shrinkB=2))


def main() -> int:
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=160)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    box(ax, 0.2, 3.6, 2.6, 1.75, "AF-чип 256×256 @375 м",
        ["VIIRS I1–I5, зениты, valid", "landcover, DEM, метео", "NaN bow-tie → предсказание 0"], "data")
    box(ax, 0.2, 1.0, 2.6, 1.9, "BS-чип 512×512 @20 м",
        ["Sentinel-2 pre/post (10 каналов)", "Sentinel-1 pre/post (VV, VH)", "DEM, slope, landcover"], "data")

    box(ax, 3.4, 3.6, 3.0, 1.75, "Кандидаты + признаки AF",
        ["день: I4>314 K & Δ>17 K", "ночь: I4>290 K & Δ>5 K", "110 признаков: контекст окон,", "z-score, блик I1–I3, aux"], "af")
    box(ax, 7.0, 3.6, 2.9, 1.75, "LightGBM × 5 фолдов",
        ["среднее вероятностей", "порог по OOF", "OOF F1 0.968 / temporal 0.960"], "af")

    box(ax, 3.4, 1.0, 3.0, 1.9, "Признаки BS",
        ["NBR, dNBR, RdNBR, индексы", "маска SCL (облака, тени)", "ΔVV, ΔVH (медианный фильтр)", "landcover → пороги по покрову"], "bs")
    box(ax, 7.0, 1.0, 2.9, 1.9, "BS-бэкенд (реестр)",
        ["сейчас: пороги dNBR (rules)", "далее: BSPredictor (task/bs)", "постобработка: MMU 25 px,", "SCL_post 0/3/9/11 → 0"], "bs")

    box(ax, 10.5, 2.75, 2.8, 1.9, "submission.csv",
        ["RLE с 1, row-major", "447 строк по шаблону", "валидация записанного файла", "inference.py: ~10 с на CPU"], "out")
    box(ax, 10.5, 0.55, 2.8, 1.45, "Сервис (FastAPI + Leaflet)",
        ["термоточки, контуры гарей", "GeoJSON / Shapefile, справка, га"], "svc")

    arrow(ax, (2.8, 4.5), (3.4, 4.5))
    arrow(ax, (6.4, 4.5), (7.0, 4.5))
    arrow(ax, (2.8, 1.95), (3.4, 1.95))
    arrow(ax, (6.4, 1.95), (7.0, 1.95))
    arrow(ax, (9.9, 4.4), (10.5, 4.0))
    arrow(ax, (9.9, 2.3), (10.5, 3.2))
    arrow(ax, (9.9, 1.5), (10.5, 1.3))
    ax.text(6.75, 5.95, "Мониторинг природных пожаров: от чипа до submission.csv и сервиса", ha="center", va="top",
            fontsize=12.5, fontweight="bold", color=INK)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"записано: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
