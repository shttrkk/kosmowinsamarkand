"""Рисунки отчёта и презентации, которые строятся из итоговых чисел (reports/final_numbers.md).

    python scripts/make_report_figures.py      # → reports/figures/report_af_contrib.png, report_score_ladder.png

Числа продублированы здесь намеренно: источник истины — reports/final_numbers.md, при его изменении поправить константы.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

# абляции AF (final_numbers.md, A4): компонент → (Δ temporal, Δ OOF), потеря F1 при удалении компонента
AF_ABLATIONS = [
    ("hard negatives у границы очага", 0.049, 0.040),
    ("aux: покров, рельеф, метео, расстояния", 0.026, 0.008),
    ("контекстные признаки окон", 0.013, 0.010),
    ("окна 5/11/15 (оставлено только 7)", 0.006, 0.005),
    ("обучение только на кандидатах", 0.006, 0.006),
    ("дневные признаки блика", 0.000, 0.002),
]
# лестница Score на train/OOF (final_numbers.md, A3, A4, A7 и report.md §6)
SCORE_LADDER = [
    ("baseline:\nдва правила", 0.3088),
    ("AF: пороги\nдень/ночь + фон", 0.5374),
    ("AF: LightGBM\n(отправка №2)", 0.5625),
    ("BS: пороги\nпо покрову", 0.5858),
    ("BS: LightGBM\n16 признаков", 0.6574),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)


def af_contrib() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=170, facecolor=SURFACE)
    _style(ax)
    names = [a[0] for a in AF_ABLATIONS][::-1]
    temporal = [a[1] for a in AF_ABLATIONS][::-1]
    oof = [a[2] for a in AF_ABLATIONS][::-1]
    bars = ax.barh(names, temporal, height=0.52, color=SERIES, linewidth=0)
    for bar, t, o in zip(bars, temporal, oof):
        ax.text(max(t, 0) + 0.0012, bar.get_y() + bar.get_height() / 2, (f"−{t:.3f}" if t else "0.000") + f"   (OOF −{o:.3f})", va="center",
                fontsize=10, color=INK)
    ax.set_xlim(0, 0.068)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("потеря F1 на временном сплите (2024) при удалении компонента", color=MUTED, fontsize=10)
    ax.tick_params(axis="y", labelsize=10.5, colors=INK)
    ax.set_title("AF: потеря F1 при удалении компонента (base: 2024 год 0.9560)", loc="left", fontsize=12,
                 color=INK, fontweight="bold", pad=12)
    fig.tight_layout()
    out = FIG / "report_af_contrib.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def score_ladder() -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=170, facecolor=SURFACE)
    _style(ax)
    names = [s[0] for s in SCORE_LADDER]
    vals = [s[1] for s in SCORE_LADDER]
    bars = ax.bar(names, vals, width=0.56, color=SERIES, linewidth=0)
    prev = None
    for bar, v in zip(bars, vals):
        label = f"{v:.4f}" if prev is None else f"{v:.4f}\n+{v - prev:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, label, ha="center", va="bottom", fontsize=10.5, color=INK)
        prev = v
    ax.set_ylim(0, 0.80)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Score (train / OOF)", color=MUTED, fontsize=10)
    ax.tick_params(axis="x", labelsize=9.5, colors=INK)
    ax.set_title("Score по шагам (BS-шаги — диагностические модели EDA)", loc="left",
                 fontsize=12, color=INK, fontweight="bold", pad=12)
    fig.tight_layout()
    out = FIG / "report_score_ladder.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for path in (af_contrib(), score_ladder()):
        print(f"записано: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
