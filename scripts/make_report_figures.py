"""Рисунки отчёта и презентации, которые строятся из итоговых чисел (reports/final_numbers.md).

    python scripts/make_report_figures.py      # → reports/figures/report_af_contrib.png, report_score_ladder.png, report_contrib.png

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
# лестница Score на OOF (final_numbers.md, A3, A4, B1): F1_af·0.35 + BS-часть
SCORE_LADDER = [
    ("baseline:\nдва правила", 0.3088),
    ("AF: пороги\nдень/ночь", 0.5374),
    ("AF:\nLightGBM", 0.5625),
    ("BS: пороги\nпо покрову", 0.5960),
    ("BS: LightGBM\nпо пикселям", 0.7283),
    ("BS: UNet →\nстек-LGBM", 0.7909),
    ("BS: событие,\n2-й UNet", 0.8248),
    ("BS: пост-\nобработка", 0.8344),
    ("v4:\nбез TTA", 0.8300),
    ("финал v5:\nмасштабы", 0.8353),
]
# вклад решений в Score на OOF (ΔScore), final_numbers.md B2
CONTRIB = [
    ("AF: пороги относительно фона, день/ночь", 0.229),
    ("BS: LightGBM по пикселям (104 признака) + декодирование", 0.132),
    ("BS: UNet выбирает регион → стек-LGBM", 0.063),
    ("BS: пороги степени по покрову", 0.033),
    ("BS: правило эталона, центр и локализация события", 0.027),
    ("AF: LightGBM на кандидатах", 0.025),
    ("BS: постобработка (MMU 25 px, дыры, 300 px от события)", 0.010),
    ("BS: второй UNet (усреднение)", 0.007),
    ("BS: многомасштабный ансамбль v5 (UNet на 0.75× и 0.5×, ворота 0.01)", 0.005),
    ("BS: отказ от TTA (результат не зависит от устройства)", -0.004),
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
    fig, ax = plt.subplots(figsize=(13.0, 4.4), dpi=170, facecolor=SURFACE)
    _style(ax)
    names = [s[0] for s in SCORE_LADDER]
    vals = [s[1] for s in SCORE_LADDER]
    bars = ax.bar(names, vals, width=0.56, color=SERIES, linewidth=0)
    prev = None
    for bar, v in zip(bars, vals):
        label = f"{v:.4f}" if prev is None else f"{v:.4f}\n{v - prev:+.3f}".replace("-", "−")
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, label, ha="center", va="bottom", fontsize=10.5, color=INK)
        prev = v
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Score (OOF, групповая валидация)", color=MUTED, fontsize=10)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.set_title("Score по шагам: 0.309 → 0.835 (отказ от TTA ради независимости от устройства, затем многомасштабный ансамбль v5)", loc="left",
                 fontsize=12, color=INK, fontweight="bold", pad=12)
    fig.tight_layout()
    out = FIG / "report_score_ladder.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def contrib() -> Path:
    fig, ax = plt.subplots(figsize=(10.4, 4.6), dpi=170, facecolor=SURFACE)
    _style(ax)
    names = [c[0] for c in CONTRIB][::-1]
    vals = [c[1] for c in CONTRIB][::-1]
    bars = ax.barh(names, vals, height=0.56, color=SERIES, linewidth=0)
    for bar, v in zip(bars, vals):
        ax.text(max(v, 0) + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:+.3f}".replace("-", "−"), va="center", fontsize=10.5,
                color=INK)
    ax.set_xlim(-0.012, 0.27)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("прирост Score (OOF) от решения", color=MUTED, fontsize=10)
    ax.tick_params(axis="y", labelsize=10.5, colors=INK)
    fig.suptitle("Вклад каждого решения в Score (OOF): всего +0.527 к baseline 0.309", x=0.012, ha="left", fontsize=12, color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / "report_contrib.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def bs_case_crops() -> list[Path]:
    """Строки из bs_error_cases.png (6 строк по 288 px) отдельными картинками для слайдов."""
    from PIL import Image

    src = Image.open(FIG / "bs_error_cases.png")
    row_h = src.height // 6
    out = []
    for name, row in (("bs_case_field.png", 0), ("bs_case_arc.png", 4), ("bs_case_cloud.png", 2)):
        src.crop((0, row * row_h + 40, src.width, (row + 1) * row_h)).save(FIG / name)
        out.append(FIG / name)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for path in [af_contrib(), score_ladder(), contrib(), *bs_case_crops()]:
        print(f"записано: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
