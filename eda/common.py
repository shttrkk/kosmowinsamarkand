"""Общие вещи для EDA-скриптов: пути, стиль фигур, палитра, словари landcover, мелкие утилиты.

Запуск скриптов — из корня репозитория: `python eda/eda_af.py`, `python eda/eda_bs.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "train"
CACHE_DIR = ROOT / "eda" / "cache"
TABLE_DIR = ROOT / "eda" / "tables"
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42
N_THREADS = 8

# категориальная палитра (фиксированный порядок, проверена на различимость при дальтонизме)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
PALETTE = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED]
C_INK, C_MUTED, C_GRID, C_SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
# степень поражения: 0 — нейтральный серый, 1→3 — одна «тёплая» шкала светлый→тёмный
SEV_COLORS = {0: "#b9b8b2", 1: "#eda100", 2: "#eb6834", 3: "#a3201f"}
SEV_NAMES = {0: "0 фон", 1: "1 слабая", 2: "2 средняя", 3: "3 сильная"}

LC_NAMES = {10: "лес", 20: "кустарник", 30: "трава", 40: "пашня", 50: "застройка", 60: "голая почва",
            70: "снег", 80: "вода", 90: "болото", 95: "мангры", 100: "мох"}
LC_COLORS = {10: "#008300", 20: "#1baf7a", 30: "#eda100", 40: "#e87ba4", 50: "#e34948", 60: "#898781",
             70: "#cde2fb", 80: "#2a78d6", 90: "#4a3aa7", 95: "#199e70", 100: "#c98500"}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE, "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.titlesize": 17,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": C_INK, "text.color": C_INK,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.grid": True, "grid.color": C_GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "legend.frameon": False, "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def ensure_dirs() -> None:
    for d in (CACHE_DIR, TABLE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str, tight: bool = True) -> None:
    path = FIG_DIR / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.relative_to(ROOT)}")


def save_table(df: pd.DataFrame, name: str, index: bool = False, floatfmt: str = "%.4g") -> pd.DataFrame:
    path = TABLE_DIR / name
    df.to_csv(path, index=index, float_format=floatfmt, encoding="utf-8")
    print(f"  [tab] {path.relative_to(ROOT)}  ({len(df)} строк)")
    return df


def save_numbers(numbers: dict, name: str) -> None:
    """Ключевые числа раздела → json (используются в reports/eda.md и eda_key_numbers.md)."""
    def conv(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [conv(x) for x in v]
        return v
    path = TABLE_DIR / name
    path.write_text(json.dumps(conv(numbers), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [num] {path.relative_to(ROOT)}")


def quantile_row(x: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, **{f"q{q:02d}": np.nan for q in qs}}
    return {"n": int(x.size), **{f"q{q:02d}": float(v) for q, v in zip(qs, np.percentile(x, qs))}}


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC-AUC одиночного признака (NaN отбрасываются); 0.5 = бесполезен, <0.5 = обратная связь."""
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(score)
    if ok.sum() == 0 or len(np.unique(label[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(label[ok], score[ok]))


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1e-12)


def log_hist(ax, data: dict, bins, xlabel: str, title: str, colors=None) -> None:
    """Нормированные гистограммы (ступеньки) в лог-масштабе по Y."""
    for i, (name, x) in enumerate(data.items()):
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        h, e = np.histogram(x, bins=bins, density=True)
        ax.step(e[:-1], np.where(h > 0, h, np.nan), where="post", label=f"{name} (n={x.size:,})".replace(",", " "),
                color=(colors or PALETTE)[i])
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("плотность (лог)")
    ax.set_title(title)
    ax.legend()
