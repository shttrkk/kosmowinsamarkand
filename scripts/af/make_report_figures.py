"""Рисунки для reports/af_module.md: важность признаков и разбор ошибок по OOF-предсказаниям.
Запуск: python scripts/af/make_report_figures.py  → reports/figures/af_*.png"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from firemon.io import read_af_chip, read_meta  # noqa: E402
from firemon.models.af.dataset import load_or_build  # noqa: E402
from firemon.models.af.train_lgbm import run_cv  # noqa: E402
from firemon.splits import load_folds  # noqa: E402
from firemon.utils import REPO_ROOT, seed_everything  # noqa: E402

FIG = REPO_ROOT / "reports" / "figures"
INK, MUTED, BLUE = "#1f2933", "#7b8794", "#2f6fb0"
LC_COLORS = {10: "#2e7d32", 20: "#9ccc65", 30: "#dce775", 40: "#f0c987", 50: "#c62828", 60: "#bdbdbd",
             80: "#1e88e5", 90: "#4db6ac"}
LC_LABELS = {10: "лес", 20: "кустарник", 30: "трава", 40: "пашня", 50: "застройка", 60: "голая почва",
             80: "вода", 90: "болото"}


def importance_figure() -> None:
    rep = json.loads((REPO_ROOT / "reports" / "af_train_report.json").read_text(encoding="utf-8"))
    names, gains = zip(*rep["importance"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(range(len(names)), [100 * g for g in gains], color=BLUE, height=0.6)
    ax.set_yticks(range(len(names)), names, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("доля суммарного gain, %", color=INK)
    ax.set_title("AF LightGBM: важность признаков (gain, сумма по 5 фолдам), top-25", color=INK, fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "af_feature_importance.png", dpi=130)
    plt.close(fig)


def error_panels(ds: dict, meta, oof: np.ndarray, thr: float, n_panels: int = 4) -> list[str]:
    y = ds["y"].astype(bool)
    pred = np.nan_to_num(oof, nan=0.0) > thr
    err = np.bincount(ds["chip_idx"][pred != y], minlength=len(meta))
    chosen = list(np.argsort(-err)[:n_panels])
    out = []
    for rank, ci in enumerate(chosen):
        cid = meta["chip_id"].iloc[ci]
        chip = read_af_chip(REPO_ROOT / "data" / "train", cid, with_mask=True)
        r = ds["chip_idx"] == ci
        pm = np.zeros((256, 256), bool)
        pm[ds["yy"][r & pred], ds["xx"][r & pred]] = True
        gt = chip["mask"] == 1
        ys, xs = np.nonzero(pm ^ gt)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        half = 24
        y0, x0 = int(np.clip(cy - half, 0, 256 - 2 * half)), int(np.clip(cx - half, 0, 256 - 2 * half))
        win = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        i4, di = chip["viirs"][3][win], (chip["viirs"][3] - chip["viirs"][4])[win]
        lc = chip["aux"][0][win]
        codes = sorted(LC_COLORS)
        lc_idx = np.searchsorted(codes, np.nan_to_num(lc, nan=codes[0]))
        res = np.zeros(gt[win].shape, int)                  # 0 фон, 1 TP, 2 FP, 3 FN
        res[gt[win] & pm[win]], res[~gt[win] & pm[win]], res[gt[win] & ~pm[win]] = 1, 2, 3
        fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
        night = bool(np.nanmean(chip["viirs"][5]) > 90)
        panels = [(i4, "I4, K", "inferno"), (di, "I4 − I5, K", "inferno")]
        for ax, (a, title, cmap) in zip(axes, panels):
            im = ax.imshow(a, cmap=cmap, interpolation="nearest")
            ax.set_title(title, fontsize=10, color=INK, loc="left")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[2].imshow(lc_idx, cmap=ListedColormap([LC_COLORS[c] for c in codes]), vmin=0, vmax=len(codes) - 1,
                       interpolation="nearest")
        present = [c for c in codes if (lc == c).any()]
        axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LC_COLORS[c]) for c in present],
                       labels=[LC_LABELS[c] for c in present], fontsize=7, loc="lower right", framealpha=0.9)
        axes[2].set_title("landcover", fontsize=10, color=INK, loc="left")
        axes[3].imshow(gt[win], cmap=ListedColormap(["#f5f7fa", INK]), interpolation="nearest")
        axes[3].set_title("эталон (GT)", fontsize=10, color=INK, loc="left")
        axes[4].imshow(res, cmap=ListedColormap(["#f5f7fa", "#9aa5b1", "#d64545", "#2f6fb0"]), vmin=0, vmax=3,
                       interpolation="nearest")
        axes[4].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#9aa5b1", "#d64545", "#2f6fb0")],
                       labels=["верно (TP)", "ложное (FP)", "пропуск (FN)"], fontsize=7, loc="lower right", framealpha=0.9)
        axes[4].set_title("OOF-предсказание", fontsize=10, color=INK, loc="left")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        n_fp, n_fn = int((res == 2).sum()), int((res == 3).sum())
        fig.suptitle(f"{cid} ({'ночь' if night else 'день'}), окно 48×48 px: FP {n_fp}, FN {n_fn} "
                     f"(по чипу ошибок {int(err[ci])}, огня {int(gt.sum())} px)", fontsize=11, color=INK, x=0.01, ha="left")
        fig.tight_layout()
        name = f"af_errors_{rank + 1}.png"
        fig.savefig(FIG / name, dpi=120)
        plt.close(fig)
        out.append(f"{name}: {cid}, ошибок {int(err[ci])}")
    return out


def main() -> None:
    seed_everything(42, with_torch=False)
    FIG.mkdir(parents=True, exist_ok=True)
    importance_figure()
    data = REPO_ROOT / "data"
    meta = read_meta(data / "train", "af").merge(load_folds("af"), on="chip_id")
    ds = load_or_build(data / "train", meta["chip_id"].tolist(), data / "cache" / "af_train.npz")
    oof, _, _ = run_cv(ds, meta["fold"].to_numpy(), None, None, ds["is_cand"])
    thr = json.loads((REPO_ROOT / "weights" / "af_lgbm_meta.json").read_text(encoding="utf-8"))["threshold"]
    for line in error_panels(ds, meta, oof, thr):
        print(line)


if __name__ == "__main__":
    main()
