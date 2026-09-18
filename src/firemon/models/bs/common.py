"""Общее для обучения BS: загрузка train в память, сплиты, оценка, декодирование вероятностей, журнал."""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ...io import list_chips, read_bs_chip
from ...metrics import bs_metrics
from ...splits import load_folds
from ...utils import REPO_ROOT

N_CLASSES = 4
W_BURN, W_SEV = 0.35, 0.30


def load_chips(data_dir: str | Path, chip_ids: list[str] | None = None, with_mask: bool = True,
               workers: int = 8) -> list[dict]:
    """Все BS-чипы в память (rasterio отпускает GIL → потоки). 224 чипа ≈ 3 ГБ, ≈ 5 с."""
    ids = chip_ids or list_chips(data_dir, "bs")
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda c: read_bs_chip(data_dir, c, with_mask=with_mask), ids))


def get_splits(chip_ids: list[str]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """{'group': [(train_idx, val_idx) × 5], 'temporal': [(train_idx, val_idx)]} по configs/folds_bs.csv."""
    folds = load_folds("bs").set_index("chip_id").loc[chip_ids]
    fold = folds["fold"].to_numpy()
    temporal = folds["temporal_split"].to_numpy()
    idx = np.arange(len(chip_ids))
    group = [(idx[fold != k], idx[fold == k]) for k in sorted(np.unique(fold))]
    return {"group": group, "temporal": [(idx[temporal == "train"], idx[temporal == "val"])]}


def bs_part(m: dict) -> float:
    """Вклад BS в Score: 0.35·IoU_burn + 0.30·mIoU_sev."""
    return W_BURN * m["iou_burn"] + W_SEV * m["miou_sev"]


def evaluate_masks(preds: dict[str, np.ndarray], gts: dict[str, np.ndarray]) -> dict:
    m = bs_metrics(preds, gts)
    m.pop("counts", None)
    m["bs_part"] = bs_part(m)
    return m


def fmt(m: dict) -> str:
    return (f"IoU_burn {m['iou_burn']:.4f} | mIoU_sev {m['miou_sev']:.4f} "
            f"({m['iou_1']:.3f}/{m['iou_2']:.3f}/{m['iou_3']:.3f}) | BS-часть {m['bs_part']:.4f}")


# ---------- декодирование вероятностей (4, H, W) → маска 0..3 ----------

def decode_probs(probs: np.ndarray, t_burn: float | None = None,
                 w: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """t_burn=None → argmax по 4 классам. Иначе: гарь = 1−p0 > t_burn, степень = argmax(p[1:]·w)+1.
    probs — (4, ...) — ось классов первая."""
    probs = np.asarray(probs, dtype=np.float32)
    if t_burn is None:
        return probs.argmax(axis=0).astype(np.uint8)
    sev = probs[1:] * np.asarray(w, np.float32).reshape((3,) + (1,) * (probs.ndim - 1))
    cls = sev.argmax(axis=0).astype(np.uint8) + 1
    return np.where(1.0 - probs[0] > t_burn, cls, 0).astype(np.uint8)


def _counts_metrics(pred: np.ndarray, gt: np.ndarray, base_fn: np.ndarray) -> tuple[float, float]:
    """IoU_burn и mIoU_sev по плоским массивам; base_fn[k] — пиксели эталона класса k вне выборки (все FN)."""
    ious = []
    for k in (1, 2, 3):
        p, g = pred == k, gt == k
        tp = np.count_nonzero(p & g)
        den = np.count_nonzero(p | g) + base_fn[k]
        ious.append(1.0 if den == 0 else tp / den)
    p, g = pred > 0, gt > 0
    den = np.count_nonzero(p | g) + base_fn[1:].sum()
    burn = 1.0 if den == 0 else np.count_nonzero(p & g) / den
    return burn, float(np.mean(ious))


def fit_decode(probs: np.ndarray, gts: np.ndarray, t_grid: np.ndarray | None = None,
               w_grid: np.ndarray | None = None, n_rounds: int = 2) -> dict:
    """Подбор t_burn (по IoU_burn) и множителей классов (координатный спуск по mIoU_sev).
    probs (N, 4, H, W), gts (N, H, W). Точно: пиксели с 1−p0 ≤ min(t_grid) и gt=0 на метрику не влияют."""
    t_grid = np.arange(0.20, 0.71, 0.02) if t_grid is None else t_grid
    w_grid = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8]) if w_grid is None else w_grid
    p = np.moveaxis(np.asarray(probs), 1, 0).reshape(4, -1)
    g = np.asarray(gts).reshape(-1)
    keep = ((1.0 - p[0].astype(np.float32)) > t_grid.min()) & (g != 255)
    base_fn = np.array([np.count_nonzero((g == k) & ~keep) for k in range(4)])
    p, g = p[:, keep].astype(np.float32), g[keep]

    best_t, best = None, -1.0
    pb = 1.0 - p[0]
    for t in t_grid:
        pr, gg = pb > t, g > 0
        iou = np.count_nonzero(pr & gg) / max(np.count_nonzero(pr | gg) + base_fn[1:].sum(), 1)
        if iou > best:
            best_t, best = float(t), iou
    w = [1.0, 1.0, 1.0]
    burn = pb > best_t
    best_m = _counts_metrics(np.where(burn, (p[1:] * np.array(w, np.float32)[:, None]).argmax(0) + 1, 0), g, base_fn)[1]
    for _ in range(n_rounds):
        for i in (0, 2):  # w2 фиксирован = 1 (масштаб не важен)
            for cand in w_grid:
                ww = list(w)
                ww[i] = float(cand)
                pred = np.where(burn, (p[1:] * np.array(ww, np.float32)[:, None]).argmax(0) + 1, 0)
                m = _counts_metrics(pred, g, base_fn)[1]
                if m > best_m + 1e-6:
                    best_m, w = m, ww
    return {"t_burn": best_t, "w": [float(x) for x in w], "iou_burn": float(best), "miou_sev": float(best_m)}


# ---------- журнал экспериментов ----------

def log_experiment(description: str, config: str, validation: str, m_group: dict | None, m_temporal: dict | None,
                   inference_time: str = "—", conclusion: str = "", seed: int = 42,
                   path: Path | None = None) -> None:
    """Добавляет строку в reports/experiments.md (обе цифры: групповой OOF / временной сплит)."""
    path = path or REPO_ROOT / "reports" / "experiments.md"
    text = path.read_text(encoding="utf-8")
    nums = [int(l.split("|")[1]) for l in text.splitlines() if l.startswith("|") and l.split("|")[1].strip().isdigit()]
    n = max(nums, default=0) + 1

    def cell(key: str, f: str = "{:.4f}") -> str:
        parts = [f.format(m[key]) if m else "—" for m in (m_group, m_temporal)]
        return " / ".join(parts)

    def sev() -> str:
        out = []
        for m in (m_group, m_temporal):
            out.append("—" if not m else f"{m['miou_sev']:.4f} ({m['iou_1']:.3f}/{m['iou_2']:.3f}/{m['iou_3']:.3f})")
        return " / ".join(out)

    row = (f"| {n} | {dt.date.today().isoformat()} | BS | {description} | {config} | {validation} | {seed} | — | "
           f"{cell('iou_burn')} | {sev()} | BS-часть {cell('bs_part')} | {inference_time} | {conclusion} |")
    lines = text.rstrip("\n").split("\n")
    last = max(i for i, l in enumerate(lines) if l.startswith("|"))
    lines.insert(last + 1, row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
