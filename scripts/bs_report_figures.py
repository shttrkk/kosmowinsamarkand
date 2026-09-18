"""Рисунки и таблицы для reports/bs_module.md по OOF финальной модели BS.

    python scripts/bs_report_figures.py --probs stack_r34_cedice --ensemble ensemble_r34_cedice_main_r34_cedice.json

Выход: reports/figures/bs_*.png и reports/bs_report_stats.json (матрица ошибок, IoU по чипам и т.п.).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.models.bs.common import decode_probs, evaluate_masks, get_splits, load_chips  # noqa: E402
from firemon.models.bs.postprocess import cloud_mask, postprocess_bs  # noqa: E402
from firemon.models.bs.rules import nbr  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
FIG = ROOT / "reports" / "figures"
SEV_CMAP = ListedColormap(["#1b1b1b", "#ffd166", "#f77f00", "#d62828"])
ERR_CMAP = ListedColormap(["#1b1b1b", "#2a9d8f", "#e63946", "#457b9d", "#f4a261"])  # ок-фон, TP, FP, FN, степень


def rgb(s2: np.ndarray) -> np.ndarray:
    x = np.moveaxis(s2[[2, 1, 0]].astype(np.float32), 0, 2)
    hi = max(np.percentile(x, 99), 1.0)
    return np.clip(x / hi, 0, 1)


def chip_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    u = np.count_nonzero((pred > 0) | (gt > 0))
    return 1.0 if u == 0 else np.count_nonzero((pred > 0) & (gt > 0)) / u


def curves(tags: list[str]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for tag in tags:
        for split in ("group", "temporal"):
            p = OOF / f"{tag}_{split}.json"
            if not p.exists():
                continue
            for k, hist in enumerate(json.loads(p.read_text(encoding="utf-8"))["history"]):
                ep = [h["epoch"] for h in hist]
                ax[0].plot(ep, [h["loss"] for h in hist], lw=0.8, label=f"{tag} {split}{k}")
                ev = [h for h in hist if "iou_burn" in h]
                ax[1].plot([h["epoch"] for h in ev], [h["iou_burn"] for h in ev], marker="o", ms=3, lw=0.8)
                ax[2].plot([h["epoch"] for h in ev], [h["miou_sev"] for h in ev], marker="o", ms=3, lw=0.8)
    for a, t in zip(ax, ("train loss", "val IoU_burn (argmax)", "val mIoU_sev (argmax)")):
        a.set_title(t), a.set_xlabel("эпоха"), a.grid(alpha=0.3)
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "bs_unet_curves.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--probs", default="stack_r34_cedice")
    ap.add_argument("--ensemble", default="ensemble_r34_cedice_main_r34_cedice.json")
    ap.add_argument("--curve-tags", nargs="*", default=["r34_cedice"])
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    curves(args.curve_tags)

    ens = json.loads((OOF / args.ensemble).read_text(encoding="utf-8"))
    dec, post = ens["final_decode"], ens["postprocess"]
    chips = load_chips(args.data_dir)
    ids = [c["chip_id"] for c in chips]
    probs = np.load(OOF / f"{args.probs}_group.npy", mmap_mode="r")
    preds = [postprocess_bs(decode_probs(probs[i], dec["t_burn"], dec["w"]), chips[i], post) for i in range(len(ids))]
    stats: dict = {"decode": dec, "postprocess": post}

    # матрица ошибок
    conf = np.zeros((4, 4), np.int64)
    for c, p in zip(chips, preds):
        ok = c["mask"] <= 3
        np.add.at(conf, (c["mask"][ok], p[ok]), 1)
    stats["confusion_gt_rows_pred_cols"] = conf.tolist()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, m, t in ((ax[0], conf / conf.sum(1, keepdims=True), "доля от эталона (строки = GT)"),
                    (ax[1], conf[1:, 1:] / conf[1:, 1:].sum(1, keepdims=True), "только внутри гари GT∩pred")):
        a.imshow(m, cmap="Blues", vmin=0, vmax=1)
        off = 4 - m.shape[0]
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                a.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", color="black" if m[i, j] < 0.6 else "white")
        a.set_xticks(range(m.shape[1]), [str(k + off) for k in range(m.shape[1])])
        a.set_yticks(range(m.shape[0]), [str(k + off) for k in range(m.shape[0])])
        a.set_xlabel("предсказание"), a.set_ylabel("эталон"), a.set_title(t)
    fig.tight_layout()
    fig.savefig(FIG / "bs_confusion.png", dpi=110)
    plt.close(fig)

    # cloud_frac vs IoU по чипам
    cf = np.array([float(c["meta"].get("cloud_frac", 0) or 0) for c in chips])
    iou = np.array([chip_iou(p, c["mask"]) for c, p in zip(chips, preds)])
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 1.0)]
    stats["cloud_bins"] = []
    for lo, hi in bins:
        sel = np.nonzero((cf >= lo) & (cf < hi))[0]
        m = evaluate_masks({i: preds[i] for i in sel}, {i: chips[i]["mask"] for i in sel}) if len(sel) else None
        stats["cloud_bins"].append({"range": [lo, hi], "n": int(len(sel)), "median_chip_iou": float(np.median(iou[sel])) if len(sel) else None,
                                    "micro": m})
    # внутри / вне облачных пикселей (SCL)
    pin, pout = {}, {}
    for i, (c, p) in enumerate(zip(chips, preds)):
        cm = cloud_mask(c)
        gi, go = c["mask"].copy(), c["mask"].copy()
        gi[~cm], go[cm] = 255, 255
        pin[i], pout[i] = (p, gi), (p, go)
    stats["inside_cloud_scl"] = evaluate_masks({i: v[0] for i, v in pin.items()}, {i: v[1] for i, v in pin.items()})
    stats["outside_cloud_scl"] = evaluate_masks({i: v[0] for i, v in pout.items()}, {i: v[1] for i, v in pout.items()})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(cf * 100, iou, s=14, alpha=0.6)
    ax.set_xlabel("cloud_frac, %"), ax.set_ylabel("IoU_burn чипа (OOF)"), ax.grid(alpha=0.3)
    ax.set_title(f"Облачность и качество контура (corr = {np.corrcoef(cf, iou)[0, 1]:+.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "bs_cloud_vs_iou.png", dpi=110)
    plt.close(fig)
    stats["cloud_iou_corr"] = float(np.corrcoef(cf, iou)[0, 1])

    # ошибки по landcover
    lc_stats = {}
    for code in (10, 30, 40, 90):
        fp = fn = tp = 0
        for c, p in zip(chips, preds):
            s = c["aux"][2] == code
            tp += np.count_nonzero(s & (p > 0) & (c["mask"] > 0))
            fp += np.count_nonzero(s & (p > 0) & (c["mask"] == 0))
            fn += np.count_nonzero(s & (p == 0) & (c["mask"] > 0))
        lc_stats[code] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "iou_burn": tp / max(tp + fp + fn, 1)}
    stats["by_landcover"] = lc_stats

    # разбор ошибок: выбираем характерные чипы
    def fp_on(code):
        return [np.count_nonzero((p > 0) & (c["mask"] == 0) & (c["aux"][2] == code)) for c, p in zip(chips, preds)]

    def sev_under(code):
        return [np.count_nonzero((p > 0) & (c["mask"] > p) & (c["aux"][2] == code)) for c, p in zip(chips, preds)]

    fn_all = [np.count_nonzero((p == 0) & (c["mask"] > 0)) for c, p in zip(chips, preds)]
    cases = {
        "ложные контуры на пашне (FP, landcover 40)": int(np.argmax(fp_on(40))),
        "пропуск гари (FN)": int(np.argmax(fn_all)),
        "облачный чип": int(np.argmax(np.where(iou < 0.9, cf, -1))),
        "недооценка степени в лесу (landcover 10)": int(np.argmax(sev_under(10))),
        "ложные контуры на траве (FP, landcover 30)": int(np.argmax(fp_on(30))),
        "типичный удачный чип": int(np.argsort(iou)[int(len(iou) * 0.75)]),
    }
    stats["cases"] = {}
    fig, ax = plt.subplots(len(cases), 6, figsize=(24, 4 * len(cases)))
    for r, (title, i) in enumerate(cases.items()):
        c, p = chips[i], preds[i]
        gt = c["mask"]
        d = nbr(c["s2_pre"]) - nbr(c["s2_post"])
        err = np.zeros(gt.shape, np.uint8)
        err[(p > 0) & (gt > 0) & (p == gt)] = 1
        err[(p > 0) & (gt == 0)] = 2
        err[(p == 0) & (gt > 0)] = 3
        err[(p > 0) & (gt > 0) & (p != gt)] = 4
        panels = [(rgb(c["s2_pre"]), None, f"{c['chip_id']} RGB pre"), (rgb(c["s2_post"]), None, "RGB post"),
                  (d, dict(cmap="RdYlGn_r", vmin=-0.2, vmax=0.8), "dNBR"),
                  (gt, dict(cmap=SEV_CMAP, vmin=0, vmax=3), "эталон"),
                  (p, dict(cmap=SEV_CMAP, vmin=0, vmax=3), f"OOF-предсказание (IoU {iou[i]:.2f})"),
                  (err, dict(cmap=ERR_CMAP, vmin=0, vmax=4), "ошибки: зел. TP, красн. FP, син. FN, оранж. степень")]
        for a, (img, kw, t) in zip(ax[r], panels):
            a.imshow(img, **(kw or {}), interpolation="nearest")
            a.set_title(t, fontsize=10), a.axis("off")
        ax[r, 0].text(0, -40, title, fontsize=13, fontweight="bold")
        stats["cases"][title] = {"chip_id": c["chip_id"], "iou_burn": float(iou[i]), "cloud_frac": float(cf[i])}
    fig.tight_layout()
    fig.savefig(FIG / "bs_error_cases.png", dpi=70)
    plt.close(fig)

    stats["final_group"] = evaluate_masks({i: preds[i] for i in range(len(ids))}, {i: chips[i]["mask"] for i in range(len(ids))})
    (ROOT / "reports" / "bs_report_stats.json").write_text(json.dumps(stats, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("final_group", "cloud_iou_corr", "by_landcover")}, indent=1, default=float))


if __name__ == "__main__":
    main()
