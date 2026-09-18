"""EDA модуля BS (гари и степень поражения, Sentinel-2/1 @20 м). Пересчитывает все таблицы и фигуры части B.

Запуск из корня репозитория:  python eda/eda_bs.py [--no-cache] [--no-gbm]
Выход: eda/tables/bs_*.csv, eda/tables/bs_numbers.json, reports/figures/eda_bs_*.png, кэш eda/cache/bs_*.npz.
Даты и геопривязка используются ТОЛЬКО для анализа и валидации, не как признаки модели.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import plt  # noqa: E402

from firemon.io import read_bs_chip, read_meta  # noqa: E402

SAMPLE_FRAC = 0.04
USGS = (0.10, 0.27, 0.44)
CLOUD_SCL = (3, 8, 9, 10)
LC_MAIN = (10, 20, 30, 40, 50, 60, 80, 90)
HB_LO, HB_HI, HB_STEP = -0.30, 1.00, 0.01          # сетка dNBR для подбора порогов
HB_EDGES = np.round(np.arange(HB_LO, HB_HI + HB_STEP / 2, HB_STEP), 4)
N = {}

INDEX_NAMES = ["NBR_pre", "NBR_post", "dNBR", "RdNBR", "NDVI_pre", "NDVI_post", "dNDVI", "NDWI_post", "dNDWI", "BAI_post", "dBAI",
               "MIRBI_post", "dMIRBI", "NBR2_post", "dNBR2", "dSWIR2", "dSWIR1", "dNIR", "VIS_pre", "VIS_post", "dVIS"]
S1_NAMES = ["VV_pre", "VH_pre", "VV_post", "VH_post", "dVV", "dVH", "dVV_m5", "dVH_m5", "VHVV_post", "dVHVV"]


def _ratio(a, b):
    return (a - b) / np.maximum(a + b, 1e-4)


def indices(pre: np.ndarray, post: np.ndarray) -> dict:
    """Спектральные индексы по S2 (каналы uint16 ×10000 → отражение)."""
    p = pre[:9].astype(np.float32) / 1e4
    q = post[:9].astype(np.float32) / 1e4
    B2, B3, B4, B8A, B11, B12 = 0, 1, 2, 6, 7, 8
    o = {}
    o["NBR_pre"], o["NBR_post"] = _ratio(p[B8A], p[B12]), _ratio(q[B8A], q[B12])
    o["dNBR"] = o["NBR_pre"] - o["NBR_post"]
    o["RdNBR"] = o["dNBR"] / np.sqrt(np.maximum(np.abs(o["NBR_pre"]), 1e-3))
    o["NDVI_pre"], o["NDVI_post"] = _ratio(p[B8A], p[B4]), _ratio(q[B8A], q[B4])
    o["dNDVI"] = o["NDVI_pre"] - o["NDVI_post"]
    o["NDWI_post"] = _ratio(q[B3], q[B8A]); o["dNDWI"] = o["NDWI_post"] - _ratio(p[B3], p[B8A])
    bai = lambda x: 1.0 / ((0.1 - x[B4]) ** 2 + (0.06 - x[B8A]) ** 2 + 1e-6)
    o["BAI_post"] = bai(q); o["dBAI"] = bai(q) - bai(p)
    mirbi = lambda x: 10 * x[B12] - 9.8 * x[B11] + 2
    o["MIRBI_post"] = mirbi(q); o["dMIRBI"] = mirbi(q) - mirbi(p)
    o["NBR2_post"] = _ratio(q[B11], q[B12]); o["dNBR2"] = _ratio(p[B11], p[B12]) - o["NBR2_post"]
    o["dSWIR2"] = q[B12] - p[B12]; o["dSWIR1"] = q[B11] - p[B11]; o["dNIR"] = q[B8A] - p[B8A]
    o["VIS_pre"] = p[[B2, B3, B4]].mean(0); o["VIS_post"] = q[[B2, B3, B4]].mean(0); o["dVIS"] = o["VIS_post"] - o["VIS_pre"]
    return o


def local_std(x: np.ndarray, w: int = 7) -> np.ndarray:
    m = ndi.uniform_filter(x, w); m2 = ndi.uniform_filter(x * x, w)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def process_chip(args):
    i, chip_id = args
    c = read_bs_chip(C.DATA_DIR, chip_id)
    m, valid = c["mask"], c["valid"]
    idx = indices(c["s2_pre"], c["s2_post"])
    lc = c["aux"][2].astype(np.uint8)
    scl_pre, scl_post = c["s2_pre"][9].astype(np.uint8), c["s2_post"][9].astype(np.uint8)
    s1p, s1q = c["s1_pre"].astype(np.float32) / 100, c["s1_post"].astype(np.float32) / 100
    s1 = {"VV_pre": s1p[0], "VH_pre": s1p[1], "VV_post": s1q[0], "VH_post": s1q[1], "dVV": s1q[0] - s1p[0], "dVH": s1q[1] - s1p[1]}
    med5 = [ndi.median_filter(x, 5) for x in (s1p[0], s1p[1], s1q[0], s1q[1])]
    s1["dVV_m5"], s1["dVH_m5"] = med5[2] - med5[0], med5[3] - med5[1]
    s1["VHVV_post"] = s1q[1] - s1q[0]; s1["dVHVV"] = (s1q[1] - s1q[0]) - (s1p[1] - s1p[0])
    # шум SAR: локальное std (7×7) на однородных несгоревших участках до/после медианного фильтра
    homog = (ndi.maximum_filter(lc, 9) == ndi.minimum_filter(lc, 9)) & (ndi.maximum_filter(np.where(m == 255, 0, m), 9) == 0)
    noise = {}
    if homog.sum() > 500:
        for nm, x in (("VV", s1p[0]), ("VH", s1p[1])):
            noise[f"{nm}_raw"] = float(np.median(local_std(x)[homog]))
            noise[f"{nm}_m3"] = float(np.median(local_std(ndi.median_filter(x, 3))[homog]))
            noise[f"{nm}_m5"] = float(np.median(local_std(med5[0 if nm == "VV" else 1])[homog]))
        dd = s1q[1] - s1p[1]
        noise["dVH_raw"] = float(np.median(local_std(dd)[homog])); noise["dVH_m5"] = float(np.median(local_std(s1["dVH_m5"])[homog]))
    # компоненты и дырки гари
    burn = (m > 0) & (m != 255)
    lab, k = ndi.label(burn, structure=np.ones((3, 3), bool))
    comp = np.bincount(lab.ravel())[1:] if k else np.array([], int)
    holes = ndi.binary_fill_holes(burn) & ~burn
    hl, hk = ndi.label(holes)
    hsz = np.bincount(hl.ravel())[1:] if hk else np.array([], int)
    # выборка пикселей
    rng = np.random.RandomState(C.SEED + i)
    sel = rng.choice(m.size, int(m.size * SAMPLE_FRAC), replace=False)
    smp = {k_: v.ravel()[sel].astype(np.float32) for k_, v in {**idx, **s1}.items()}
    smp.update(chip=np.full(sel.size, i, np.int16), mask=m.ravel()[sel], lc=lc.ravel()[sel], scl_pre=scl_pre.ravel()[sel],
               scl_post=scl_post.ravel()[sel], valid=valid.ravel()[sel], dem=c["aux"][0].ravel()[sel], slope=c["aux"][1].ravel()[sel],
               s1_nodata=((c["s1_pre"] == 0).all(0) | (c["s1_post"] == 0).all(0)).ravel()[sel])
    full = dict(dnbr=idx["dNBR"].astype(np.float16), mask=m, lc=lc, scl_pre=scl_pre, scl_post=scl_post)
    return dict(sample=smp, full=full, noise=noise, comp=comp, holes=hsz, slope_const=bool(c["aux"][1].min() == c["aux"][1].max()))


def load_all(use_cache: bool):
    meta = read_meta(C.DATA_DIR, "bs").copy()
    p_s, p_f, p_c = C.CACHE_DIR / "bs_sample.npz", C.CACHE_DIR / "bs_full.npz", C.CACHE_DIR / "bs_perchip.npz"
    if use_cache and p_s.exists() and p_f.exists() and p_c.exists():
        print("кэш: eda/cache/bs_*.npz")
        return meta, dict(np.load(p_s)), dict(np.load(p_f)), dict(np.load(p_c, allow_pickle=True))
    t0 = time.time()
    with ThreadPoolExecutor(C.N_THREADS) as ex:
        res = list(ex.map(process_chip, list(enumerate(meta["chip_id"]))))
    S = {k: np.concatenate([r["sample"][k] for r in res]) for k in res[0]["sample"]}
    F = {k: np.stack([r["full"][k] for r in res]) for k in res[0]["full"]}
    noise = pd.DataFrame([r["noise"] for r in res])
    PC = {"comp_sizes": np.concatenate([r["comp"] for r in res]), "comp_chip": np.concatenate([np.full(len(r["comp"]), i) for i, r in enumerate(res)]),
          "hole_sizes": np.concatenate([r["holes"] for r in res]), "slope_const": np.array([r["slope_const"] for r in res]),
          **{f"noise_{c_}": noise[c_].to_numpy(float) for c_ in noise.columns}}
    np.savez(p_s, **S); np.savez(p_f, **F); np.savez(p_c, **PC)
    print(f"обработано {len(meta)} BS-чипов за {time.time() - t0:.0f} с; выборка {S['mask'].size:,} пикселей")
    return meta, S, F, PC


# ----------------------------------------------------------------------------- метрики по гистограммам
def sev_metrics_from_conf(conf: np.ndarray) -> dict:
    """conf[gt, pred] 4×4 → IoU_burn, IoU(1..3), mIoU_sev, Score_bs-часть."""
    iou = []
    for k in (1, 2, 3):
        tp = conf[k, k]; den = conf[k, :].sum() + conf[:, k].sum() - tp
        iou.append(tp / den if den > 0 else 1.0)
    tpb = conf[1:, 1:].sum(); denb = conf.sum() - conf[0, 0]
    return {"IoU_burn": tpb / denb if denb > 0 else 1.0, "IoU1": iou[0], "IoU2": iou[1], "IoU3": iou[2], "mIoU_sev": float(np.mean(iou))}


def conf_add(conf: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
    conf += np.bincount(y.astype(np.int64) * 4 + pred.astype(np.int64), minlength=16).reshape(4, 4)


def conf_from_hist(h: np.ndarray, thr) -> np.ndarray:
    """h[класс, бин dNBR] + пороги (t1,t2,t3) → матрица ошибок 4×4."""
    e = np.searchsorted(HB_EDGES, np.asarray(thr) - 1e-9) + 1      # +1: бин 0 — всё, что ниже HB_LO
    cs = np.concatenate([np.zeros((4, 1)), np.cumsum(h, axis=1)], axis=1)
    cuts = [0, e[0], e[1], e[2], h.shape[1]]
    return np.stack([cs[:, cuts[j + 1]] - cs[:, cuts[j]] for j in range(4)], axis=1)


def best_thresholds(h: np.ndarray, objective: str = "mIoU_sev"):
    """Полный перебор t1<t2<t3 по сетке (шаг 0.02) — максимум mIoU_sev на гистограмме h[класс, бин]."""
    grid = np.arange(0.0, 0.90, 0.02)
    cs = np.concatenate([np.zeros((4, 1)), np.cumsum(h, axis=1)], axis=1)
    e = np.searchsorted(HB_EDGES, grid - 1e-9) + 1
    tot = h.sum(1)
    best, best_thr = -1.0, None
    G = len(grid)
    i2, i3 = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    ok23 = i3 > i2
    for a in range(G):
        seg1 = cs[:, e[i2]] - cs[:, e[a]][:, None, None]                       # [t1,t2)
        seg2 = cs[:, e[i3]] - cs[:, e[i2]]                                       # [t2,t3)
        seg3 = cs[:, -1][:, None, None] - cs[:, e[i3]]                           # [t3,∞)
        score = np.zeros((G, G))
        for k, seg in ((1, seg1), (2, seg2), (3, seg3)):
            tp = seg[k]; den = tot[k] + seg.sum(0) - tp
            score += np.where(den > 0, tp / np.maximum(den, 1), 1.0)
        score = np.where(ok23 & (i2 > a), score / 3, -1)
        j = np.unravel_index(np.argmax(score), score.shape)
        if score[j] > best:
            best, best_thr = float(score[j]), (float(grid[a]), float(grid[j[0]]), float(grid[j[1]]))
    return best_thr, best


def chip_lc_hist(F) -> np.ndarray:
    """H[чип, lc-индекс (последний — «прочие»), класс, бин dNBR] по пикселям с эталоном != 255."""
    n = len(F["mask"])
    nb = len(HB_EDGES) + 1
    H = np.zeros((n, len(LC_MAIN) + 1, 4, nb), np.int64)
    lut = np.full(256, len(LC_MAIN), np.int64)
    for j, code in enumerate(LC_MAIN):
        lut[code] = j
    for i in range(n):
        m = F["mask"][i].ravel(); ok = m != 255
        b = np.searchsorted(HB_EDGES, F["dnbr"][i].ravel()[ok].astype(np.float32), side="right")
        flat = (lut[F["lc"][i].ravel()[ok]] * 4 + m[ok]) * nb + b
        H[i] = np.bincount(flat, minlength=H[i].size).reshape(H[i].shape)
    return H


# ----------------------------------------------------------------------------- B1 классы
def b1_classes(meta, F, PC):
    print("B1. классы")
    m = F["mask"]
    n = len(m)
    cnt = np.stack([(m == k).reshape(n, -1).sum(1) for k in (0, 1, 2, 3, 255)], 1)
    tot = cnt[:, :4].sum()
    N["b1_class_share_all"] = {k: float(cnt[:, k].sum() / tot) for k in range(4)}
    burn = cnt[:, 1:4].sum(1)
    N["b1_class_share_in_burn"] = {k: float(cnt[:, k].sum() / burn.sum()) for k in (1, 2, 3)}
    N["b1_mask_nodata_share"] = float(cnt[:, 4].sum() / m.size); N["b1_chips_with_mask_nodata"] = int((cnt[:, 4] > 0).sum())
    bf = burn / cnt[:, :4].sum(1)
    N["b1_burn_frac_mean"] = float(bf.mean()); N["b1_burn_frac_median"] = float(np.median(bf)); N["b1_burn_frac_max"] = float(bf.max())
    N["b1_burn_frac_min"] = float(bf.min())
    for k in (1, 2, 3):
        N[f"b1_chips_without_class{k}"] = int((cnt[:, k] == 0).sum())
    dom = np.argmax(cnt[:, 1:4], 1) + 1
    N["b1_chips_by_dominant_class"] = {int(k): int((dom == k).sum()) for k in (1, 2, 3)}
    N["b1_chips_single_class_ge90"] = int((cnt[:, 1:4].max(1) / np.maximum(burn, 1) >= 0.9).sum())
    N["b1_chips_cloud_gt25"] = int((meta["cloud_frac"] > 0.25).sum()); N["b1_cloud_frac_median"] = float(meta["cloud_frac"].median())
    N["b1_chips_cloud_gt10"] = int((meta["cloud_frac"] > 0.10).sum())
    cs, hs = PC["comp_sizes"], PC["hole_sizes"]
    N["b1_components"] = int(cs.size); N["b1_comp_per_chip_median"] = float(np.median(np.bincount(PC["comp_chip"].astype(int), minlength=n)))
    rows = []
    for lo, hi in ((1, 1), (2, 4), (5, 9), (10, 24), (25, 99), (100, 999), (1000, 10 ** 9)):
        s = cs[(cs >= lo) & (cs <= hi)]
        rows.append({"размер, px (1 px = 0.04 га)": f"{lo}–{hi}" if hi < 10 ** 9 else f"≥{lo}", "компонент": s.size, "доля компонент": s.size / cs.size,
                     "пикселей гари": int(s.sum()), "доля площади гари": s.sum() / cs.sum()})
    C.save_table(pd.DataFrame(rows), "bs_b1_components.csv")
    for t in (5, 10, 25):
        N[f"b1_area_share_comp_lt{t}px"] = float(cs[cs < t].sum() / cs.sum()); N[f"b1_comp_share_lt{t}px"] = float((cs < t).mean())
    N["b1_largest_comp_area_share"] = float(sum(cs[PC["comp_chip"] == i].max() for i in range(n) if (PC["comp_chip"] == i).any()) / cs.sum())
    N["b1_holes"] = int(hs.size); N["b1_hole_px_share_of_burn"] = float(hs.sum() / cs.sum()); N["b1_hole_median_px"] = float(np.median(hs)) if hs.size else 0
    N["b1_holes_lt10px_share"] = float((hs < 10).mean()) if hs.size else 0
    C.save_table(pd.DataFrame({"chip_id": meta["chip_id"], "px0": cnt[:, 0], "px1": cnt[:, 1], "px2": cnt[:, 2], "px3": cnt[:, 3], "px255": cnt[:, 4],
                               "burn_frac": bf, "cloud_frac": meta["cloud_frac"]}), "bs_b1_per_chip.csv")

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    sh = [N["b1_class_share_all"][k] * 100 for k in range(4)]
    b = ax[0].bar([C.SEV_NAMES[k] for k in range(4)], sh, color=[C.SEV_COLORS[k] for k in range(4)])
    ax[0].bar_label(b, fmt="%.1f %%"); ax[0].set_ylabel("% пикселей"); ax[0].set_title("Доли классов (все чипы)")
    ax[1].hist(bf * 100, bins=np.arange(0, 60, 2.5), color=C.C_BLUE, edgecolor=C.C_SURFACE)
    ax[1].axvline(np.median(bf) * 100, color=C.C_INK, ls="--", lw=1.5); ax[1].text(np.median(bf) * 100 + 1, ax[1].get_ylim()[1] * 0.9, f"медиана {np.median(bf) * 100:.1f} %")
    ax[1].set_xlabel("доля гари в чипе, %"); ax[1].set_ylabel("число чипов"); ax[1].set_title("Площадь гари на чип")
    ax[2].hist(np.log10(cs), bins=np.arange(0, 5.4, 0.2), color=C.C_ORANGE, edgecolor=C.C_SURFACE, weights=cs / cs.sum() * 100, label="вклад в площадь гари, %")
    ax[2].hist(np.log10(cs), bins=np.arange(0, 5.4, 0.2), histtype="step", color=C.C_BLUE, lw=2, weights=np.full(cs.size, 100 / cs.size), label="доля компонент, %")
    ax[2].set_xlabel("log10 размера связной компоненты гари, px"); ax[2].set_ylabel("%"); ax[2].set_title("Компоненты гари"); ax[2].legend()
    fig.suptitle("BS: баланс классов и геометрия гарей", y=1.02)
    C.savefig(fig, "eda_bs_01_classes.png")


# ----------------------------------------------------------------------------- B2 индексы
def pair_auc(x, y, a, b):
    s = (y == a) | (y == b)
    return C.auc(x[s], (y[s] == b))


def b2_indices(S):
    print("B2. индексы против классов")
    ok = (S["mask"] != 255) & (S["valid"] > 0)
    clear = ok & ~np.isin(S["scl_post"], CLOUD_SCL) & ~np.isin(S["scl_pre"], CLOUD_SCL)
    y = S["mask"]
    S = dict(S)
    for nm in S1_NAMES:                                       # пиксели без S1 (нули в файле) не должны искажать статистику
        S[nm] = np.where(S["s1_nodata"] > 0, np.nan, S[nm])
    rows, qrows = [], []
    for nm in INDEX_NAMES + S1_NAMES + ["slope", "dem"]:
        x = S[nm].astype(np.float64)
        a_burn = C.auc(x[clear], y[clear] > 0)
        rows.append({"признак": nm, "AUC гарь/фон": a_burn, "|AUC-0.5| гарь/фон": abs(a_burn - 0.5), "AUC 1→2": pair_auc(x[clear], y[clear], 1, 2),
                     "AUC 2→3": pair_auc(x[clear], y[clear], 2, 3), "AUC гарь/фон (все пиксели, с облаками)": C.auc(x[ok], y[ok] > 0)})
        for k in range(4):
            qrows.append({"признак": nm, "класс": k, **C.quantile_row(x[clear & (y == k)])})
    t = pd.DataFrame(rows)
    t["сила 1→2"] = (t["AUC 1→2"] - 0.5).abs(); t["сила 2→3"] = (t["AUC 2→3"] - 0.5).abs()
    t["ранг (сумма)"] = (t["|AUC-0.5| гарь/фон"] + t["сила 1→2"] + t["сила 2→3"]).rank(ascending=False).astype(int)
    t = C.save_table(t.sort_values("ранг (сумма)"), "bs_b2_index_auc.csv", floatfmt="%.4f")
    C.save_table(pd.DataFrame(qrows), "bs_b2_index_quantiles.csv")
    N["b2_top_burn"] = t.sort_values("|AUC-0.5| гарь/фон", ascending=False).head(6)[["признак", "AUC гарь/фон"]].values.tolist()
    N["b2_top_12"] = t.sort_values("сила 1→2", ascending=False).head(5)[["признак", "AUC 1→2"]].values.tolist()
    N["b2_top_23"] = t.sort_values("сила 2→3", ascending=False).head(5)[["признак", "AUC 2→3"]].values.tolist()
    N["b2_dNBR_auc"] = [float(t.loc[t["признак"] == "dNBR", c_].iloc[0]) for c_ in ("AUC гарь/фон", "AUC 1→2", "AUC 2→3")]
    N["b2_dNBR_auc_with_clouds"] = float(t.loc[t["признак"] == "dNBR", "AUC гарь/фон (все пиксели, с облаками)"].iloc[0])
    N["b2_dNBR_median_by_class"] = {k: float(np.median(S["dNBR"][clear & (y == k)])) for k in range(4)}

    show = ["dNBR", "RdNBR", "NBR_post", "dNDVI", "dNBR2", "BAI_post", "MIRBI_post", "dSWIR2", "VIS_post", "dVH_m5"]
    fig, ax = plt.subplots(2, 5, figsize=(21, 8.5))
    for a_, nm in zip(ax.ravel(), show):
        data = [S[nm][clear & (y == k)] for k in range(4)]
        data = [d[np.isfinite(d)] for d in data]
        bp = a_.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, medianprops={"color": C.C_INK}, widths=0.65)
        for p_, k in zip(bp["boxes"], range(4)):
            p_.set_facecolor(C.SEV_COLORS[k])
        a_.set_xticklabels(["0", "1", "2", "3"])
        r = t[t["признак"] == nm].iloc[0]
        a_.set_title(f"{nm}\nAUC гарь {r['AUC гарь/фон']:.2f} · 1→2 {r['AUC 1→2']:.2f} · 2→3 {r['AUC 2→3']:.2f}", fontsize=12)
    fig.supxlabel("класс степени поражения (0 — фон)"); fig.suptitle("BS: индексы по классам (ящик 25–75 %, усы 5–95 %, безоблачные пиксели)", y=1.0)
    fig.tight_layout()
    C.savefig(fig, "eda_bs_02_indices.png")
    return t


# ----------------------------------------------------------------------------- B3 пороги по покрову
def b3_thresholds(meta, F, H):
    print("B3. пороги dNBR по типам покрова")
    Hl = H.sum(0)                                   # [lc, класс, бин]
    Hg = Hl.sum(0)
    thr_glob, _ = best_thresholds(Hg)
    centers = np.r_[HB_LO - HB_STEP / 2, HB_EDGES[:-1] + HB_STEP / 2, HB_HI + HB_STEP / 2]

    def hq(h, q):
        cdf = np.cumsum(h) / max(h.sum(), 1)
        return float(centers[min(np.searchsorted(cdf, q), len(centers) - 1)]) if h.sum() > 0 else float("nan")
    rows, thr_lc = [], {}
    names = [C.LC_NAMES[c_] for c_ in LC_MAIN] + ["прочие"]
    conf_usgs = np.zeros((4, 4)); conf_glob = np.zeros((4, 4)); conf_lc = np.zeros((4, 4))
    for j, nm in enumerate(names):
        h = Hl[j]
        burn_px = h[1:].sum()
        enough = burn_px >= 20000 and all(h[k].sum() >= 2000 for k in (1, 2, 3))
        thr, _ = best_thresholds(h) if enough else (thr_glob, None)
        thr_lc[j] = thr
        cu, cg, cl = conf_from_hist(h, USGS), conf_from_hist(h, thr_glob), conf_from_hist(h, thr)
        conf_usgs += cu; conf_glob += cg; conf_lc += cl
        mu, mg, ml = sev_metrics_from_conf(cu), sev_metrics_from_conf(cg), sev_metrics_from_conf(cl)
        rows.append({"landcover": LC_MAIN[j] if j < len(LC_MAIN) else -1, "название": nm, "доля площади": h.sum() / Hg.sum(), "доля гари": burn_px / Hg[1:].sum(),
                     "доля сгоревшего внутри покрова": burn_px / max(h.sum(), 1),
                     "состав гари 1/2/3, %": "/".join(f"{h[k].sum() / max(burn_px, 1) * 100:.0f}" for k in (1, 2, 3)),
                     "медиана dNBR кл.0": hq(h[0], .5), "медиана dNBR кл.1": hq(h[1], .5), "медиана dNBR кл.2": hq(h[2], .5), "медиана dNBR кл.3": hq(h[3], .5),
                     "лучшие пороги t1/t2/t3": "/".join(f"{x:.2f}" for x in thr) + ("" if enough else " (глоб.)"),
                     "mIoU USGS": mu["mIoU_sev"], "mIoU глоб. опт.": mg["mIoU_sev"], "mIoU по покрову": ml["mIoU_sev"],
                     "IoU_burn USGS": mu["IoU_burn"], "IoU_burn глоб. опт.": mg["IoU_burn"], "IoU_burn по покрову": ml["IoU_burn"]})
    C.save_table(pd.DataFrame(rows), "bs_b3_thresholds_by_landcover.csv")
    res = []
    for nm, cf, th in (("USGS 0.10/0.27/0.44 (заглушка)", conf_usgs, USGS), ("глобальные оптимальные", conf_glob, thr_glob), ("оптимальные по landcover", conf_lc, None)):
        res.append({"правило": nm, "пороги": "/".join(f"{x:.2f}" for x in th) if th else "см. таблицу", **sev_metrics_from_conf(cf)})
    t = C.save_table(pd.DataFrame(res), "bs_b3_threshold_rules_pooled.csv", floatfmt="%.4f")
    # «оракул контура»: если границы гари известны точно, сколько даёт разбиение на степени одним dNBR
    conf_o_glob, conf_o_lc = np.zeros((4, 4)), np.zeros((4, 4))
    Ho = Hl.copy(); Ho[:, 0] = 0
    thr_o_glob, _ = best_thresholds(Ho.sum(0))
    thr_o = {}
    for j in range(len(names)):
        h = Ho[j]
        enough = h[1:].sum() >= 20000 and all(h[k].sum() >= 2000 for k in (1, 2, 3))
        thr_o[names[j]] = best_thresholds(h)[0] if enough else thr_o_glob
        cg, cl = conf_from_hist(h, (-9, thr_o_glob[1], thr_o_glob[2])), conf_from_hist(h, (-9, thr_o[names[j]][1], thr_o[names[j]][2]))
        conf_o_glob += cg; conf_o_lc += cl
    N["b3_oracle_perimeter_mIoU_global_thr"] = sev_metrics_from_conf(conf_o_glob)["mIoU_sev"]
    N["b3_oracle_perimeter_mIoU_lc_thr"] = sev_metrics_from_conf(conf_o_lc)["mIoU_sev"]
    N["b3_oracle_perimeter_acc_lc_thr"] = float(np.trace(conf_o_lc[1:, 1:]) / conf_o_lc[1:].sum())
    N["b3_oracle_thr_t2_t3_by_lc"] = {k: list(v[1:]) for k, v in thr_o.items()}
    N["b3_thr_global_opt"] = list(thr_glob)
    N["b3_thr_by_lc"] = {names[j]: list(v) for j, v in thr_lc.items()}
    N["b3_pooled"] = t.to_dict("records")
    N["b3_median_dnbr_by_lc_class"] = {r["название"]: [r["медиана dNBR кл.1"], r["медиана dNBR кл.2"], r["медиана dNBR кл.3"]] for r in rows}

    show = [j for j, c_ in enumerate(LC_MAIN) if Hl[j][1:].sum() >= 20000]
    fig, ax = plt.subplots(1, len(show), figsize=(5.2 * len(show), 4.8), sharey=True, squeeze=False)
    xs = centers[1:-1]
    for a_, j in zip(ax[0], show):
        for k in range(4):
            h = Hl[j][k][1:-1].astype(float)
            a_.plot(xs, ndi.uniform_filter1d(h / max(h.sum(), 1) / HB_STEP, 3), color=C.SEV_COLORS[k], label=C.SEV_NAMES[k])
        for x_ in thr_lc[j]:
            a_.axvline(x_, color=C.C_INK, lw=1.2)
        for x_ in USGS:
            a_.axvline(x_, color=C.C_MUTED, lw=1.2, ls=":")
        a_.set_xlim(-0.2, 0.9); a_.set_xlabel("dNBR"); a_.set_title(f"{names[j]}: пороги " + "/".join(f"{x:.2f}" for x in thr_lc[j]))
    ax[0, 0].set_ylabel("плотность"); ax[0, -1].legend(fontsize=10)
    fig.suptitle("BS: распределение dNBR по классам внутри типов покрова (сплошные линии — лучшие пороги, пунктир — USGS)", y=1.03)
    C.savefig(fig, "eda_bs_03_thresholds_landcover.png")
    return thr_glob, thr_lc


# ----------------------------------------------------------------------------- B4 ложные положительные
def b4_false_positives(meta, S, F):
    print("B4. ложные положительные dNBR")
    ok = (S["mask"] != 255) & (S["valid"] > 0)
    y = S["mask"]
    hi = S["dNBR"] > USGS[0]
    cloud = np.isin(S["scl_post"], CLOUD_SCL) | np.isin(S["scl_pre"], CLOUD_SCL)
    fp = ok & hi & (y == 0); tp = ok & hi & (y > 0)
    N["b4_fp_px_share_of_bg"] = float(fp.sum() / (ok & (y == 0)).sum()); N["b4_precision_dnbr_gt01"] = float(tp.sum() / (tp.sum() + fp.sum()))
    N["b4_recall_dnbr_gt01"] = float(tp.sum() / (ok & (y > 0)).sum()); N["b4_fp_under_cloud_share"] = float(cloud[fp].mean())
    rows = []
    for code in LC_MAIN:
        a = ok & (S["lc"] == code)
        if a.sum() < 1000:
            continue
        rows.append({"landcover": code, "название": C.LC_NAMES[code], "доля площади": a.sum() / ok.sum(), "доля FP": (fp & a).sum() / fp.sum(),
                     "FP как доля фона покрова": (fp & a).sum() / max((a & (y == 0)).sum(), 1), "precision dNBR>0.1": (tp & a).sum() / max(((tp | fp) & a).sum(), 1),
                     "доля FP под облаком/тенью": float(cloud[fp & a].mean()) if (fp & a).any() else 0})
    tl = C.save_table(pd.DataFrame(rows), "bs_b4_fp_by_landcover.csv")
    N["b4_fp_share_by_lc"] = {r["название"]: float(r["доля FP"]) for _, r in tl.iterrows()}
    N["b4_precision_by_lc"] = {r["название"]: float(r["precision dNBR>0.1"]) for _, r in tl.iterrows()}
    fpc, tpc = fp & ~cloud, tp & ~cloud
    rows, arows = [], []
    feats = ["NDVI_pre", "NDVI_post", "dNDVI", "VIS_pre", "VIS_post", "dVIS", "NBR_post", "NBR_pre", "dNBR2", "BAI_post", "MIRBI_post", "dSWIR2", "dNIR", "dVH_m5", "dVV_m5"]
    for scope, sel in (("все покровы", np.ones_like(ok)), ("пашня (40)", S["lc"] == 40), ("трава (30)", S["lc"] == 30), ("лес (10)", S["lc"] == 10)):
        s_ = (fpc | tpc) & sel
        for nm in feats:
            rows.append({"область": scope, "признак": nm, "группа": "FP (dNBR>0.1, класс 0)", **C.quantile_row(S[nm][fpc & sel])})
            rows.append({"область": scope, "признак": nm, "группа": "гарь (dNBR>0.1)", **C.quantile_row(S[nm][tpc & sel])})
            arows.append({"область": scope, "признак": nm, "AUC гарь vs FP": C.auc(S[nm][s_].astype(float), y[s_] > 0)})
    C.save_table(pd.DataFrame(rows), "bs_b4_fp_feature_quantiles.csv")
    ta = pd.DataFrame(arows).pivot(index="признак", columns="область", values="AUC гарь vs FP").reset_index()
    C.save_table(ta, "bs_b4_fp_feature_auc.csv", floatfmt="%.4f")
    N["b4_auc_fp_vs_burn_cropland"] = {r["признак"]: float(r["пашня (40)"]) for _, r in ta.iterrows()}
    N["b4_vis_post_median_fp_vs_burn_cropland"] = [float(np.median(S["VIS_post"][fpc & (S["lc"] == 40)])), float(np.median(S["VIS_post"][tpc & (S["lc"] == 40)]))]
    N["b4_dvis_median_fp_vs_burn_cropland"] = [float(np.median(S["dVIS"][fpc & (S["lc"] == 40)])), float(np.median(S["dVIS"][tpc & (S["lc"] == 40)]))]
    N["b4_ndvi_pre_median_fp_vs_burn_cropland"] = [float(np.median(S["NDVI_pre"][fpc & (S["lc"] == 40)])), float(np.median(S["NDVI_pre"][tpc & (S["lc"] == 40)]))]

    # примеры: чипы с наибольшим числом безоблачных FP на пашне
    dn = F["dnbr"]; m = F["mask"]
    cl = np.isin(F["scl_post"], CLOUD_SCL) | np.isin(F["scl_pre"], CLOUD_SCL)
    fpf = (dn > USGS[0]) & (m == 0) & (F["lc"] == 40) & ~cl
    order = np.argsort(-fpf.reshape(len(m), -1).sum(1))[:4]
    from matplotlib.colors import ListedColormap
    sev_cmap = ListedColormap([C.SEV_COLORS[k] for k in range(4)])
    fig, ax = plt.subplots(len(order), 4, figsize=(17, 4.3 * len(order)))
    for r_, i in enumerate(order):
        c = read_bs_chip(C.DATA_DIR, meta["chip_id"].iloc[i])
        for c_, key, ttl in ((0, "s2_pre", "RGB до"), (1, "s2_post", "RGB после")):
            rgb = np.clip(c[key][[2, 1, 0]].astype(np.float32).transpose(1, 2, 0) / 3000, 0, 1) ** 0.8
            ax[r_, c_].imshow(rgb); ax[r_, c_].set_title(f"{meta['chip_id'].iloc[i]}: {ttl}")
        im = ax[r_, 2].imshow(dn[i].astype(np.float32), cmap="RdGy_r", vmin=-0.6, vmax=0.6); fig.colorbar(im, ax=ax[r_, 2], fraction=0.046)
        ax[r_, 2].set_title(f"dNBR (FP на пашне: {fpf[i].sum() / fpf[i].size * 100:.0f} % чипа)")
        ax[r_, 3].imshow(np.where(m[i] == 255, 0, m[i]), cmap=sev_cmap, vmin=-0.5, vmax=3.5, interpolation="nearest"); ax[r_, 3].set_title("эталон (0–3)")
        for a_ in ax[r_]:
            a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.SEV_COLORS[k]) for k in range(4)]
    fig.legend(handles, [C.SEV_NAMES[k] for k in range(4)], loc="lower center", ncol=4)
    fig.suptitle("BS: высокий dNBR без гари — убранные/вспаханные поля (чипы с максимумом ложных срабатываний на пашне)", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    C.savefig(fig, "eda_bs_04_false_positives.png", tight=False)


# ----------------------------------------------------------------------------- B4b зона события
def enclosing_circle(pts: np.ndarray, iters: int = 300):
    """Приближённая минимальная охватывающая окружность (Badoiu–Clarkson) — без внешних зависимостей."""
    c = pts.mean(0)
    for k in range(1, iters + 1):
        far = pts[np.argmax(((pts - c) ** 2).sum(1))]
        c = c + (far - c) / (k + 1)
    return c, float(np.sqrt(((pts - c) ** 2).sum(1).max()))


def b4b_event_zone(meta, F, thr_glob):
    """Эталон размечен только в зоне «своего» пожара: гарь ограничена дугами окружности (буфер события).
    Оракул: тот же порог dNBR, но только внутри охватывающей окружности гари → верхняя оценка пользы от локализации события."""
    print("B4b. зона события (буфер вокруг пожара)")
    m, dn = F["mask"], F["dnbr"]
    n = len(m)
    yy, xx = np.mgrid[:512, :512]
    excl = (0, 3, 9, 11)
    conf_all, conf_in, conf_in_nc = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))
    rows = []
    for i in range(n):
        b = m[i] > 0
        pts = np.argwhere(b)[:, ::-1].astype(np.float64)
        edge_pts = pts[ndi.binary_erosion(b)[b] == 0] if len(pts) > 2000 else pts
        (cx, cy), r = enclosing_circle(edge_pts)
        dist = np.hypot(xx - cx, yy - cy)
        inside = dist <= r + 0.5
        pred = np.digitize(dn[i].astype(np.float32), thr_glob)
        nc = ~(np.isin(F["scl_post"][i], excl) | np.isin(F["scl_pre"][i], excl))
        conf_add(conf_all, m[i].ravel(), pred.ravel())
        conf_add(conf_in, m[i].ravel(), np.where(inside, pred, 0).ravel())
        conf_add(conf_in_nc, m[i].ravel(), np.where(inside & nc, pred, 0).ravel())
        hi_out = (pred > 0) & ~inside
        rows.append({"chip_id": meta["chip_id"].iloc[i], "radius_km": r * 0.02, "center_offset_km": float(np.hypot(cx - 255.5, cy - 255.5)) * 0.02,
                     "burn_px_on_arc": int((b & (np.abs(dist - r) <= 2)).sum()), "circle_share_of_chip": float(inside.mean()),
                     "dNBR_pos_outside_share_of_chip": float(hi_out.mean())})
    df = C.save_table(pd.DataFrame(rows), "bs_b4b_event_zone.csv")
    res = []
    for nm, cf in (("пороги dNBR по всему чипу", conf_all), ("то же, только внутри окружности события (оракул)", conf_in),
                   ("внутри окружности и вне SCL 0/3/9/11 (оракул)", conf_in_nc)):
        res.append({"вариант": nm, **sev_metrics_from_conf(cf)})
    t = C.save_table(pd.DataFrame(res), "bs_b4b_event_zone_oracle.csv", floatfmt="%.4f")
    N["b4b_oracle"] = t.to_dict("records")
    N["b4b_radius_km_q05_q50_q95"] = [float(df.radius_km.quantile(q)) for q in (.05, .5, .95)]
    N["b4b_center_offset_km_q10_q50_q90"] = [float(df.center_offset_km.quantile(q)) for q in (.1, .5, .9)]
    N["b4b_chips_arc_px_ge30"] = int((df.burn_px_on_arc >= 30).sum())
    N["b4b_circle_share_of_chip_median"] = float(df.circle_share_of_chip.median())
    N["b4b_fp_share_outside_circle"] = float(1 - (conf_in[0, 1:].sum() / max(conf_all[0, 1:].sum(), 1)))


# ----------------------------------------------------------------------------- B4c окружность события и центр чипа
def b4c_event_circle(meta, F, thr_lc):
    """Геометрия буфера события относительно ЧИПА: можно ли использовать расстояние до центра чипа как приор.
    Центр/радиус — охватывающая окружность эталонной гари (оракул); «лишняя гарь» — dNBR ≥ порога слабой степени
    своего покрова дальше R от центра окружности."""
    print("B4c. окружность события относительно центра чипа")
    m, dn, lc = F["mask"], F["dnbr"], F["lc"]
    n = len(m)
    yy, xx = np.mgrid[:512, :512]
    c0 = 255.5
    d_chip = np.hypot(xx - c0, yy - c0)                      # расстояние до центра чипа, px
    t1_lut = np.full(256, thr_lc[len(LC_MAIN)][0], np.float32)
    for j, code in enumerate(LC_MAIN):
        t1_lut[code] = thr_lc[j][0]
    excl = (0, 3, 9, 11)
    rho_grid = np.arange(60, 365, 10)                        # радиус приора от центра чипа, px (до угла 362)
    tp_r, fp_r = np.zeros(len(rho_grid)), np.zeros(len(rho_grid))
    tp_rc, fp_rc = np.zeros(len(rho_grid)), np.zeros(len(rho_grid))   # то же + маска SCL
    ring_edges = np.arange(0, 370, 10)
    ring_burn, ring_all, ring_hi = np.zeros(len(ring_edges) - 1), np.zeros(len(ring_edges) - 1), np.zeros(len(ring_edges) - 1)
    rows, geo = [], []
    burn_total = 0
    for i in range(n):
        b = m[i] > 0
        burn_total += int(b.sum())
        pts = np.argwhere(b)[:, ::-1].astype(np.float64)
        edge_pts = pts[ndi.binary_erosion(b)[b] == 0] if len(pts) > 2000 else pts
        (cx, cy), r = enclosing_circle(edge_pts)
        gx, gy = pts[:, 0].mean(), pts[:, 1].mean()
        hi = dn[i].astype(np.float32) >= t1_lut[lc[i]]
        nc = ~(np.isin(F["scl_post"][i], excl) | np.isin(F["scl_pre"][i], excl))
        d_ev = np.hypot(xx - cx, yy - cy)
        extra = hi & (d_ev > r + 0.5)
        extra_nc = extra & nc
        touch = {"top": bool(b[0].any()), "bottom": bool(b[-1].any()), "left": bool(b[:, 0].any()), "right": bool(b[:, -1].any())}
        far = float(d_chip[b].max())                          # самый дальний от центра чипа пиксель гари
        rows.append({"chip_id": meta["chip_id"].iloc[i], "burn_px": int(b.sum()),
                     "circle_dx_px": cx - c0, "circle_dy_px": cy - c0, "circle_offset_px": float(np.hypot(cx - c0, cy - c0)),
                     "centroid_dx_px": gx - c0, "centroid_dy_px": gy - c0, "centroid_offset_px": float(np.hypot(gx - c0, gy - c0)),
                     "radius_px": r, "radius_km": r * 0.02, "burn_max_dist_from_chip_center_px": far,
                     "chip_center_inside_circle": bool(np.hypot(cx - c0, cy - c0) <= r), "chip_center_is_burn": bool(b[255:257, 255:257].any()),
                     "circle_fits_in_chip": bool(cx - r >= 0 and cy - r >= 0 and cx + r <= 511 and cy + r <= 511),
                     "touches_edge": any(touch.values()), "edges_touched": int(sum(touch.values())), "edge_burn_px": int(b[0].sum() + b[-1].sum() + b[1:-1, 0].sum() + b[1:-1, -1].sum()),
                     "hi_dnbr_px": int(hi.sum()), "extra_px": int(extra.sum()), "extra_px_noSCL": int(extra_nc.sum()),
                     "extra_vs_burn": float(extra.sum() / max(b.sum(), 1))})
        geo.append((cx, cy, r, gx, gy))
        dflat, bflat, hflat, ncflat = d_chip.ravel(), b.ravel(), hi.ravel(), nc.ravel()
        k = np.searchsorted(rho_grid, dflat, side="left")      # пиксель входит в приор с индексом ≥ k
        for acc_tp, acc_fp, sel in ((tp_r, fp_r, hflat), (tp_rc, fp_rc, hflat & ncflat)):
            acc_tp += np.cumsum(np.bincount(k[sel & bflat], minlength=len(rho_grid) + 1))[:-1]
            acc_fp += np.cumsum(np.bincount(k[sel & ~bflat], minlength=len(rho_grid) + 1))[:-1]
        ring = np.clip(np.digitize(dflat, ring_edges) - 1, 0, len(ring_edges) - 2)
        ring_all += np.bincount(ring, minlength=len(ring_all)); ring_burn += np.bincount(ring[bflat], minlength=len(ring_all))
        ring_hi += np.bincount(ring[hflat & ~bflat], minlength=len(ring_all))
    df = C.save_table(pd.DataFrame(rows), "bs_b4c_event_circle_per_chip.csv")

    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    srows = []
    for col, nm in (("circle_dx_px", "центр окружности: смещение по x, px (+ вправо)"), ("circle_dy_px", "центр окружности: смещение по y, px (+ вниз)"),
                    ("circle_offset_px", "центр окружности: модуль смещения, px"), ("centroid_dx_px", "центроид гари: смещение по x, px"),
                    ("centroid_dy_px", "центроид гари: смещение по y, px"), ("centroid_offset_px", "центроид гари: модуль смещения, px"),
                    ("radius_px", "радиус окружности, px"), ("radius_km", "радиус окружности, км"),
                    ("burn_max_dist_from_chip_center_px", "самый дальний пиксель гари от центра чипа, px")):
        x = df[col]
        srows.append({"величина": nm, "min": x.min(), **{f"q{int(q * 100):02d}": x.quantile(q) for q in qs}, "max": x.max(), "среднее": x.mean(),
                      "max |·|": x.abs().max()})
    ts = C.save_table(pd.DataFrame(srows), "bs_b4c_event_circle_summary.csv")

    iou_r = tp_r / (burn_total + fp_r); iou_rc = tp_rc / (burn_total + fp_rc)
    tr = pd.DataFrame({"радиус приора от центра чипа, px": rho_grid, "км": rho_grid * 0.02, "доля площади чипа": [float((d_chip <= q).mean()) for q in rho_grid],
                       "доля эталонной гари внутри": [float(ring_burn[: np.searchsorted(ring_edges, q)].sum() / burn_total) for q in rho_grid],
                       "IoU_burn: порог dNBR по покрову внутри приора": iou_r, "то же + маска SCL 0/3/9/11": iou_rc})
    C.save_table(tr, "bs_b4c_center_prior_radius.csv")
    prof = pd.DataFrame({"расстояние до центра чипа, px (левая граница кольца)": ring_edges[:-1], "доля гари в эталоне": ring_burn / np.maximum(ring_all, 1),
                         "доля ложного высокого dNBR": ring_hi / np.maximum(ring_all, 1)})
    C.save_table(prof, "bs_b4c_burn_profile_by_distance.csv")

    kb = int(np.argmax(iou_rc))
    N["b4c_circle_offset_px_q50_q95_max"] = [float(df.circle_offset_px.quantile(.5)), float(df.circle_offset_px.quantile(.95)), float(df.circle_offset_px.max())]
    N["b4c_circle_dx_px_q05_q50_q95"] = [float(df.circle_dx_px.quantile(q)) for q in (.05, .5, .95)]
    N["b4c_circle_dy_px_q05_q50_q95"] = [float(df.circle_dy_px.quantile(q)) for q in (.05, .5, .95)]
    N["b4c_centroid_offset_px_q50_q95_max"] = [float(df.centroid_offset_px.quantile(.5)), float(df.centroid_offset_px.quantile(.95)), float(df.centroid_offset_px.max())]
    N["b4c_radius_px_q05_q50_q95_max"] = [float(df.radius_px.quantile(q)) for q in (.05, .5, .95)] + [float(df.radius_px.max())]
    N["b4c_chips_touch_edge"] = int(df.touches_edge.sum()); N["b4c_chips_touch_edge_share"] = float(df.touches_edge.mean())
    N["b4c_chips_touch_2plus_edges"] = int((df.edges_touched >= 2).sum())
    N["b4c_chip_center_inside_circle_share"] = float(df.chip_center_inside_circle.mean()); N["b4c_chip_center_is_burn_share"] = float(df.chip_center_is_burn.mean())
    N["b4c_circle_fits_in_chip_share"] = float(df.circle_fits_in_chip.mean())
    N["b4c_chips_with_extra"] = int((df.extra_px > 0).sum()); N["b4c_chips_with_extra_ge_burn"] = int((df.extra_px >= df.burn_px).sum())
    N["b4c_extra_px_total"] = int(df.extra_px.sum()); N["b4c_extra_px_total_noSCL"] = int(df.extra_px_noSCL.sum()); N["b4c_burn_px_total"] = burn_total
    N["b4c_extra_to_burn_ratio"] = float(df.extra_px.sum() / burn_total); N["b4c_extra_to_burn_ratio_noSCL"] = float(df.extra_px_noSCL.sum() / burn_total)
    N["b4c_extra_vs_burn_per_chip_q50_q95"] = [float(df.extra_vs_burn.quantile(.5)), float(df.extra_vs_burn.quantile(.95))]
    N["b4c_best_center_prior_px"] = int(rho_grid[kb]); N["b4c_best_center_prior_iou"] = float(iou_rc[kb]); N["b4c_no_prior_iou"] = float(iou_rc[-1])
    N["b4c_iou_at_prior_px"] = {int(q): float(v) for q, v in zip(rho_grid, iou_rc) if q in (150, 200, 250, 300, 360)}
    N["b4c_burn_share_within_px"] = {int(q): float(v) for q, v in zip(rho_grid, tr["доля эталонной гари внутри"]) if q in (150, 200, 250, 300)}
    N["b4c_burn_prob_center_vs_corner"] = [float(prof.iloc[:5, 1].mean()), float(prof.iloc[30:, 1].mean())]
    far_all = df.burn_max_dist_from_chip_center_px
    N["b4c_burn_max_dist_px_q50_q95_max"] = [float(far_all.quantile(.5)), float(far_all.quantile(.95)), float(far_all.max())]

    # --- картинка 1: статистика
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    ax[0].scatter(df.circle_dx_px, df.circle_dy_px, s=22, color=C.C_BLUE, alpha=0.75, label="центр окружности")
    ax[0].scatter(df.centroid_dx_px, df.centroid_dy_px, s=14, color=C.C_ORANGE, alpha=0.6, marker="^", label="центроид гари")
    ax[0].axhline(0, color=C.C_MUTED, lw=1); ax[0].axvline(0, color=C.C_MUTED, lw=1)
    ax[0].set_xlim(-256, 256); ax[0].set_ylim(256, -256); ax[0].set_aspect("equal")
    ax[0].set_xlabel("смещение по x от центра чипа, px"); ax[0].set_ylabel("смещение по y, px"); ax[0].set_title("Где событие относительно центра чипа"); ax[0].legend(loc="upper right", fontsize=10, frameon=True)
    ax[1].hist(df.radius_px, bins=np.arange(0, 370, 20), color=C.C_BLUE, edgecolor=C.C_SURFACE)
    ax[1].axvline(df.radius_px.median(), color=C.C_INK, ls="--", lw=1.5)
    ax[1].set_xlabel("радиус окружности, px (1 px = 20 м)"); ax[1].set_ylabel("число чипов"); ax[1].set_title(f"Радиус: медиана {df.radius_px.median():.0f} px = {df.radius_km.median():.1f} км")
    ax[2].plot(ring_edges[:-1] + 5, prof.iloc[:, 1] * 100, color=C.C_RED, label="гарь в эталоне")
    ax[2].plot(ring_edges[:-1] + 5, prof.iloc[:, 2] * 100, color=C.C_BLUE, label="высокий dNBR, но фон")
    ax[2].axvline(256, color=C.C_MUTED, lw=1, ls=":"); ax[2].text(258, ax[2].get_ylim()[1] * 0.9, "край чипа", fontsize=10, color=C.C_MUTED)
    ax[2].set_xlabel("расстояние до центра чипа, px"); ax[2].set_ylabel("% пикселей кольца"); ax[2].set_title("Профиль по расстоянию до центра чипа"); ax[2].legend(fontsize=10)
    ax[3].plot(rho_grid, iou_r, color=C.C_BLUE, label="порог dNBR по покрову"); ax[3].plot(rho_grid, iou_rc, color=C.C_ORANGE, label="+ маска SCL 0/3/9/11")
    ax[3].scatter([rho_grid[kb]], [iou_rc[kb]], s=80, color=C.C_INK, zorder=5)
    ax[3].annotate(f"{rho_grid[kb]} px: {iou_rc[kb]:.3f}", (rho_grid[kb], iou_rc[kb]), xytext=(-40, -28), textcoords="offset points")
    ax[3].set_xlabel("радиус приора от центра чипа, px"); ax[3].set_ylabel("IoU_burn (пул пикселей)"); ax[3].set_title("Жёсткий приор «круг от центра чипа»"); ax[3].legend(loc="lower right", fontsize=10)
    fig.suptitle("BS: буфер события относительно чипа — событие не центрировано, приор по центру чипа почти ничего не даёт", y=1.03)
    C.savefig(fig, "eda_bs_04c_event_circle_stats.png")

    # --- картинка 2: примеры (явная дуга; максимум «лишней» гари; касание края; сильное смещение)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    sev_cmap = ListedColormap([C.SEV_COLORS[k] for k in range(4)])
    big = df.burn_px > 5000
    picks, used = [], set()
    for ttl, order in (("максимум «лишней» гари", (df.extra_px * big).sort_values(ascending=False).index),
                       ("гарь касается края", (df.edge_burn_px * big).sort_values(ascending=False).index),
                       ("максимальное смещение центра", (df.circle_offset_px * big).sort_values(ascending=False).index),
                       ("событие почти по центру", (df.circle_offset_px + 1e6 * ~big).sort_values().index)):
        for i in order:
            if i not in used:
                picks.append((ttl, int(i))); used.add(i)
                break
    fig, ax = plt.subplots(2, len(picks), figsize=(5.2 * len(picks), 10.6))
    for c_, (ttl, i) in enumerate(picks):
        cx, cy, r, gx, gy = geo[i]
        hi = dn[i].astype(np.float32) >= t1_lut[lc[i]]
        extra = hi & (np.hypot(xx - cx, yy - cy) > r + 0.5)
        ax[0, c_].imshow(m[i], cmap=sev_cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax[0, c_].set_title(f"{meta['chip_id'].iloc[i]}\n{ttl}", fontsize=13)
        show = np.zeros((512, 512, 4), np.float32)
        show[hi & ~extra] = (0.64, 0.13, 0.12, 1.0); show[extra] = (0.16, 0.47, 0.84, 1.0)
        ax[1, c_].imshow(np.full((512, 512, 3), 0.93)); ax[1, c_].imshow(show, interpolation="nearest")
        ax[1, c_].set_title(f"dNBR ≥ t1(покров): внутри R {int((hi & ~extra).sum()):,} px,\nснаружи {int(extra.sum()):,} px (эталон {int((m[i] > 0).sum()):,})".replace(",", " "), fontsize=12)
        for a_ in ax[:, c_]:
            a_.add_patch(Circle((cx, cy), r, fill=False, ec=C.C_INK, lw=1.8, ls="--"))
            a_.plot([c0], [c0], marker="+", ms=16, mew=2.5, color="#1baf7a"); a_.plot([cx], [cy], marker="x", ms=11, mew=2.5, color=C.C_INK)
            a_.set_xlim(0, 511); a_.set_ylim(511, 0); a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.SEV_COLORS[k]) for k in (1, 2, 3)] + [plt.Rectangle((0, 0), 1, 1, color=(0.64, 0.13, 0.12)), plt.Rectangle((0, 0), 1, 1, color=(0.16, 0.47, 0.84)),
               plt.Line2D([], [], color=C.C_INK, ls="--"), plt.Line2D([], [], color="#1baf7a", marker="+", ls="", ms=12, mew=2.5), plt.Line2D([], [], color=C.C_INK, marker="x", ls="", ms=9, mew=2.5)]
    fig.legend(handles, ["1 слабая", "2 средняя", "3 сильная", "высокий dNBR внутри окружности", "«лишний» высокий dNBR снаружи", "окружность события (оракул)", "центр чипа", "центр окружности"],
               loc="lower center", ncol=4, fontsize=11)
    fig.suptitle("BS: эталон (верх) и пиксели с dNBR выше порога слабой степени своего покрова (низ)", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    C.savefig(fig, "eda_bs_04c_event_circle_examples.png", tight=False)
    return df, ts, tr


# ----------------------------------------------------------------------------- B5 облака
def b5_clouds(meta, F, thr_glob):
    print("B5. облака, тени, SCL")
    m, dn = F["mask"], F["dnbr"]
    ok = m != 255
    names = {0: "nodata", 1: "saturated", 2: "dark", 3: "тень облака", 4: "растительность", 5: "голая почва", 6: "вода", 7: "unclassified",
             8: "облако средн.", 9: "облако выс.", 10: "cirrus", 11: "снег"}
    rows = []
    for code, nm in names.items():
        a, b = F["scl_pre"] == code, F["scl_post"] == code
        bb = b & ok
        rows.append({"SCL": code, "название": nm, "доля pre": a.mean(), "доля post": b.mean(), "доля гари среди пикселей (pre)": (m[a & ok] > 0).mean() if (a & ok).any() else np.nan,
                     "доля гари среди пикселей (post)": (m[bb] > 0).mean() if bb.any() else np.nan,
                     "доля всей гари (post)": (bb & (m > 0)).sum() / (ok & (m > 0)).sum(),
                     "состав гари 1/2/3, %": "/".join(f"{(m[bb] == k).sum() / max((m[bb] > 0).sum(), 1) * 100:.0f}" for k in (1, 2, 3))})
    C.save_table(pd.DataFrame(rows), "bs_b5_scl.csv")
    cl_pre, cl_post = np.isin(F["scl_pre"], CLOUD_SCL), np.isin(F["scl_post"], CLOUD_SCL)
    N["b5_cloud_share_pre"] = float(cl_pre.mean()); N["b5_cloud_share_post"] = float(cl_post.mean()); N["b5_cloud_share_any"] = float((cl_pre | cl_post).mean())
    N["b5_s2_nodata_share"] = float(((F["scl_pre"] == 0) | (F["scl_post"] == 0)).mean())
    N["b5_mask255_share"] = float((~ok).mean()); N["b5_mask255_on_s2_nodata"] = float((((F["scl_pre"] == 0) | (F["scl_post"] == 0)) & ~ok).sum() / max((~ok).sum(), 1))
    N["b5_mask_labelled_on_s2_nodata_px"] = int((((F["scl_pre"] == 0) | (F["scl_post"] == 0)) & ok).sum())
    N["b5_burn_on_s2_nodata_px"] = int((((F["scl_pre"] == 0) | (F["scl_post"] == 0)) & ok & (m > 0)).sum())
    burn = ok & (m > 0)
    N["b5_burn_share_under_cloud_post"] = float((burn & cl_post).sum() / burn.sum()); N["b5_burn_share_under_cloud_any"] = float((burn & (cl_pre | cl_post)).sum() / burn.sum())
    N["b5_burn_frac_cloud_vs_clear"] = [float((m[ok & cl_post] > 0).mean()), float((m[ok & ~cl_post] > 0).mean())]
    N["b5_burn_under_shadow_px"] = int((burn & (F["scl_post"] == 3)).sum()); N["b5_burn_under_dark_px"] = int((burn & (F["scl_post"] == 2)).sum())
    rows = []
    groups = {"ясно (pre и post)": ~cl_pre & ~cl_post, "облако 8/9 в post": np.isin(F["scl_post"], (8, 9)), "cirrus 10 в post": F["scl_post"] == 10,
              "тень 3 в post": F["scl_post"] == 3, "облако/тень только в pre": cl_pre & ~cl_post}
    for g, sel in groups.items():
        sel = sel & ok
        cf = np.zeros((4, 4))
        d = dn[sel].astype(np.float32); y = m[sel]
        pred = np.digitize(d, thr_glob)
        conf_add(cf, y, pred)
        mt = sev_metrics_from_conf(cf)
        rows.append({"группа пикселей": g, "доля пикселей": sel.sum() / ok.sum(), "доля гари": (y > 0).mean(),
                     "медиана dNBR фон": float(np.median(d[y == 0])) if (y == 0).any() else np.nan, "медиана dNBR гарь": float(np.median(d[y > 0])) if (y > 0).any() else np.nan,
                     "AUC dNBR гарь/фон": C.auc(d[::7].astype(float), (y[::7] > 0)), "IoU_burn (глоб. пороги)": mt["IoU_burn"], "mIoU_sev (глоб. пороги)": mt["mIoU_sev"]})
    t = C.save_table(pd.DataFrame(rows), "bs_b5_cloud_groups.csv", floatfmt="%.4f")
    N["b5_groups"] = t.to_dict("records")
    # чипы: облачные против ясных
    cf_chip = meta["cloud_frac"].to_numpy()
    res = []
    for g, sel in (("cloud_frac ≤ 5 %", cf_chip <= 0.05), ("5–25 %", (cf_chip > 0.05) & (cf_chip <= 0.25)), ("> 25 %", cf_chip > 0.25)):
        cf = np.zeros((4, 4))
        for i in np.flatnonzero(sel):
            o = ok[i]
            conf_add(cf, m[i][o], np.digitize(dn[i][o].astype(np.float32), thr_glob))
        res.append({"чипы": g, "число": int(sel.sum()), **sev_metrics_from_conf(cf)})
    tc = C.save_table(pd.DataFrame(res), "bs_b5_cloudy_chips_rule.csv", floatfmt="%.4f")
    N["b5_rule_by_chip_cloudiness"] = tc.to_dict("records")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(t))
    ax[0].bar(x - 0.2, t["IoU_burn (глоб. пороги)"], 0.4, color=C.C_BLUE, label="IoU_burn"); ax[0].bar(x + 0.2, t["mIoU_sev (глоб. пороги)"], 0.4, color=C.C_ORANGE, label="mIoU_sev")
    ax[0].set_xticks(x); ax[0].set_xticklabels([g.replace(" в ", "\nв ").replace(" (", "\n(").replace(" только", "\nтолько") for g in t["группа пикселей"]], fontsize=10)
    ax[0].set_title("Правило dNBR по группам SCL"); ax[0].legend(); ax[0].set_ylim(0, 1)
    ax[1].bar(x, t["доля гари"] * 100, color=[C.C_AQUA] + [C.C_MUTED] * (len(t) - 1))
    ax[1].set_xticks(x); ax[1].set_xticklabels([g.replace(" в ", "\nв ").replace(" (", "\n(").replace(" только", "\nтолько") for g in t["группа пикселей"]], fontsize=10)
    ax[1].set_ylabel("% пикселей группы размечено гарью"); ax[1].set_title("Доля гари в эталоне по группам SCL")
    fig.suptitle("BS: под тенью и плотным облаком в post эталон ≈ 0; под cirrus и облаком в pre гарь размечена, а dNBR ломается", y=1.02)
    C.savefig(fig, "eda_bs_05_clouds.png")


# ----------------------------------------------------------------------------- B6 Sentinel-1
def b6_sar(S, PC):
    print("B6. Sentinel-1")
    ok = (S["mask"] != 255) & (S["s1_nodata"] == 0)
    y = S["mask"]
    rows = []
    for nm in S1_NAMES:
        for k in range(4):
            rows.append({"признак": nm, "класс": k, **C.quantile_row(S[nm][ok & (y == k)])})
    C.save_table(pd.DataFrame(rows), "bs_b6_sar_quantiles.csv")
    rows = []
    scopes = {"все": np.ones_like(ok), "лес (10)": S["lc"] == 10, "трава (30)": S["lc"] == 30, "пашня (40)": S["lc"] == 40, "болото (90)": S["lc"] == 90}
    for nm in ["dVV", "dVH", "dVV_m5", "dVH_m5", "dVHVV", "VH_post", "VV_post", "dNBR"]:
        r = {"признак": nm}
        for sc, sel in scopes.items():
            s_ = ok & sel
            r[f"AUC гарь/фон: {sc}"] = C.auc(S[nm][s_].astype(float), y[s_] > 0)
        s_ = ok & (S["lc"] == 10)
        r["AUC 2→3: лес"] = pair_auc(S[nm][s_].astype(float), y[s_], 2, 3); r["AUC 1→2: лес"] = pair_auc(S[nm][s_].astype(float), y[s_], 1, 2)
        rows.append(r)
    t = C.save_table(pd.DataFrame(rows), "bs_b6_sar_auc.csv", floatfmt="%.4f")
    N["b6_sar_auc"] = t.set_index("признак").round(4).to_dict("index")
    noise = {k[6:]: float(np.nanmedian(v)) for k, v in PC.items() if k.startswith("noise_")}
    N["b6_noise_local_std_db"] = noise
    C.save_table(pd.DataFrame([noise]), "bs_b6_sar_noise.csv")
    N["b6_s1_nodata_share"] = float((S["s1_nodata"] > 0).mean())
    nd_chip = pd.Series(S["s1_nodata"] > 0).groupby(S["chip"]).mean()
    N["b6_chips_s1_nodata_gt50"] = int((nd_chip > 0.5).sum()); N["b6_chips_s1_nodata_any"] = int((nd_chip > 0.001).sum())
    N["b6_dVH_median_by_class_forest"] = {k: float(np.median(S["dVH_m5"][ok & (S["lc"] == 10) & (y == k)])) for k in range(4)}
    N["b6_dVV_median_by_class_grass"] = {k: float(np.median(S["dVV_m5"][ok & (S["lc"] == 30) & (y == k)])) for k in range(4)}
    N["b6_dVV_median_by_class_crop"] = {k: float(np.median(S["dVV_m5"][ok & (S["lc"] == 40) & (y == k)])) for k in range(4)}

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a_, (sc, code, feat) in zip(ax[:2], (("лес", 10, "dVH_m5"), ("трава + пашня", (30, 40), "dVV_m5"))):
        sel = ok & np.isin(S["lc"], code)
        data = [S[feat][sel & (y == k)] for k in range(4)]
        bp = a_.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, medianprops={"color": C.C_INK}, widths=0.65)
        for p_, k in zip(bp["boxes"], range(4)):
            p_.set_facecolor(C.SEV_COLORS[k])
        a_.set_xticklabels([C.SEV_NAMES[k] for k in range(4)]); a_.axhline(0, color=C.C_MUTED, lw=1)
        a_.set_ylabel(f"{feat[:3]} после медианы 5×5, дБ"); a_.set_title(f"{sc}: {feat[:3]} по классам")
    lab = ["VV", "VH"]
    xx = np.arange(2)
    for k, (suffix, nm) in enumerate((("raw", "без фильтра"), ("m3", "медиана 3×3"), ("m5", "медиана 5×5"))):
        b = ax[2].bar(xx + (k - 1) * 0.27, [noise.get(f"{l}_{suffix}", np.nan) for l in lab], 0.25, color=C.PALETTE[k], label=nm)
        ax[2].bar_label(b, fmt="%.2f", fontsize=11)
    ax[2].set_xticks(xx); ax[2].set_xticklabels(lab); ax[2].set_ylabel("локальное std 7×7, дБ"); ax[2].set_title("Спекл-шум и медианный фильтр"); ax[2].legend()
    fig.suptitle("BS: Sentinel-1 — слабый, но независимый от облаков сигнал", y=1.02)
    C.savefig(fig, "eda_bs_06_sar.png")


# ----------------------------------------------------------------------------- B7 рельеф
def b7_relief(S, PC):
    print("B7. рельеф")
    ok = S["mask"] != 255
    y = S["mask"]
    rows = []
    for nm in ("slope", "dem"):
        for k in range(4):
            rows.append({"признак": nm, "класс": k, **C.quantile_row(S[nm][ok & (y == k)])})
    C.save_table(pd.DataFrame(rows), "bs_b7_relief_quantiles.csv")
    N["b7_auc_slope"] = [C.auc(S["slope"][ok].astype(float), y[ok] > 0), pair_auc(S["slope"].astype(float), y, 1, 2), pair_auc(S["slope"].astype(float), y, 2, 3)]
    N["b7_auc_dem"] = [C.auc(S["dem"][ok].astype(float), y[ok] > 0), pair_auc(S["dem"].astype(float), y, 1, 2), pair_auc(S["dem"].astype(float), y, 2, 3)]
    N["b7_slope_q50_q95"] = [float(np.percentile(S["slope"][ok], 50)), float(np.percentile(S["slope"][ok], 95))]
    N["b7_chips_constant_slope"] = int(PC["slope_const"].sum())
    N["b7_slope_unique_values"] = int(np.unique(S["slope"]).size)
    sel = ok & (S["lc"] == 10)
    N["b7_auc_slope_forest_23"] = pair_auc(S["slope"][sel].astype(float), y[sel], 2, 3)


# ----------------------------------------------------------------------------- B8 даты
def b8_dates(meta, F):
    print("B8. даты и фенология")
    from scipy.stats import spearmanr
    n = len(meta)
    gap = (pd.to_datetime(meta["date_post"]) - pd.to_datetime(meta["date_pre"])).dt.days.to_numpy()
    s1gap = (pd.to_datetime(meta["s1_date_post"]) - pd.to_datetime(meta["s1_date_pre"])).dt.days.to_numpy()
    month = pd.to_datetime(meta["date_post"]).dt.month.to_numpy(); year = pd.to_datetime(meta["date_post"]).dt.year.to_numpy()
    m, dn = F["mask"], F["dnbr"]
    med_burn, med_bg, domlc, sev_mean = np.zeros(n), np.zeros(n), np.zeros(n, int), np.zeros(n)
    for i in range(n):
        b = (m[i] > 0) & (m[i] != 255)
        med_burn[i] = np.median(dn[i][b].astype(np.float32)); med_bg[i] = np.median(dn[i][m[i] == 0].astype(np.float32))
        domlc[i] = np.bincount(F["lc"][i][b]).argmax(); sev_mean[i] = m[i][b].mean()
    area = meta["burn_area_ha"].to_numpy()
    df = pd.DataFrame({"chip_id": meta["chip_id"], "gap_days": gap, "s1_gap_days": s1gap, "month": month, "year": year, "dom_lc": domlc,
                       "median_dNBR_burn": med_burn, "median_dNBR_bg": med_bg, "mean_severity": sev_mean, "burn_area_ha": area})
    C.save_table(df, "bs_b8_per_chip_dates.csv")
    N["b8_gap_median"] = float(np.median(gap)); N["b8_gap_minmax"] = [int(gap.min()), int(gap.max())]
    rows = []
    for sc, sel in (("все", np.ones(n, bool)), ("лес", domlc == 10), ("трава", domlc == 30), ("пашня", domlc == 40)):
        if sel.sum() < 8:
            continue
        r1, r2, r3 = spearmanr(gap[sel], med_burn[sel]), spearmanr(gap[sel], area[sel]), spearmanr(gap[sel], sev_mean[sel])
        rows.append({"доминирующий покров гари": sc, "чипов": int(sel.sum()), "ρ(разрыв, медиана dNBR гари)": r1.statistic, "p": r1.pvalue,
                     "ρ(разрыв, площадь)": r2.statistic, "p ": r2.pvalue, "ρ(разрыв, средняя степень)": r3.statistic, "p  ": r3.pvalue,
                     "медиана dNBR гари: разрыв ≤15 д": float(np.median(med_burn[sel & (gap <= 15)])) if (sel & (gap <= 15)).any() else np.nan,
                     "медиана dNBR гари: разрыв >25 д": float(np.median(med_burn[sel & (gap > 25)])) if (sel & (gap > 25)).any() else np.nan})
    t = C.save_table(pd.DataFrame(rows), "bs_b8_gap_correlation.csv")
    N["b8_gap_corr"] = t.iloc[:, :4].to_dict("records")
    ty = df.groupby("year").agg(чипов=("gap_days", "size"), медиана_dNBR_гари=("median_dNBR_burn", "median"), медиана_dNBR_фона=("median_dNBR_bg", "median"),
                                средняя_степень=("mean_severity", "mean"), площадь_га=("burn_area_ha", "median")).reset_index()
    tm = df.groupby("month").agg(чипов=("gap_days", "size"), медиана_dNBR_гари=("median_dNBR_burn", "median"), медиана_dNBR_фона=("median_dNBR_bg", "median"),
                                 средняя_степень=("mean_severity", "mean")).reset_index()
    C.save_table(ty, "bs_b8_by_year.csv"); C.save_table(tm, "bs_b8_by_month.csv")
    N["b8_chips_by_year"] = {int(r.year): int(r.чипов) for r in ty.itertuples()}
    N["b8_bg_dnbr_by_month"] = {int(r.month): float(r.медиана_dNBR_фона) for r in tm.itertuples()}
    N["b8_bg_dnbr_chip_std"] = float(np.std(med_bg)); N["b8_bg_dnbr_chip_q05_q95"] = [float(np.percentile(med_bg, 5)), float(np.percentile(med_bg, 95))]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for code, nm in ((10, "лес"), (30, "трава"), (40, "пашня")):
        s_ = domlc == code
        ax[0].scatter(gap[s_], med_burn[s_], s=36, color=C.LC_COLORS[code], label=f"{nm} (n={s_.sum()})", alpha=0.85, edgecolors=C.C_SURFACE, lw=0.6)
    ax[0].set_xlabel("разрыв pre→post, дней"); ax[0].set_ylabel("медиана dNBR внутри гари"); ax[0].set_title("Разрыв между снимками и сила сигнала"); ax[0].legend()
    ax[1].bar(tm["month"].astype(str), tm["чипов"], color=C.C_BLUE); ax[1].set_xlabel("месяц post-снимка"); ax[1].set_ylabel("число чипов"); ax[1].set_title("Сезонность")
    ax[2].hist(med_bg, bins=np.arange(-0.3, 0.35, 0.025), color=C.C_AQUA, edgecolor=C.C_SURFACE); ax[2].axvline(0, color=C.C_INK, lw=1)
    ax[2].set_xlabel("медиана dNBR ФОНА по чипу"); ax[2].set_ylabel("число чипов"); ax[2].set_title("Фенологический сдвиг dNBR несгоревшего фона")
    fig.suptitle("BS: даты — только для анализа (в тесте их нет)", y=1.02)
    C.savefig(fig, "eda_bs_08_dates.png")
    return df


# ----------------------------------------------------------------------------- B9 валидация
def b9_validation(meta, S, H, run_gbm: bool):
    print("B9. валидация")
    from sklearn.model_selection import KFold
    folds = pd.read_csv(C.ROOT / "configs" / "folds_bs.csv").set_index("chip_id").loc[meta["chip_id"]]
    fold = folds["fold"].to_numpy(); val24 = (folds["temporal_split"] == "val").to_numpy()
    cls = H.sum(1).sum(-1)                                  # [чип, класс]
    rows = []
    for k in range(5):
        s_ = fold == k
        rows.append({"fold": k, "чипов": int(s_.sum()), "блоков": int(folds["group"][s_].nunique()), **{f"доля кл.{c_}": cls[s_, c_].sum() / cls[s_].sum() for c_ in range(4)},
                     "из них 2024": int((s_ & val24).sum())})
    rows.append({"fold": "val=2024", "чипов": int(val24.sum()), "блоков": int(folds["group"][val24].nunique()), **{f"доля кл.{c_}": cls[val24, c_].sum() / cls[val24].sum() for c_ in range(4)},
                 "из них 2024": int(val24.sum())})
    C.save_table(pd.DataFrame(rows), "bs_b9_folds.csv")
    N["b9_group_leak"] = int((folds.groupby("group")["fold"].nunique() > 1).sum()); N["b9_blocks"] = int(folds["group"].nunique())
    N["b9_unique_fire_events"] = int(meta["fire_event_id"].nunique())
    # пересечения чипов между собой (bbox) и между фолдами
    x0, y0, x1, y1, ep = (meta[c_].to_numpy() for c_ in ("x_min", "y_min", "x_max", "y_max", "epsg"))
    ov = (x0[:, None] < x1[None]) & (x1[:, None] > x0[None]) & (y0[:, None] < y1[None]) & (y1[:, None] > y0[None]) & (ep[:, None] == ep[None])
    iu = np.triu(ov, 1)
    N["b9_overlapping_chip_pairs"] = int(iu.sum()); N["b9_overlapping_pairs_across_group_folds"] = int((iu & (fold[:, None] != fold[None])).sum())
    rs_fold = np.empty(len(meta), int)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=C.SEED).split(np.arange(len(meta)))):
        rs_fold[te] = k
    N["b9_overlapping_pairs_across_random_folds"] = int((iu & (rs_fold[:, None] != rs_fold[None])).sum())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dist = np.where(ep[:, None] == ep[None], np.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None]), np.inf)
    np.fill_diagonal(dist, np.inf)
    for nm, f_ in (("group", fold), ("random", rs_fold)):
        dd = np.where(f_[:, None] != f_[None], dist, np.inf).min(1)
        N[f"b9_nearest_other_fold_chip_km_median_{nm}"] = float(np.median(dd[np.isfinite(dd)]) / 1000)
        N[f"b9_chips_with_other_fold_neighbor_lt10km_{nm}"] = int((dd < 10240).sum())

    def rule_cv(fold_of_chip, val_sel=None):
        conf = np.zeros((4, 4))
        splits = [(fold_of_chip != k, fold_of_chip == k) for k in range(5)] if val_sel is None else [(~val_sel, val_sel)]
        for tr, te in splits:
            Ht = H[tr].sum(0)
            thr_g, _ = best_thresholds(Ht.sum(0))
            for j in range(H.shape[1]):
                h = Ht[j]
                enough = h[1:].sum() >= 20000 and all(h[k].sum() >= 2000 for k in (1, 2, 3))
                thr = best_thresholds(h)[0] if enough else thr_g
                conf += conf_from_hist(H[te].sum(0)[j], thr)
        return sev_metrics_from_conf(conf)
    res = []
    for nm, r in (("случайный KFold по чипам", rule_cv(rs_fold)), ("GroupKFold по блокам 50 км (folds_bs.csv)", rule_cv(fold)), ("временной сплит (val = 2024)", rule_cv(None, val24))):
        res.append({"модель": "пороги dNBR по landcover", "схема": nm, **r})

    if run_gbm:
        import lightgbm as lgb
        ok = (S["mask"] != 255)
        feats = ["dNBR", "NBR_post", "NBR_pre", "dNDVI", "NDVI_pre", "dNBR2", "MIRBI_post", "BAI_post", "VIS_post", "dVIS", "dSWIR2", "dNIR", "dVH_m5", "dVV_m5", "lc", "slope"]
        X = np.c_[tuple(S[f].astype(np.float32) for f in feats)][ok]; yv = S["mask"][ok].astype(int); ch = S["chip"][ok].astype(int)
        params = dict(objective="multiclass", num_class=4, n_estimators=200, learning_rate=0.08, num_leaves=63, min_child_samples=100, subsample=0.5, subsample_freq=1,
                      colsample_bytree=0.8, random_state=C.SEED, deterministic=True, force_col_wise=True, n_jobs=C.N_THREADS, verbose=-1)

        def gbm_cv(fold_of_chip, val_sel=None):
            conf = np.zeros((4, 4))
            splits = [(fold_of_chip[ch] != k, fold_of_chip[ch] == k) for k in range(5)] if val_sel is None else [(~val_sel[ch], val_sel[ch])]
            for tr, te in splits:
                mdl = lgb.LGBMClassifier(**params).fit(X[tr], yv[tr])
                conf_add(conf, yv[te], mdl.predict_proba(X[te]).argmax(1))
            return sev_metrics_from_conf(conf)
        for nm, r in (("случайный KFold по чипам", gbm_cv(rs_fold)), ("GroupKFold по блокам 50 км (folds_bs.csv)", gbm_cv(fold)), ("временной сплит (val = 2024)", gbm_cv(None, val24))):
            res.append({"модель": f"LightGBM, {len(feats)} попиксельных признаков ({SAMPLE_FRAC:.0%} пикселей)", "схема": nm, **r})
    t = pd.DataFrame(res)
    t["BS-часть Score = 0.35·IoU_burn + 0.30·mIoU_sev"] = 0.35 * t["IoU_burn"] + 0.30 * t["mIoU_sev"]
    C.save_table(t, "bs_b9_kfold_vs_group.csv", floatfmt="%.4f")
    N["b9_results"] = t.to_dict("records")

    fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    sch = list(dict.fromkeys(t["схема"]))
    for a_, metric in zip(ax, ("IoU_burn", "mIoU_sev")):
        for i, mdl in enumerate(dict.fromkeys(t["модель"])):
            v = [t[(t["модель"] == mdl) & (t["схема"] == s)][metric].iloc[0] for s in sch]
            b = a_.bar(np.arange(len(sch)) + (i - 0.5) * 0.38, v, 0.36, color=C.PALETTE[i], label=mdl.split(",")[0])
            a_.bar_label(b, fmt="%.3f", fontsize=11)
        a_.set_xticks(np.arange(len(sch))); a_.set_xticklabels([s.replace(" (", "\n(").replace(" по ", "\nпо ", 1) for s in sch], fontsize=11); a_.set_title(metric); a_.set_ylim(0, 1)
    ax[0].legend(loc="upper right", fontsize=10)
    fig.suptitle("BS: одна и та же модель при разных схемах валидации", y=1.02)
    C.savefig(fig, "eda_bs_09_validation.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true", help="пересчитать eda/cache/bs_*.npz")
    ap.add_argument("--no-gbm", action="store_true", help="пропустить LightGBM в п.9 (быстрее)")
    ap.add_argument("--only-b4c", action="store_true", help="пересчитать только п. B4c (окружность события), ~1 мин")
    args = ap.parse_args()
    np.random.seed(C.SEED)
    C.setup_style(); C.ensure_dirs()
    t0 = time.time()
    meta, S, F, PC = load_all(use_cache=not args.no_cache)
    H = chip_lc_hist(F)
    b1_classes(meta, F, PC)
    b2_indices(S)
    thr_glob, thr_lc = b3_thresholds(meta, F, H)
    if args.only_b4c:                                         # быстрый пересчёт одного пункта: числа дописываются в существующий json
        import json
        keep = set(N)
        b4c_event_circle(meta, F, thr_lc)
        path = C.TABLE_DIR / "bs_numbers.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        new = {k: v for k, v in N.items() if k not in keep}
        N.clear(); N.update(old); N.update(new)
        C.save_numbers(N, "bs_numbers.json")
        print(f"готово за {time.time() - t0:.0f} с")
        return
    b4_false_positives(meta, S, F)
    b4b_event_zone(meta, F, thr_glob)
    b4c_event_circle(meta, F, thr_lc)
    b5_clouds(meta, F, thr_glob)
    b6_sar(S, PC)
    b7_relief(S, PC)
    b8_dates(meta, F)
    b9_validation(meta, S, H, run_gbm=not args.no_gbm)
    C.save_numbers(N, "bs_numbers.json")
    print(f"готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
