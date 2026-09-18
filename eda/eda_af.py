"""EDA модуля AF (активное горение, VIIRS 375 м). Пересчитывает все таблицы и фигуры части A.

Запуск из корня репозитория:  python eda/eda_af.py [--no-cache]
Выход: eda/tables/af_*.csv, eda/tables/af_numbers.json, reports/figures/eda_af_*.png, кэш eda/cache/af_pixels.npz.
Геопривязка и даты используются ТОЛЬКО для анализа и валидации, не как признаки модели.
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

from firemon.io import read_af_chip, read_meta  # noqa: E402
from firemon.splits import af_groups  # noqa: E402

WINDOWS = (5, 7, 11, 15)
HOT_I4, HOT_D = 320.0, 20.0          # «горячий» пиксель п.2
BASE_I4, BASE_D = 320.0, 15.0        # правило-заглушка (experiments.md №0)
PERS_I4, PERS_D = 310.0, 15.0        # «горячий» для карты повторяемости п.5
N_BG_PER_CHIP = 500
N = {}                               # ключевые числа → af_numbers.json


# ----------------------------------------------------------------------------- загрузка
def load_all():
    meta = read_meta(C.DATA_DIR, "af").copy()
    ids = meta["chip_id"].tolist()
    t0 = time.time()
    with ThreadPoolExecutor(C.N_THREADS) as ex:
        chips = list(ex.map(lambda c: read_af_chip(C.DATA_DIR, c), ids))
    V = np.stack([c["viirs"] for c in chips])
    A = np.stack([c["aux"] for c in chips])
    M = np.stack([c["mask"] for c in chips])
    valid = np.stack([c["valid"] for c in chips])
    print(f"прочитано {len(ids)} AF-чипов за {time.time() - t0:.1f} с")
    meta["dt"] = pd.to_datetime(meta["acq_datetime"], utc=True)
    meta["hour"] = meta["dt"].dt.hour
    meta["month"] = meta["dt"].dt.month
    meta["year"] = meta["dt"].dt.year
    meta["tile"] = af_groups(meta)
    meta["night"] = np.nanmedian(V[:, 5].reshape(len(ids), -1), axis=1) > 90
    return meta, V, A, M, valid


# ----------------------------------------------------------------------------- контекст
def ctx_stats(x: np.ndarray, ok: np.ndarray, w: int):
    """Среднее и std соседей в окне w×w без центрального пикселя, только по ok-пикселям."""
    k = float(w * w)
    xz = np.where(ok, x, 0.0).astype(np.float64)
    okf = ok.astype(np.float64)
    s = ndi.uniform_filter(xz, w, mode="constant") * k - xz
    s2 = ndi.uniform_filter(xz * xz, w, mode="constant") * k - xz * xz
    n = ndi.uniform_filter(okf, w, mode="constant") * k - okf
    n = np.where(n < 2.5, np.nan, n)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0))
    return mean, std


def build_pixel_table(meta, V, A, M, valid, use_cache: bool) -> dict:
    """Таблица пикселей: весь огонь + кандидаты + случайный фон, с контекстными признаками. Кэшируется."""
    path = C.CACHE_DIR / "af_pixels.npz"
    if use_cache and path.exists():
        print(f"кэш: {path.relative_to(C.ROOT)}")
        return dict(np.load(path))
    t0 = time.time()
    rng = np.random.RandomState(C.SEED)
    cols: dict[str, list] = {}

    def add(name, arr):
        cols.setdefault(name, []).append(np.asarray(arr))

    for i in range(len(meta)):
        v, a, m, ok = V[i], A[i], M[i], valid[i]
        i4, i5 = v[3], v[4]
        d = i4 - i5
        night = v[5] > 90
        fire = m == 1
        cand = ok & np.where(night, (i4 > 295) & (d > 8), (i4 > PERS_I4) & (d > PERS_D))
        bg = np.zeros_like(ok)
        idx = np.flatnonzero(ok & ~fire)
        bg.flat[rng.choice(idx, min(N_BG_PER_CHIP, idx.size), replace=False)] = True
        sel = fire | cand | bg
        rr, cc = np.nonzero(sel)
        add("chip", np.full(rr.size, i, np.int32)); add("row", rr.astype(np.int16)); add("col", cc.astype(np.int16))
        add("fire", fire[sel]); add("cand", cand[sel]); add("bg", bg[sel]); add("night", night[sel])
        for j, nm in enumerate(["I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith"]):
            add(nm, v[j][sel])
        for j, nm in enumerate(["landcover", "dem", "t2m", "rh2m", "wind_speed"]):
            add(nm, a[j][sel])
        add("d45", d[sel])
        add("dist_invalid", ndi.distance_transform_edt(ok)[sel].astype(np.float32))
        # «робастный» фон: из статистики соседей исключаются потенциальные очаги (как в VNP14)
        ok_rob = ok & ~np.where(night, d > 10, d > 25)
        for w in WINDOWS:
            for fname, x in (("I4", i4), ("d45", d)):
                for vname, okk in (("", ok), ("_rob", ok_rob)):
                    mean, std = ctx_stats(x, okk, w)
                    add(f"delta_{fname}_w{w}{vname}", (x - mean)[sel].astype(np.float32))
                    add(f"z_{fname}_w{w}{vname}", ((x - mean) / np.maximum(std, 0.5))[sel].astype(np.float32))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(path, **out)
    print(f"таблица пикселей: {out['fire'].size:,} строк, {time.time() - t0:.1f} с → {path.relative_to(C.ROOT)}")
    return out


# ----------------------------------------------------------------------------- A1 дисбаланс
def a1_imbalance(meta, M, valid):
    print("A1. дисбаланс")
    fire = M == 1
    n_fire = fire.reshape(len(M), -1).sum(1)
    N["a1_fire_px"] = int(fire.sum())
    N["a1_fire_frac_all"] = fire.sum() / fire.size
    N["a1_fire_frac_valid"] = fire.sum() / valid.sum()
    N["a1_chips_with_fire"] = int((n_fire > 0).sum())
    N["a1_chips_no_fire"] = int((n_fire == 0).sum())
    pos = n_fire[n_fire > 0]
    N["a1_nfire_median_pos"] = float(np.median(pos)); N["a1_nfire_max"] = int(pos.max())
    N["a1_nfire_q25_q75"] = [float(np.percentile(pos, 25)), float(np.percentile(pos, 75))]
    N["a1_meta_mismatch"] = int((meta["n_fire_px"].fillna(0).to_numpy() != n_fire).sum())
    # топ-10 % чипов держат какую долю огня
    srt = np.sort(n_fire)[::-1]
    N["a1_top10pct_chips_share"] = srt[: len(srt) // 10].sum() / srt.sum()

    sizes, per_chip = [], []
    s8 = np.ones((3, 3), bool)
    for i in range(len(M)):
        lab, n = ndi.label(fire[i], structure=s8)
        per_chip.append(n)
        if n:
            sizes.extend(np.bincount(lab.ravel())[1:].tolist())
    sizes = np.array(sizes)
    N["a1_n_components"] = int(sizes.size)
    N["a1_comp_median"] = float(np.median(sizes)); N["a1_comp_max"] = int(sizes.max())
    bins = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 25), (26, 50), (51, 10 ** 6)]
    rows = []
    for lo, hi in bins:
        s = sizes[(sizes >= lo) & (sizes <= hi)]
        rows.append({"размер компоненты, px": f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 10 ** 6 else f"≥{lo}"),
                     "компонент": s.size, "доля компонент": s.size / sizes.size,
                     "пикселей огня": int(s.sum()), "доля пикселей огня": s.sum() / sizes.sum()})
    C.save_table(pd.DataFrame(rows), "af_a1_components.csv")
    N["a1_single_px_comp_share"] = (sizes == 1).mean()
    N["a1_single_px_fire_share"] = (sizes == 1).sum() / sizes.sum()
    N["a1_le2_px_fire_share"] = sizes[sizes <= 2].sum() / sizes.sum()
    pc = np.array(per_chip)
    N["a1_comp_per_pos_chip_median"] = float(np.median(pc[pc > 0])); N["a1_comp_per_chip_max"] = int(pc.max())
    C.save_table(pd.DataFrame({"chip_id": meta["chip_id"], "n_fire_px": n_fire, "n_components": pc,
                               "night": meta["night"]}), "af_a1_per_chip.csv")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    ax[0].hist(pos, bins=np.arange(0, 251, 10), color=C.C_BLUE, edgecolor=C.C_SURFACE)
    ax[0].axvline(np.median(pos), color=C.C_INK, ls="--", lw=1.5)
    ax[0].text(np.median(pos) + 4, ax[0].get_ylim()[1] * 0.9, f"медиана {np.median(pos):.0f} px")
    ax[0].set_xlabel("пикселей огня на чип (только чипы с огнём)"); ax[0].set_ylabel("число чипов")
    ax[0].set_title(f"Чипов с огнём {len(pos)}, без огня {int((n_fire == 0).sum())}")
    t = pd.DataFrame(rows)
    x = np.arange(len(t))
    ax[1].bar(x - 0.2, t["доля компонент"] * 100, 0.4, label="доля компонент", color=C.C_BLUE)
    ax[1].bar(x + 0.2, t["доля пикселей огня"] * 100, 0.4, label="доля пикселей огня", color=C.C_ORANGE)
    ax[1].set_xticks(x); ax[1].set_xticklabels(t["размер компоненты, px"])
    ax[1].set_xlabel("размер связной компоненты (8-связность), px"); ax[1].set_ylabel("%")
    ax[1].set_title(f"Компонент огня {sizes.size}, медиана {np.median(sizes):.0f} px"); ax[1].legend()
    cum = np.cumsum(srt) / srt.sum()
    ax[2].plot(np.arange(1, len(srt) + 1) / len(srt) * 100, cum * 100, color=C.C_BLUE)
    ax[2].set_xlabel("доля чипов, % (по убыванию n_fire_px)"); ax[2].set_ylabel("накопленная доля пикселей огня, %")
    ax[2].set_title("Концентрация огня по чипам")
    fig.suptitle(f"AF: дисбаланс — огонь = {N['a1_fire_frac_all'] * 100:.3f} % пикселей", y=1.02)
    C.savefig(fig, "eda_af_01_imbalance.png")


# ----------------------------------------------------------------------------- A2 сигнал
def a2_signal(meta, V, A, M, valid):
    print("A2. сигнал")
    i4, i5 = V[:, 3], V[:, 4]
    d = i4 - i5
    night = V[:, 5] > 90
    fire = (M == 1)
    nf = valid & ~fire
    rows = []
    for tod, sel in (("день", ~night), ("ночь", night)):
        for cls, cm in (("огонь", fire), ("не огонь", nf)):
            for nm, x in (("I4", i4), ("I5", i5), ("I4-I5", d)):
                q = C.quantile_row(x[sel & cm], qs=(1, 5, 25, 50, 75, 95, 99))
                rows.append({"время": tod, "класс": cls, "признак": nm, **q})
    C.save_table(pd.DataFrame(rows), "af_a2_quantiles.csv")

    N["a2_I4_max"] = float(np.nanmax(i4))
    N["a2_fire_sat367_share"] = float((i4[fire] >= 367).mean())
    N["a2_fire_I4_ge355_share"] = float((i4[fire] >= 355).mean())
    N["a2_fire_I4_median_day"] = float(np.median(i4[fire & ~night])); N["a2_fire_I4_median_night"] = float(np.median(i4[fire & night]))
    N["a2_fire_d_median_day"] = float(np.median(d[fire & ~night])); N["a2_fire_d_median_night"] = float(np.median(d[fire & night]))
    N["a2_bg_I4_median_day"] = float(np.median(i4[nf & ~night])); N["a2_bg_I4_median_night"] = float(np.median(i4[nf & night]))
    N["a2_bg_I4_q99_day"] = float(np.percentile(i4[nf & ~night], 99)); N["a2_bg_I4_q99_night"] = float(np.percentile(i4[nf & night], 99))
    N["a2_fire_d_lt10"] = int((d[fire] < 10).sum()); N["a2_fire_d_lt0"] = int((d[fire] < 0).sum())
    N["a2_fire_I4_lt310_day"] = int((i4[fire & ~night] < 310).sum())

    lc = A[:, 0]
    rows = []
    for name, hot in (("I4>320", i4 > HOT_I4), ("I4-I5>20", d > HOT_D), ("I4>320 и I4-I5>20", (i4 > HOT_I4) & (d > HOT_D)),
                      ("правило-заглушка I4>320 и I4-I5>15", (i4 > BASE_I4) & (d > BASE_D))):
        hot = hot & valid
        for tod, sel in (("день", ~night), ("ночь", night)):
            h = hot & sel
            rows.append({"условие": name, "время": tod, "горячих": int(h.sum()), "из них огонь": int((h & fire).sum()),
                         "не огонь": int((h & ~fire).sum()), "precision": (h & fire).sum() / max(h.sum(), 1),
                         "recall": (h & fire).sum() / max((fire & sel).sum(), 1)})
    t = C.save_table(pd.DataFrame(rows), "af_a2_hot_nonfire.csv")
    r = t[(t["условие"] == "I4>320 и I4-I5>20")]
    N["a2_hot_nonfire_day"] = int(r[r["время"] == "день"]["не огонь"].iloc[0])
    N["a2_hot_nonfire_night"] = int(r[r["время"] == "ночь"]["не огонь"].iloc[0])

    hotnf = (i4 > HOT_I4) & (d > HOT_D) & nf
    rows = []
    for code, nm in C.LC_NAMES.items():
        a_all = (lc == code) & valid
        if a_all.sum() == 0:
            continue
        rows.append({"landcover": code, "название": nm, "доля площади": a_all.sum() / valid.sum(),
                     "горячих не-огонь": int((hotnf & a_all).sum()), "доля горячих не-огонь": (hotnf & a_all).sum() / max(hotnf.sum(), 1),
                     "обогащение": ((hotnf & a_all).sum() / max(hotnf.sum(), 1)) / (a_all.sum() / valid.sum()),
                     "на 1 млн px покрова": (hotnf & a_all).sum() / a_all.sum() * 1e6})
    C.save_table(pd.DataFrame(rows), "af_a2_hot_nonfire_landcover.csv")

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for r_, (tod, sel) in enumerate((("день", ~night), ("ночь", night))):
        for c_, (nm, x, bins) in enumerate((("I4, K", i4, np.arange(230, 362, 2)), ("I5, K", i5, np.arange(220, 342, 2)),
                                            ("I4 − I5, K", d, np.arange(-10, 85, 1.5)))):
            C.log_hist(ax[r_, c_], {"огонь": x[sel & fire], "не огонь": x[sel & nf]}, bins, nm, f"{nm.split(',')[0]} — {tod}",
                       colors=[C.C_RED, C.C_BLUE])
    fig.suptitle("AF: яркостные температуры пикселей огня и фона (нормированные гистограммы, лог-шкала)", y=1.0)
    fig.tight_layout()
    C.savefig(fig, "eda_af_02_signal.png")


def a2b_rule(V, M, valid):
    """Какое простое правило лучше всего воспроизводит эталон: перебор порогов (I4, I4−I5) отдельно днём и ночью."""
    print("A2b. подбор порогового правила")
    i4 = np.nan_to_num(V[:, 3]); d = np.nan_to_num(V[:, 3] - V[:, 4])
    night = V[:, 5] > 90
    fire = M == 1
    rows, best = [], {}
    for tod, sel in (("день", ~night & valid), ("ночь", night & valid)):
        # достаточно кандидатов: всё, что ниже самых мягких порогов сетки, правилу не нужно
        c = sel & (i4 > 288) & (d > 0)
        x4, xd, y = i4[c], d[c], fire[c]
        n_fire = int((fire & (V[:, 5] > 90 if tod == "ночь" else ~night)).sum())
        for t4 in np.arange(290.0, 346.0, 1.0):
            m4 = x4 > t4
            for td in (0.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0):
                p = m4 & (xd > td)
                tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = n_fire - tp
                rows.append({"время": tod, "I4 >": t4, "I4-I5 >": td, "TP": tp, "FP": fp, "FN": fn,
                             "precision": tp / max(tp + fp, 1), "recall": tp / max(n_fire, 1), "F1": C.f1_from_counts(tp, fp, fn)})
    t = pd.DataFrame(rows)
    C.save_table(t, "af_a2b_rule_grid.csv")
    tot = dict(tp=0, fp=0, fn=0)
    for tod in ("день", "ночь"):
        b = t[t["время"] == tod].sort_values("F1", ascending=False).iloc[0]
        best[tod] = b
        for k in tot:
            tot[k] += int(b[k.upper()])
        N[f"a2b_best_rule_{'day' if tod == 'день' else 'night'}"] = {"I4>": float(b["I4 >"]), "d>": float(b["I4-I5 >"]), "F1": float(b["F1"]),
                                                                    "P": float(b["precision"]), "R": float(b["recall"]), "FP": int(b["FP"]), "FN": int(b["FN"])}
    N["a2b_best_rule_F1_total"] = C.f1_from_counts(tot["tp"], tot["fp"], tot["fn"])
    N["a2b_fire_I4_min_day"] = float(V[:, 3][fire & ~night].min()); N["a2b_fire_I4_min_night"] = float(V[:, 3][fire & night].min())
    N["a2b_fire_I4_q01_day"] = float(np.percentile(V[:, 3][fire & ~night], 1))
    N["a2b_fire_d_min_night"] = float((V[:, 3] - V[:, 4])[fire & night].min())
    N["a2b_fire_d_q01_day"] = float(np.percentile((V[:, 3] - V[:, 4])[fire & ~night], 1))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    for a_, tod in zip(ax, ("день", "ночь")):
        sub = t[t["время"] == tod]
        for k, td in enumerate((0.0, 10.0, 15.0, 25.0)):
            s_ = sub[sub["I4-I5 >"] == td]
            a_.plot(s_["I4 >"], s_["F1"], color=C.PALETTE[k], label=f"I4−I5 > {td:.0f} K")
        b = best[tod]
        a_.scatter([b["I4 >"]], [b["F1"]], s=90, color=C.C_INK, zorder=5)
        a_.annotate(f"F1 = {b['F1']:.3f}\nI4 > {b['I4 >']:.0f}, Δ > {b['I4-I5 >']:.0f}", (b["I4 >"], b["F1"]), xytext=(10, -40), textcoords="offset points")
        a_.set_xlabel("порог I4, K"); a_.set_ylabel("F1 по пулу пикселей"); a_.set_title(f"{tod}: F1 порогового правила"); a_.legend()
        a_.set_ylim(0, 1)
    fig.suptitle(f"AF: раздельные пороги день/ночь дают F1 = {N['a2b_best_rule_F1_total']:.3f} (единый порог-заглушка — 0.244)", y=1.02)
    C.savefig(fig, "eda_af_02b_rule.png")


# ----------------------------------------------------------------------------- A3 контекст
def a3_context(P):
    print("A3. контекстные признаки")
    fire, bg, cand, night = P["fire"], P["bg"], P["cand"], P["night"]
    base = fire | bg                       # огонь против случайного фона
    hard = fire | (cand & ~fire)           # огонь против «горячих, но не огонь» (кандидаты)
    feats = [("I4", "—", "сырое"), ("d45", "—", "сырое")]
    for w in WINDOWS:
        for f in ("I4", "d45"):
            for kind in ("delta", "z"):
                for rob in ("", "_rob"):
                    feats.append((f"{kind}_{f}_w{w}{rob}", w, "без очагов в фоне" if rob else "все соседи"))
    rows = []
    for name, w, variant in feats:
        x = P[name]
        row = {"признак": name, "окно": w, "фон": variant}
        for tag, sub in (("всё", np.ones_like(fire)), ("день", ~night), ("ночь", night)):
            row[f"AUC vs случайный фон ({tag})"] = C.auc(x[base & sub], fire[base & sub])
        for tag, sub in (("день", ~night), ("ночь", night)):
            row[f"AUC vs горячие не-огонь ({tag})"] = C.auc(x[hard & sub], fire[hard & sub])
        rows.append(row)
    t = C.save_table(pd.DataFrame(rows), "af_a3_context_auc.csv", floatfmt="%.4f")
    key = "AUC vs горячие не-огонь (день)"
    best = t.sort_values(key, ascending=False).iloc[0]
    N["a3_best_feature_hard_day"] = best["признак"]; N["a3_best_auc_hard_day"] = float(best[key])
    for nm in ("I4", "d45"):
        N[f"a3_auc_raw_{nm}_hard_day"] = float(t.loc[t["признак"] == nm, key].iloc[0])
        N[f"a3_auc_raw_{nm}_rand_all"] = float(t.loc[t["признак"] == nm, "AUC vs случайный фон (всё)"].iloc[0])
    N["a3_n_cand_nonfire"] = int((cand & ~fire).sum()); N["a3_n_bg"] = int(bg.sum())

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    for a_, tod in zip(ax, ("день", "ночь")):
        col = f"AUC vs горячие не-огонь ({tod})"
        i = 0
        for f, fl in (("I4", "I4"), ("d45", "I4−I5")):
            for rob, rl, ls in (("", "все соседи", "-"), ("_rob", "фон без очагов", "--")):
                y = [t.loc[t["признак"] == f"z_{f}_w{w}{rob}", col].iloc[0] for w in WINDOWS]
                a_.plot(WINDOWS, y, marker="o", ms=8, ls=ls, color=C.PALETTE[i // 2], label=f"z({fl}), {rl}")
                i += 1
        for f, fl, c_ in (("I4", "I4 сырой", C.C_BLUE), ("d45", "I4−I5 сырой", C.C_ORANGE)):
            a_.axhline(t.loc[t["признак"] == f, col].iloc[0], color=c_, lw=1.2, ls=":", label=fl)
        a_.set_xticks(WINDOWS); a_.set_xticklabels([f"{w}×{w}" for w in WINDOWS])
        a_.set_xlabel("окно фона"); a_.set_title(f"{tod}: огонь против «горячих, но не огонь»")
    ax[0].set_ylabel("ROC-AUC одиночного признака")
    h_, l_ = ax[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=11, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("AF: разделяющая сила контекстных z-score в зависимости от окна", y=1.02)
    C.savefig(fig, "eda_af_03_context_auc.png")
    return t


# ----------------------------------------------------------------------------- A4 день/ночь
def a4_daynight(meta, V, M, valid, P):
    print("A4. день/ночь")
    i4, d = V[:, 3], V[:, 3] - V[:, 4]
    fire = M == 1
    nfire = fire.reshape(len(M), -1).sum(1)
    hotnf = ((i4 > HOT_I4) & (d > HOT_D) & valid & ~fire).reshape(len(M), -1).sum(1)
    basefp = ((i4 > BASE_I4) & (d > BASE_D) & valid & ~fire).reshape(len(M), -1).sum(1)
    bgmed = np.array([np.nanmedian(i4[i][valid[i] & ~fire[i]]) if valid[i].any() else np.nan for i in range(len(M))])
    nvalid = valid.reshape(len(M), -1).sum(1)
    df = pd.DataFrame({"chip_id": meta["chip_id"], "night": meta["night"].to_numpy(), "hour": meta["hour"].to_numpy(),
                       "n_fire": nfire, "hot_nonfire": hotnf, "base_fp": basefp, "bg_I4_median": bgmed, "n_valid": nvalid})
    sz_mixed = ((V[:, 5] > 90).reshape(len(M), -1).mean(1))
    N["a4_mixed_daynight_chips"] = int(((sz_mixed > 0) & (sz_mixed < 1)).sum())

    def agg(g):
        return pd.Series({"чипов": len(g), "чипов с огнём": int((g.n_fire > 0).sum()), "пикселей огня": int(g.n_fire.sum()),
                          "доля огня, ‰": g.n_fire.sum() / g.n_valid.sum() * 1000,
                          "медиана n_fire (чипы с огнём)": g.n_fire[g.n_fire > 0].median(),
                          "фон I4: медиана по чипам, K": g.bg_I4_median.median(),
                          "горячих не-огонь (I4>320, Δ>20)": int(g.hot_nonfire.sum()),
                          "FP правила-заглушки": int(g.base_fp.sum())})
    t = df.groupby("night").apply(agg).reset_index()
    t["night"] = t["night"].map({False: "день", True: "ночь"})
    C.save_table(t, "af_a4_daynight.csv")
    th = df.groupby("hour").apply(agg).reset_index()
    C.save_table(th, "af_a4_by_hour.csv")
    for tod, k in (("день", "day"), ("ночь", "night")):
        r = t[t["night"] == tod].iloc[0]
        N[f"a4_chips_{k}"] = int(r["чипов"]); N[f"a4_fire_px_{k}"] = int(r["пикселей огня"])
        N[f"a4_bg_I4_{k}"] = float(r["фон I4: медиана по чипам, K"]); N[f"a4_base_fp_{k}"] = int(r["FP правила-заглушки"])

    # блик: отражательные каналы у дневных кандидатов
    day = ~P["night"]
    groups = {"огонь (день)": P["fire"] & day, "горячие не-огонь (день)": P["cand"] & ~P["fire"] & day & (P["I4"] > HOT_I4) & (P["d45"] > HOT_D),
              "кандидаты не-огонь (день)": P["cand"] & ~P["fire"] & day, "случайный фон (день)": P["bg"] & day}
    rows = []
    for g, sel in groups.items():
        for b in ("I1", "I2", "I3"):
            rows.append({"группа": g, "канал": b, **C.quantile_row(P[b][sel], qs=(5, 25, 50, 75, 95, 99))})
    C.save_table(pd.DataFrame(rows), "af_a4_reflectance_hot.csv")
    hard = day & (P["fire"] | (P["cand"] & ~P["fire"]))
    rows = []
    ndvi = (P["I2"] - P["I1"]) / (P["I2"] + P["I1"] + 1e-6)
    for nm, x in (("I1", P["I1"]), ("I2", P["I2"]), ("I3", P["I3"]), ("NDVI=(I2-I1)/(I2+I1)", ndvi), ("I3/I2", P["I3"] / (P["I2"] + 1e-6)),
                  ("I4/I5", P["I4"] / P["I5"])):
        rows.append({"признак": nm, "AUC огонь vs кандидаты не-огонь (день)": C.auc(x[hard], P["fire"][hard])})
    ta = C.save_table(pd.DataFrame(rows), "af_a4_reflectance_auc.csv")
    N["a4_auc_refl_day"] = {r["признак"]: float(r.iloc[1]) for _, r in ta.iterrows()}
    glint = day & ~P["fire"] & P["cand"] & (P["I3"] > 0.3) & (P["I1"] > 0.3)
    N["a4_glint_like_cand"] = int(glint.sum())
    N["a4_fire_I3_gt03"] = int((P["fire"] & day & (P["I3"] > 0.3)).sum())
    # у ночных чипов отражательные каналы
    nt = P["night"]
    N["a4_night_I1_finite_share"] = float(np.isfinite(P["I1"][nt]).mean()) if nt.any() else float("nan")
    N["a4_night_I1_abs_median"] = float(np.nanmedian(np.abs(P["I1"][nt]))) if nt.any() else float("nan")
    N["a4_night_I3_q99"] = float(np.nanpercentile(P["I3"][nt], 99)) if nt.any() else float("nan")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    hrs = th["hour"].to_numpy()
    ax[0].bar(np.arange(len(hrs)), th["чипов"], color=C.C_BLUE, label="все чипы")
    ax[0].bar(np.arange(len(hrs)), th["чипов с огнём"], color=C.C_ORANGE, width=0.5, label="с огнём")
    ax[0].set_xticks(np.arange(len(hrs))); ax[0].set_xticklabels(hrs)
    ax[0].set_xlabel("час пролёта, UTC"); ax[0].set_ylabel("число чипов"); ax[0].set_title("Пролёты по часам"); ax[0].legend()
    bins = np.arange(255, 325, 2.5)
    ax[1].hist(df.bg_I4_median[~df.night], bins=bins, color=C.C_ORANGE, alpha=0.85, label="день")
    ax[1].hist(df.bg_I4_median[df.night], bins=bins, color=C.C_VIOLET, alpha=0.85, label="ночь")
    ax[1].set_xlabel("медиана I4 фона по чипу, K"); ax[1].set_ylabel("число чипов"); ax[1].set_title("Фон I4: день против ночи"); ax[1].legend()
    sel_f, sel_h = groups["огонь (день)"], groups["кандидаты не-огонь (день)"]
    rs = np.random.RandomState(0)
    ih = rs.choice(np.flatnonzero(sel_h), min(6000, int(sel_h.sum())), replace=False)
    ax[2].scatter(P["I3"][ih], P["I4"][ih], s=5, color=C.C_BLUE, alpha=0.35, label="кандидаты не-огонь", rasterized=True)
    ax[2].scatter(P["I3"][sel_f], P["I4"][sel_f], s=5, color=C.C_RED, alpha=0.5, label="огонь", rasterized=True)
    ax[2].set_xlabel("I3 (1.61 мкм), отражение"); ax[2].set_ylabel("I4, K"); ax[2].set_title("День: I3 против I4 у кандидатов")
    ax[2].legend(markerscale=3, loc="lower right", frameon=True)
    fig.suptitle("AF: день и ночь", y=1.02)
    C.savefig(fig, "eda_af_04_daynight.png")
    return df


# ----------------------------------------------------------------------------- A5 техноген
def a5_technogenic(meta, V, A, M, valid):
    print("A5. повторяющиеся термоаномалии (техноген)")
    i4, d = V[:, 3], V[:, 3] - V[:, 4]
    night = V[:, 5] > 90
    fire = M == 1
    hot = (i4 > PERS_I4) & (d > PERS_D) & valid
    base = (i4 > BASE_I4) & (d > BASE_D) & valid
    lc = A[:, 0]
    tiles = meta.groupby("tile").indices
    N["a5_tiles"] = len(tiles)
    N["a5_passes_per_tile_median"] = float(np.median([len(v) for v in tiles.values()]))
    N["a5_passes_per_tile_minmax"] = [int(min(len(v) for v in tiles.values())), int(max(len(v) for v in tiles.values()))]

    nfp = (i4 > 294) & (d > 10) & valid & night & ~fire      # ночные «как огонь по порогам, но не огонь»
    nfire_n = fire & night
    N["a5_night_rule_fp"] = int(nfp.sum()); N["a5_night_rule_fp_chips"] = int(nfp.reshape(len(M), -1).any(1).sum())
    N["a5_night_rule_fp_lc_share"] = {int(c_): float((lc[nfp] == c_).mean()) for c_ in C.LC_NAMES if (lc[nfp] == c_).any()}
    N["a5_night_fire_lc_share"] = {int(c_): float((lc[nfire_n] == c_).mean()) for c_ in C.LC_NAMES if (lc[nfire_n] == c_).any()}
    N["a5_night_rule_fp_d_median"] = float(np.median(d[nfp])); N["a5_night_fire_d_median"] = float(np.median(d[nfire_n]))
    nfp_on_pers = 0
    rows, lc_any, lc_night, examples = [], [], [], []
    tot = dict(pers_any=0, pers_night=0, fp_any=0, fp_night=0, fp_total=int((base & ~fire).sum()),
               fp_night_total=int((base & ~fire & night).sum()), refire=0, fire_px_tile=0)
    for tile, idx in tiles.items():
        idx = np.asarray(idx)
        n_hot = hot[idx].sum(0); n_fire = fire[idx].sum(0)
        n_hot_night = (hot[idx] & night[idx]).sum(0)
        # ночью порог мягче: ночной фон холодный, техноген виден при I4>295, Δ>8
        hot_n2 = ((i4[idx] > 295) & (d[idx] > 8) & valid[idx] & night[idx]).sum(0)
        pers_any = (n_hot >= 3) & (n_fire == 0)
        pers_night = (hot_n2 >= 2) & (n_fire == 0)
        fp_any = int((base[idx] & ~fire[idx] & pers_any[None]).sum())
        fp_night = int((base[idx] & ~fire[idx] & pers_night[None]).sum())
        nfp_on_pers += int((nfp[idx] & pers_night[None]).sum())
        tot["pers_any"] += int(pers_any.sum()); tot["pers_night"] += int(pers_night.sum())
        tot["fp_any"] += fp_any; tot["fp_night"] += fp_night
        tot["refire"] += int((n_fire >= 2).sum()); tot["fire_px_tile"] += int((n_fire >= 1).sum())
        lc0 = lc[idx[0]]
        lc_any.append(lc0[pers_any]); lc_night.append(lc0[pers_night])
        rows.append({"tile": tile, "пролётов": len(idx), "ночных": int(night[idx].reshape(len(idx), -1).mean(1).round().sum()),
                     "пикс. горячий≥3 и огонь=0": int(pers_any.sum()), "пикс. ночной горячий≥2 и огонь=0": int(pers_night.sum()),
                     "FP заглушки на них (любые)": fp_any, "FP заглушки на них (ночные)": fp_night,
                     "пикс. огонь≥2 пролётов": int((n_fire >= 2).sum())})
        if pers_night.sum() > 0:
            examples.append((int(pers_night.sum()), tile, idx, pers_night, hot_n2))
    t = C.save_table(pd.DataFrame(rows).sort_values("пикс. ночной горячий≥2 и огонь=0", ascending=False), "af_a5_tiles.csv")
    N.update({"a5_pers_any_px": tot["pers_any"], "a5_pers_night_px": tot["pers_night"], "a5_base_fp_total": tot["fp_total"],
              "a5_base_fp_on_pers_any": tot["fp_any"], "a5_base_fp_on_pers_night": tot["fp_night"],
              "a5_base_fp_night_total": tot["fp_night_total"],
              "a5_tiles_with_pers_night": int((t["пикс. ночной горячий≥2 и огонь=0"] > 0).sum()),
              "a5_refire_px": tot["refire"], "a5_fire_px_unique_loc": tot["fire_px_tile"]})

    N["a5_night_rule_fp_on_pers_night"] = nfp_on_pers
    rows = []
    la, ln = np.concatenate(lc_any), np.concatenate(lc_night)
    area = {code: ((lc == code) & valid).sum() / valid.sum() for code in C.LC_NAMES}
    for code, nm in C.LC_NAMES.items():
        if area[code] == 0:
            continue
        rows.append({"landcover": code, "название": nm, "доля площади": area[code],
                     "дневные+ночные (горячий≥3)": int((la == code).sum()), "доля": (la == code).mean() if la.size else 0,
                     "ночные (горячий≥2)": int((ln == code).sum()), "доля ночных": (ln == code).mean() if ln.size else 0,
                     "обогащение ночных": ((ln == code).mean() / area[code]) if ln.size else 0})
    tl = C.save_table(pd.DataFrame(rows), "af_a5_persistent_landcover.csv")
    N["a5_pers_night_lc_share"] = {int(r.landcover): float(r["доля ночных"]) for _, r in tl.iterrows() if r["доля ночных"] > 0}

    # примеры: 4 тайла с наибольшим числом ночных повторяющихся аномалий
    examples.sort(key=lambda e: -e[0])
    ex = examples[:4]
    if ex:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        codes = sorted(C.LC_NAMES)
        cmap = ListedColormap([C.LC_COLORS[c] for c in codes]); norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)
        fig, ax = plt.subplots(len(ex), 3, figsize=(15, 4.6 * len(ex)), squeeze=False)
        for r_, (npx, tile, idx, pers, cnt) in enumerate(ex):
            # пролёт, где повторяющиеся пиксели самые горячие; окно 64×64 вокруг самого плотного скопления
            best = idx[np.argmax([(np.nan_to_num(d[j]) * pers).sum() * night[j].any() for j in idx])]
            dens = ndi.uniform_filter(pers.astype(float), 15)
            cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
            y0, x0 = int(np.clip(cy - 32, 0, 192)), int(np.clip(cx - 32, 0, 192))
            sl = (slice(y0, y0 + 64), slice(x0, x0 + 64))
            im = ax[r_, 0].imshow(i4[best][sl], cmap="inferno"); fig.colorbar(im, ax=ax[r_, 0], fraction=0.046, label="I4, K")
            yy, xx = np.nonzero(pers[sl]); ax[r_, 0].scatter(xx, yy, s=60, facecolors="none", edgecolors="#3987e5", lw=1.5, label="повтор. аномалия")
            yy, xx = np.nonzero(fire[best][sl])
            if yy.size:
                ax[r_, 0].scatter(xx, yy, s=18, marker="x", color="#1baf7a", label="огонь (эталон)")
            ax[r_, 0].set_title(f"{meta.chip_id.iloc[best]}: I4, ночной пролёт"); ax[r_, 0].legend(loc="upper right", fontsize=9, frameon=True)
            ax[r_, 1].imshow(lc[idx[0]][sl], cmap=cmap, norm=norm, interpolation="nearest")
            yy, xx = np.nonzero(pers[sl]); ax[r_, 1].scatter(xx, yy, s=60, facecolors="none", edgecolors=C.C_INK, lw=1.5)
            ax[r_, 1].set_title("landcover (WorldCover)")
            im = ax[r_, 2].imshow(cnt[sl], cmap="Blues", vmin=0); fig.colorbar(im, ax=ax[r_, 2], fraction=0.046, label="ночных пролётов «горячий»")
            ax[r_, 2].set_title(f"повторяемость: {npx} px в тайле, {len(idx)} пролётов")
            for a_ in ax[r_]:
                a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
        present = sorted({int(v) for _, _, idx, _, _ in ex for v in np.unique(lc[idx[0]][np.isfinite(lc[idx[0]])])})
        handles = [plt.Rectangle((0, 0), 1, 1, color=C.LC_COLORS[c]) for c in present if c in C.LC_COLORS]
        fig.legend(handles, [C.LC_NAMES[c] for c in present if c in C.LC_NAMES], loc="lower center", ncol=len(handles), fontsize=11)
        fig.suptitle("AF: повторяющиеся ночные термоаномалии без разметки «огонь» — кандидаты в техноген (фрагменты 64×64 px = 24 км)", y=0.995)
        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        C.savefig(fig, "eda_af_05_technogenic.png", tight=False)


# ----------------------------------------------------------------------------- A6 landcover/метео
def a6_landcover_meteo(meta, A, M, valid, P):
    print("A6. landcover и метео")
    lc = A[:, 0]
    fire = M == 1
    rows = []
    for code, nm in C.LC_NAMES.items():
        a_all = (lc == code) & valid
        if a_all.sum() == 0:
            continue
        rows.append({"landcover": code, "название": nm, "доля площади": a_all.sum() / valid.sum(), "пикселей огня": int((fire & a_all).sum()),
                     "доля огня": (fire & a_all).sum() / fire.sum(), "огня на 1 млн px": (fire & a_all).sum() / a_all.sum() * 1e6,
                     "обогащение": ((fire & a_all).sum() / fire.sum()) / (a_all.sum() / valid.sum())})
    tl = C.save_table(pd.DataFrame(rows), "af_a6_fire_by_landcover.csv")
    N["a6_fire_share_by_lc"] = {int(r.landcover): float(r["доля огня"]) for _, r in tl.iterrows()}
    N["a6_lc_nan_px"] = int(np.isnan(lc).sum())

    names = ["dem", "t2m", "rh2m", "wind_speed"]
    chipmed = {nm: np.nanmedian(A[:, j + 1].reshape(len(A), -1), axis=1) for j, nm in enumerate(names)}
    chipstd = {nm: np.nanstd(A[:, j + 1].reshape(len(A), -1), axis=1) for j, nm in enumerate(names)}
    has = (meta["n_fire_px"].fillna(0) > 0).to_numpy()
    base = P["fire"] | P["bg"]
    rows = []
    for nm in names:
        rows.append({"признак": nm, "медиана: чипы с огнём": float(np.median(chipmed[nm][has])), "медиана: чипы без огня": float(np.median(chipmed[nm][~has])),
                     "AUC чип (есть огонь)": C.auc(chipmed[nm], has), "AUC пиксель (огонь vs фон)": C.auc(P[nm][base], P["fire"][base]),
                     "std внутри чипа (медиана)": float(np.median(chipstd[nm]))})
    ta = C.save_table(pd.DataFrame(rows), "af_a6_meteo_auc.csv")
    N["a6_meteo_auc_chip"] = {r["признак"]: float(r["AUC чип (есть огонь)"]) for _, r in ta.iterrows()}
    N["a6_meteo_auc_pixel"] = {r["признак"]: float(r["AUC пиксель (огонь vs фон)"]) for _, r in ta.iterrows()}

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    x = np.arange(len(tl))
    ax[0].bar(x - 0.2, tl["доля площади"] * 100, 0.4, color=C.C_BLUE, label="доля площади")
    ax[0].bar(x + 0.2, tl["доля огня"] * 100, 0.4, color=C.C_ORANGE, label="доля пикселей огня")
    ax[0].set_xticks(x); ax[0].set_xticklabels(tl["название"], rotation=30, ha="right"); ax[0].set_ylabel("%")
    ax[0].set_title("Огонь по типам покрова"); ax[0].legend()
    for a_, nm, lab in zip(ax[1:], ("t2m", "rh2m", "wind_speed"), ("t2m, °C", "rh2m, %", "ветер, м/с")):
        bins = np.linspace(np.nanmin(chipmed[nm]), np.nanmax(chipmed[nm]), 18)
        a_.hist(chipmed[nm][has], bins=bins, color=C.C_ORANGE, alpha=0.85, density=True, label="чипы с огнём")
        a_.hist(chipmed[nm][~has], bins=bins, histtype="step", color=C.C_BLUE, lw=2, density=True, label="без огня")
        a_.set_xlabel(lab); a_.set_title(f"{lab.split(',')[0]}: AUC(чип) = {C.auc(chipmed[nm], has):.2f}"); a_.legend(fontsize=10)
    ax[1].set_ylabel("плотность")
    fig.suptitle("AF: покров и метеоусловия", y=1.03)
    C.savefig(fig, "eda_af_06_landcover_meteo.png")


# ----------------------------------------------------------------------------- A7 valid / NaN
def a7_valid(meta, V, A, M, valid):
    print("A7. valid / NaN")
    n = len(M)
    vch = V[:, 7] > 0.5
    fin45 = np.isfinite(V[:, 3]) & np.isfinite(V[:, 4])
    night = V[:, 5] > 90
    fire = M == 1
    N["a7_invalid_px_channel"] = int((~vch).sum()); N["a7_invalid_share_channel"] = float((~vch).mean())
    N["a7_nan_I4I5_where_valid1"] = int((vch & ~fin45).sum())
    N["a7_finite_I4I5_where_valid0"] = int((~vch & fin45).sum())
    vf = vch.reshape(n, -1).mean(1)
    N["a7_chips_validfrac_lt1_channel"] = int((vf < 1).sum()); N["a7_chips_validfrac_lt1_meta"] = int((meta["valid_frac"] < 1).sum())
    N["a7_min_valid_frac"] = float(vf.min()); N["a7_meta_vs_channel_maxdiff"] = float(np.abs(vf - meta["valid_frac"].to_numpy()).max())
    io_valid_frac = valid.reshape(n, -1).mean(1)
    N["a7_chips_iovalid_lt1"] = int((io_valid_frac < 1).sum()); N["a7_invalid_share_io"] = float((~valid).mean())
    for j, b in enumerate(["I1", "I2", "I3"]):
        nan_b = ~np.isfinite(V[:, j])
        N[f"a7_nan_{b}_day_share"] = float(nan_b[~night].mean()); N[f"a7_nan_{b}_night_share"] = float(nan_b[night].mean())
    N["a7_nan_aux_px"] = {nm: int(np.isnan(A[:, j]).sum()) for j, nm in enumerate(["landcover", "dem", "t2m", "rh2m", "wind_speed"])}
    # где невалидные пиксели: доля в компонентах, касающихся края чипа
    touch = 0
    for i in np.flatnonzero(io_valid_frac < 1):
        lab, k = ndi.label(~valid[i])
        edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        touch += int(np.isin(lab, edge[edge > 0]).sum())
    N["a7_invalid_in_edge_components_share"] = touch / max(int((~valid).sum()), 1)
    # огонь у границы валидности
    near = {1: 0, 2: 0, 3: 0, 5: 0}
    fire_in_partial = 0
    for i in np.flatnonzero(io_valid_frac < 1):
        dist = ndi.distance_transform_edt(valid[i])
        fire_in_partial += int(fire[i].sum())
        for k in near:
            near[k] += int((fire[i] & (dist <= k)).sum())
    N["a7_fire_px_in_partial_chips"] = fire_in_partial
    N["a7_fire_within_k_px_of_invalid"] = near
    N["a7_fire_on_invalid"] = int((fire & ~valid).sum())
    N["a7_mask_nodata_px"] = int((M == 255).sum())
    rows = pd.DataFrame({"chip_id": meta["chip_id"], "valid_frac_meta": meta["valid_frac"], "valid_frac_channel": vf,
                         "valid_frac_io": io_valid_frac, "night": meta["night"], "n_fire": fire.reshape(n, -1).sum(1)})
    C.save_table(rows[rows.valid_frac_io < 1].sort_values("valid_frac_io"), "af_a7_partial_chips.csv")

    worst = rows.sort_values("valid_frac_io").index[:4]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    for a_, i in zip(ax, worst):
        a_.imshow(np.where(valid[i], V[i, 3], np.nan), cmap="inferno")
        a_.imshow(np.where(~valid[i], 1.0, np.nan), cmap="Greys", vmin=0, vmax=2)
        yy, xx = np.nonzero(fire[i]); a_.scatter(xx, yy, s=8, color="#1baf7a", label=f"огонь ({yy.size} px)")
        a_.set_title(f"{meta.chip_id.iloc[i]}: valid = {io_valid_frac[i]:.2f}"); a_.legend(loc="lower right", fontsize=10, frameon=True)
        a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
    fig.suptitle("AF: чипы с наименьшей долей валидных пикселей (серое — нет данных, цвет — I4)", y=1.02)
    C.savefig(fig, "eda_af_07_valid.png")


def a7b_nan_structure(meta, V, M):
    """Где NaN в I4/I5 при valid = 1: проверяем гипотезы «край сцены», «облака», «bow-tie» (связь с sensor_zenith)."""
    print("A7b. структура NaN в I4/I5")
    n = len(M)
    vch = V[:, 7] > 0.5
    nan = ~np.isfinite(V[:, 3]) & vch
    fire = M == 1
    senz = V[:, 6]
    frac = nan.reshape(n, -1).mean(1)
    N["a7b_nan_same_in_I4_I5"] = bool((np.isfinite(V[:, 3]) == np.isfinite(V[:, 4])).all())
    N["a7b_chips_with_nan"] = int((frac > 0).sum()); N["a7b_chips_with_nan_share"] = float((frac > 0).mean())
    N["a7b_nan_share_all"] = float(nan.mean()); N["a7b_nan_share_in_affected_chips"] = float(frac[frac > 0].mean())
    rows = []
    edges = [0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 71]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s_ = (senz >= lo) & (senz < hi) & vch
        rows.append({"sensor_zenith, °": f"{lo}–{hi}", "доля пикселей": s_.mean(), "доля NaN": nan[s_].mean() if s_.any() else np.nan,
                     "пикселей огня": int((fire & s_).sum())})
    t = C.save_table(pd.DataFrame(rows), "af_a7b_nan_by_sensor_zenith.csv")
    chip_senz = np.nanmedian(senz.reshape(n, -1), axis=1)
    from scipy.stats import spearmanr
    N["a7b_spearman_chip_senz_vs_nan"] = float(spearmanr(chip_senz, frac).statistic)
    N["a7b_min_senz_with_nan"] = float(np.nanmin(senz[nan])) if nan.any() else float("nan")
    for lo, hi in ((0, 30), (30, 45), (45, 71)):
        s_ = (chip_senz >= lo) & (chip_senz < hi)
        N[f"a7b_chips_senz_{lo}_{hi}"] = {"чипов": int(s_.sum()), "доля чипов с NaN": float((frac[s_] > 0).mean()), "средняя доля NaN": float(frac[s_].mean())}
    night = meta["night"].to_numpy()
    N["a7b_nan_share_day_night"] = [float(frac[~night].mean()), float(frac[night].mean())]

    def run_lengths(a):
        d = np.diff(np.pad(a.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        return (np.nonzero(d == -1)[1] - np.nonzero(d == 1)[1])
    idx = np.flatnonzero(frac > 0.02)
    hr = np.concatenate([run_lengths(nan[i]) for i in idx]); vr = np.concatenate([run_lengths(nan[i].T) for i in idx])
    N["a7b_run_len_horizontal_q50_q95"] = [float(np.percentile(hr, 50)), float(np.percentile(hr, 95))]
    N["a7b_run_len_vertical_q50_q95"] = [float(np.percentile(vr, 50)), float(np.percentile(vr, 95))]
    adj = ndi.binary_dilation(nan, structure=np.ones((1, 3, 3), bool)) & ~nan & vch
    rest = vch & ~nan & ~adj
    N["a7b_I5_median_adjacent_vs_rest"] = [float(np.nanmedian(V[:, 4][adj])), float(np.nanmedian(V[:, 4][rest]))]
    N["a7b_fire_adjacent_to_nan"] = int((fire & adj).sum()); N["a7b_fire_adjacent_to_nan_share"] = float((fire & adj).sum() / fire.sum())
    N["a7b_fire_px_in_chips_with_nan"] = int(fire[frac > 0].sum())
    # сколько огня потенциально «съедено»: компоненты огня, касающиеся NaN
    touch = 0; comps = 0
    for i in np.flatnonzero(frac > 0):
        lab, k = ndi.label(fire[i], structure=np.ones((3, 3), bool))
        comps += k
        touch += len(np.unique(lab[adj[i] & fire[i]]))
    N["a7b_fire_components_touching_nan"] = [int(touch), int(comps)]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(t))
    ax[0].bar(x, t["доля NaN"] * 100, color=C.C_BLUE); ax[0].set_xticks(x); ax[0].set_xticklabels(t["sensor_zenith, °"], rotation=45, ha="right")
    ax[0].set_xlabel("зенитный угол сенсора, °"); ax[0].set_ylabel("% пикселей с NaN в I4/I5 (valid = 1)"); ax[0].set_title("NaN появляются только при угле > 36°")
    ax[1].scatter(chip_senz, frac * 100, s=18, color=C.C_BLUE, alpha=0.7)
    ax[1].set_xlabel("медианный sensor_zenith чипа, °"); ax[1].set_ylabel("% NaN на чипе"); ax[1].set_title(f"По чипам: ρ Спирмена = {N['a7b_spearman_chip_senz_vs_nan']:.2f}")
    i = int(np.argmax(np.where(frac < 0.4, frac, 0)))
    ax[2].imshow(np.where(nan[i], np.nan, V[i, 3]), cmap="inferno"); ax[2].imshow(np.where(nan[i], 1.0, np.nan), cmap="Greys", vmin=0, vmax=2)
    yy, xx = np.nonzero(fire[i]); ax[2].scatter(xx, yy, s=10, color="#1baf7a", label=f"огонь ({yy.size} px)")
    ax[2].set_title(f"{meta.chip_id.iloc[i]}: полосы удаления bow-tie ({frac[i] * 100:.0f} % NaN)"); ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].grid(False)
    ax[2].legend(loc="lower right", fontsize=10, frameon=True)
    fig.suptitle("AF: NaN в I4/I5 — не облака и не край сцены, а удалённые строки bow-tie на краю полосы обзора", y=1.02)
    C.savefig(fig, "eda_af_07b_nan_bowtie.png")


# ----------------------------------------------------------------------------- A8 спутники
def a8_satellites(meta, V, A, M, valid):
    print("A8. спутники")
    n = len(M)
    bgm = valid & (M != 1)
    med = lambda x: np.array([np.nanmedian(x[i][bgm[i]]) if bgm[i].any() else np.nan for i in range(n)])
    i4m, i5m = med(V[:, 3]), med(V[:, 4])
    t2m = np.nanmedian(A[:, 2].reshape(n, -1), axis=1)   # t2m в файлах — в кельвинах (не °C, как в описании)
    df = pd.DataFrame({"satellite": meta["satellite"].to_numpy(), "night": meta["night"].to_numpy(), "I4": i4m, "I5": i5m,
                       "I4-I5": i4m - i5m, "I5-t2m": i5m - t2m, "I4-t2m": i4m - t2m, "month": meta["month"].to_numpy()})
    t = df.groupby(["night", "satellite"]).agg(чипов=("I4", "size"), I4_медиана=("I4", "median"), I5_медиана=("I5", "median"),
                                              I4_I5_медиана=("I4-I5", "median"), I4_I5_iqr=("I4-I5", lambda x: x.quantile(.75) - x.quantile(.25)),
                                              I5_минус_t2m=("I5-t2m", "median"), I5_минус_t2m_iqr=("I5-t2m", lambda x: x.quantile(.75) - x.quantile(.25)),
                                              I4_минус_t2m=("I4-t2m", "median")).reset_index()
    t["night"] = t["night"].map({False: "день", True: "ночь"})
    C.save_table(t, "af_a8_satellites.csv")
    from scipy.stats import mannwhitneyu
    pv = {}
    for nt, tag in ((False, "день"), (True, "ночь")):
        a = df[(df.night == nt) & (df.satellite == "SNPP")]["I5-t2m"].dropna(); b = df[(df.night == nt) & (df.satellite == "NOAA20")]["I5-t2m"].dropna()
        if len(a) > 5 and len(b) > 5:
            pv[tag] = float(mannwhitneyu(a, b).pvalue)
    N["a8_mw_pvalue_I5_minus_t2m_SNPP_vs_NOAA20"] = pv
    N["a8_bg_I4_chip_median_range"] = [float(np.nanmin(i4m)), float(np.nanmax(i4m))]
    N["a8_bg_I4_chip_median_std_day"] = float(np.nanstd(i4m[~df.night])); N["a8_bg_I4_chip_median_std_night"] = float(np.nanstd(i4m[df.night.to_numpy()]))
    N["a8_sat_median_I4_I5"] = {f"{r.night}/{r.satellite}": float(r.I4_I5_медиана) for r in t.itertuples()}

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sats = ["SNPP", "NOAA20", "NOAA21"]
    for a_, col, lab in zip(ax, ("I4-I5", "I5-t2m"), ("медиана фона I4 − I5 по чипу, K", "медиана фона I5 − t2m (ERA5) по чипу, K")):
        data, labels, colors = [], [], []
        for nt, tag in ((False, "день"), (True, "ночь")):
            for s in sats:
                x = df[(df.night == nt) & (df.satellite == s)][col].dropna()
                if len(x) >= 3:
                    data.append(x); labels.append(f"{s}\n{tag}\nn={len(x)}"); colors.append(C.C_ORANGE if not nt else C.C_VIOLET)
        bp = a_.boxplot(data, patch_artist=True, medianprops={"color": C.C_INK}, flierprops={"markersize": 3})
        a_.set_xticklabels(labels)
        for p, c_ in zip(bp["boxes"], colors):
            p.set_facecolor(c_); p.set_alpha(0.8)
        a_.set_ylabel(lab); a_.tick_params(axis="x", labelsize=10)
    ax[0].set_title("Спектральная разность фона"); ax[1].set_title("Фон относительно температуры воздуха")
    fig.suptitle("AF: различия между спутниками малы по сравнению с разбросом фона между чипами", y=1.02)
    C.savefig(fig, "eda_af_08_satellites.png")


# ----------------------------------------------------------------------------- A9 валидация
def a9_validation(meta, P, M):
    print("A9. валидация: случайный KFold против GroupKFold по тайлу")
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    folds = pd.read_csv(C.ROOT / "configs" / "folds_af.csv").set_index("chip_id").loc[meta["chip_id"]]
    t = meta.groupby("tile").size()
    N["a9_tiles"] = int(len(t)); N["a9_chips"] = int(len(meta)); N["a9_passes_per_tile_mean"] = float(t.mean())
    same_pass = meta.groupby("acq_datetime").size()
    N["a9_unique_passes"] = int(len(same_pass)); N["a9_chips_in_multi_tile_passes"] = int(same_pass[same_pass > 1].sum())
    ft = pd.DataFrame({"fold": folds["fold"].to_numpy(), "tile": meta["tile"].to_numpy(), "n_fire": meta["n_fire_px"].fillna(0).to_numpy(),
                       "val2024": (folds["temporal_split"] == "val").to_numpy()})
    tf = ft.groupby("fold").agg(чипов=("tile", "size"), тайлов=("tile", "nunique"), пикселей_огня=("n_fire", "sum"),
                                чипов_с_огнём=("n_fire", lambda x: int((x > 0).sum()))).reset_index()
    C.save_table(tf, "af_a9_folds.csv")
    N["a9_tile_leak_between_folds"] = int((ft.groupby("tile")["fold"].nunique() > 1).sum())
    N["a9_temporal_val_chips"] = int(ft.val2024.sum()); N["a9_temporal_val_fire_px"] = int(ft.n_fire[ft.val2024].sum())

    cand = P["cand"]
    chip = P["chip"][cand]
    y = P["fire"][cand].astype(int)
    fire_total_per_chip = (M == 1).reshape(len(M), -1).sum(1)
    fire_in_cand = np.bincount(chip, weights=y, minlength=len(meta))
    N["a9_candidates"] = int(cand.sum()); N["a9_cand_recall"] = float(y.sum() / fire_total_per_chip.sum())
    feat_sets = {
        "3 признака (I4, I4−I5, z(I4−I5) 11×11)": ["I4", "d45", "z_d45_w11_rob"],
        "3 признака + «паспорт места» (dem, t2m, rh2m, wind, landcover)": ["I4", "d45", "z_d45_w11_rob", "dem", "t2m", "rh2m", "wind_speed", "landcover"],
    }
    rs = np.random.RandomState(C.SEED)
    chip_kfold = np.empty(len(meta), int)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=C.SEED).split(np.arange(len(meta)))):
        chip_kfold[te] = k
    schemes = {"случайный KFold по чипам": chip_kfold, "GroupKFold по тайлу (folds_af.csv)": folds["fold"].to_numpy()}
    params = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=1.0, random_state=C.SEED, deterministic=True, force_col_wise=True, n_jobs=C.N_THREADS, verbose=-1)

    def f1_at(prob, mask_rows, thr):
        pred = prob[mask_rows] >= thr
        yy = y[mask_rows]
        tp = (pred & (yy == 1)).sum(); fp = (pred & (yy == 0)).sum()
        chips_here = np.unique(chip[mask_rows])
        fn = fire_total_per_chip[chips_here].sum() - tp      # включая огонь вне кандидатов
        return C.f1_from_counts(tp, fp, fn)

    rows = []
    for fs_name, cols in feat_sets.items():
        X = np.c_[tuple(P[c][cand] for c in cols)].astype(np.float32)
        for sc_name, fold_of_chip in schemes.items():
            prob = np.zeros(len(y))
            for k in range(5):
                te = fold_of_chip[chip] == k
                m = lgb.LGBMClassifier(**params).fit(X[~te], y[~te])
                prob[te] = m.predict_proba(X[te])[:, 1]
            allr = np.ones(len(y), bool)
            thr = np.arange(0.1, 0.91, 0.05)
            f = [f1_at(prob, allr, t_) for t_ in thr]
            rows.append({"признаки": fs_name, "схема": sc_name, "F1 OOF (порог 0.5)": f1_at(prob, allr, 0.5),
                         "F1 OOF (лучший порог)": max(f), "лучший порог": float(thr[int(np.argmax(f))])})
        tr = ~ft.val2024.to_numpy()[chip]
        m = lgb.LGBMClassifier(**params).fit(X[tr], y[tr])
        prob = np.zeros(len(y)); prob[~tr] = m.predict_proba(X[~tr])[:, 1]
        f = [f1_at(prob, ~tr, t_) for t_ in thr]
        rows.append({"признаки": fs_name, "схема": "временной сплит (val = 2024)", "F1 OOF (порог 0.5)": f1_at(prob, ~tr, 0.5),
                     "F1 OOF (лучший порог)": max(f), "лучший порог": float(thr[int(np.argmax(f))])})
    # правило без обучения — одинаково при любой схеме
    rule = (P["I4"][cand] > BASE_I4) & (P["d45"][cand] > BASE_D)
    tp = (rule & (y == 1)).sum(); fp = (rule & (y == 0)).sum()
    rows.append({"признаки": "правило I4>320 и I4−I5>15 (без обучения)", "схема": "любая", "F1 OOF (порог 0.5)": C.f1_from_counts(tp, fp, fire_total_per_chip.sum() - tp),
                 "F1 OOF (лучший порог)": np.nan, "лучший порог": np.nan})
    tv = C.save_table(pd.DataFrame(rows), "af_a9_kfold_vs_group.csv", floatfmt="%.4f")
    N["a9_results"] = [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in tv.to_dict("records")]

    fig, ax = plt.subplots(figsize=(11, 5))
    sub = tv[tv["схема"] != "любая"]
    sch = list(dict.fromkeys(sub["схема"]))
    for i, fs_name in enumerate(feat_sets):
        v = [sub[(sub["признаки"] == fs_name) & (sub["схема"] == s)]["F1 OOF (порог 0.5)"].iloc[0] for s in sch]
        b = ax.bar(np.arange(len(sch)) + (i - 0.5) * 0.38, v, 0.36, color=C.PALETTE[i], label=fs_name)
        ax.bar_label(b, fmt="%.3f", fontsize=11)
    ax.set_xticks(np.arange(len(sch))); ax.set_xticklabels([s.replace(" (", "\n(") for s in sch])
    ax.set_ylabel("F1_af (OOF, порог 0.5)"); ax.set_ylim(0, 1); ax.legend(loc="lower left", fontsize=10)
    ax.set_title("AF: оценка одной и той же модели при разных схемах валидации")
    C.savefig(fig, "eda_af_09_validation.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-cache", action="store_true", help="пересчитать eda/cache/af_pixels.npz")
    args = ap.parse_args()
    np.random.seed(C.SEED)
    C.setup_style(); C.ensure_dirs()
    t0 = time.time()
    meta, V, A, M, valid = load_all()
    P = build_pixel_table(meta, V, A, M, valid, use_cache=not args.no_cache)
    a1_imbalance(meta, M, valid)
    a2_signal(meta, V, A, M, valid)
    a2b_rule(V, M, valid)
    a3_context(P)
    a4_daynight(meta, V, M, valid, P)
    a5_technogenic(meta, V, A, M, valid)
    a6_landcover_meteo(meta, A, M, valid, P)
    a7_valid(meta, V, A, M, valid)
    a7b_nan_structure(meta, V, M)
    a8_satellites(meta, V, A, M, valid)
    a9_validation(meta, P, M)
    C.save_numbers(N, "af_numbers.json")
    print(f"готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
