"""Признаки BS-модуля: `compute_bs_features(chip) -> (F, 512, 512) float32`.

Всё векторизовано (numpy + cv2/scipy), без циклов по пикселям. Признаки разбиты на группы
(`FEATURE_GROUPS`) — группы используются в абляциях и позволяют считать только нужное
(UNet берёт подмножество `UNET_CHANNELS`, контекст ему не нужен).

Группы: spectral (отражения и индексы pre/post), diff (разности), scl (флаги SCL, расстояние до
облака), sar (Sentinel-1 после медианы 5×5), aux (рельеф, landcover), ctx (оконные статистики),
lcrel (dNBR относительно медианы dNBR гари для данного покрова — таблица из train).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

try:  # cv2 ставится вместе с albumentations; без него — scipy (медленнее, результат тот же)
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

S2 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
B2, B3, B4, B5, B6, B7, B8A, B11, B12, SCL = range(10)
LC_CODES = [10, 20, 30, 40, 50, 60, 80, 90]
SCL_CODES = {"nodata": (0, 1), "dark": (2,), "shadow": (3,), "water": (6,), "cloud_med": (8,),
             "cloud_high": (9,), "cirrus": (10,), "snow": (11,)}
EPS = 1e-6
DIST_CAP = 64.0  # px, насыщение расстояния до облака
# Пороги степени по медианному dNBR 3×3 для основных покровов — оценены по train (EDA, reports/bs_module.md):
# внутри контура гари правило с этими порогами совпадает с эталоном на 97 %. Остальные покровы — шкала USGS.
RULE_THR_DEFAULT = (0.10, 0.27, 0.44)
RULE_THR = {10: (0.125, 0.27, 0.575), 30: (0.065, 0.205, 0.385), 40: (0.07, 0.18, 0.38), 90: (0.08, 0.33, 0.61)}

_INDICES = ["NBR", "NDVI", "NDWI", "NBR2", "MIRBI", "SAVI", "bright"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "spectral": ([f"pre_{b}" for b in S2] + [f"post_{b}" for b in S2]
                 + [f"pre_{i}" for i in _INDICES] + [f"post_{i}" for i in _INDICES] + ["post_BAI"]),
    "diff": ["dNBR", "RdNBR", "RBR", "dNDVI", "dNBR2", "dB12", "dB8A", "dBright", "dRedEdge", "dMIRBI", "dNDWI"],
    "scl": ([f"scl_pre_{k}" for k in SCL_CODES] + [f"scl_post_{k}" for k in SCL_CODES]
            + ["scl_bad_any", "dist_cloud", "invalid"]),
    "sar": ["pre_VV", "pre_VH", "post_VV", "post_VH", "dVV", "dVH", "pre_VHVV", "post_VHVV", "dVHVV", "sar_nodata"],
    "aux": ["dem_norm", "slope"],
    "lc": ["landcover"] + [f"lc_{c}" for c in LC_CODES],
    "ctx": ["dNBR_m3", "dNBR_m7", "dNBR_m15", "dNDVI_m3", "dNDVI_m7", "dNDVI_m15", "dNBR_s7", "dNBR_med3",
            "dNBR_med5"],
    "lcrel": ["dNBR_rel"],
    "rule": ["rule_sev"],
    "geom": ["dist_center", "coord_x", "coord_y"],
    # H4 (ночные эксперименты): медианы 3×3 прочих разностных индексов; в финальные модели НЕ входят (опция стека)
    "med2": ["dMIRBI_med3", "dNBR2_med3", "RBR_med3"],
    "event": ["dist_strong_cc", "dist_strong_centroid", "dist_dnbr_centroid", "cc_rank", "cc_area_rel",
              "cc_area_log"],
}
FEATURE_NAMES: list[str] = [n for names in FEATURE_GROUPS.values() for n in names]
GROUP_OF = {n: g for g, names in FEATURE_GROUPS.items() for n in names}

# входные каналы UNet (контекст сеть строит сама)
UNET_CHANNELS: list[str] = (
    [f"pre_{b}" for b in S2] + [f"post_{b}" for b in S2]
    + ["dNBR", "RdNBR", "dNDVI", "dNBR2"]
    + [f"scl_pre_{k}" for k in ("shadow", "water", "cloud_med", "cloud_high", "cirrus")]
    + [f"scl_post_{k}" for k in ("dark", "shadow", "water", "cloud_med", "cloud_high", "cirrus")]
    + ["invalid", "pre_VV", "pre_VH", "post_VV", "post_VH", "sar_nodata", "dem_norm", "slope"]
    + [f"lc_{c}" for c in (10, 30, 40, 50, 80, 90)]
)

# v2: + медианный dNBR 3×3 — именно по нему порогуется эталон (см. reports/bs_module.md)
# v3: + результат пиксельного правила (степень 0..3) — сети остаётся выбрать, какие пятна относятся к пожару
CHANNEL_SETS: dict[str, list[str]] = {"v1": UNET_CHANNELS, "v2": UNET_CHANNELS + ["dNBR_med3"],
                                      "v3": UNET_CHANNELS + ["dNBR_med3", "rule_sev"],
                                      "v4": UNET_CHANNELS + ["dNBR_med3", "rule_sev", "dist_center"],
                                      "v5": UNET_CHANNELS + ["dNBR_med3", "rule_sev", "dist_center", "coord_x",
                                                             "coord_y", "dist_strong_cc", "cc_area_rel"]}

DEFAULT_LC_MEDIANS = {c: 0.25 for c in LC_CODES}


def groups_for(names: list[str]) -> list[str]:
    """Минимальный набор групп признаков, покрывающий `names` (в порядке FEATURE_GROUPS)."""
    need = {GROUP_OF[n] for n in names}
    return [g for g in FEATURE_GROUPS if g in need]


def feature_names(groups: list[str] | None = None) -> list[str]:
    """Имена признаков в порядке каналов результата `compute_bs_features(chip, groups=groups)`."""
    if groups is None:
        return list(FEATURE_NAMES)
    return [n for g, names in FEATURE_GROUPS.items() if g in groups for n in names]


def load_lc_medians(path: str | Path | None) -> dict[int, float]:
    """Таблица «landcover → медиана dNBR внутри гари» (train). Нет файла — константа."""
    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return {int(k): float(v) for k, v in json.load(f).items()}
    return dict(DEFAULT_LC_MEDIANS)


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Нормализованная разность (a−b)/(a+b); знаменатель ≤0 → 0."""
    den = a + b
    return np.divide(a - b, den, out=np.zeros_like(den), where=den > EPS)


def _box_mean(x: np.ndarray, k: int) -> np.ndarray:
    if cv2 is not None:
        return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)
    return ndi.uniform_filter(x, k, mode="reflect")


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    if cv2 is not None:
        return cv2.GaussianBlur(np.ascontiguousarray(x, dtype=np.float32), (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
    return ndi.gaussian_filter(x.astype(np.float32), sigma, mode="constant")


def _median(x: np.ndarray, k: int) -> np.ndarray:
    """Медиана k×k (k = 3 или 5: cv2.medianBlur поддерживает float32 только для них)."""
    if cv2 is not None:
        return cv2.medianBlur(np.ascontiguousarray(x, dtype=np.float32), k)
    return ndi.median_filter(x, k, mode="nearest")


def _median5(x: np.ndarray) -> np.ndarray:
    return _median(x, 5)


def _dist_to(mask: np.ndarray) -> np.ndarray:
    """Расстояние (px) до ближайшего True, насыщение DIST_CAP; нет True → DIST_CAP."""
    if not mask.any():
        return np.full(mask.shape, DIST_CAP, np.float32)
    if cv2 is not None:
        d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    else:
        d = ndi.distance_transform_edt(~mask)
    return np.minimum(d, DIST_CAP).astype(np.float32)


def _indices(r: np.ndarray) -> dict[str, np.ndarray]:
    """r — отражения (9, H, W) float32."""
    return {
        "NBR": _nd(r[B8A], r[B12]),
        "NDVI": _nd(r[B8A], r[B4]),
        "NDWI": _nd(r[B3], r[B8A]),
        "NBR2": _nd(r[B11], r[B12]),
        "MIRBI": 10.0 * r[B12] - 9.8 * r[B11] + 2.0,
        "SAVI": 1.5 * (r[B8A] - r[B4]) / (r[B8A] + r[B4] + 0.5),
        "bright": (r[B2] + r[B3] + r[B4]) / 3.0,
    }


def rule_severity(dnbr_med3: np.ndarray, lc: np.ndarray, scl_pre: np.ndarray, scl_post: np.ndarray) -> np.ndarray:
    """Пиксельное правило эталона: пороги по медианному dNBR (свои для покрова) + маска SCL → степень 0..3."""
    sev = np.digitize(dnbr_med3, RULE_THR_DEFAULT).astype(np.float32)
    for code, thr in RULE_THR.items():
        sel = lc == code
        if sel.any():
            sev[sel] = np.digitize(dnbr_med3[sel], thr)
    sev[np.isin(scl_pre, (0, 3, 6, 11)) | np.isin(scl_post, (0, 3, 9, 11))] = 0.0
    return sev


def event_features(rule_sev: np.ndarray, dnbr_med3: np.ndarray) -> dict[str, np.ndarray]:
    """Локализация события по содержимому чипа (эталон размечает только «свой» пожар в буфере ~4 км):
    расстояния до крупнейшей связной компоненты «сильного dNBR» (правило ≥ 2-й степени) и до её центроида,
    до dNBR-взвешенного центроида; ранг/относительная площадь компоненты правила, которой принадлежит пиксель."""
    h, w = rule_sev.shape
    yy, xx = np.ogrid[:h, :w]
    norm = 256.0
    out: dict[str, np.ndarray] = {}
    strong = rule_sev >= 2
    lab, n = ndi.label(strong, structure=np.ones((3, 3), bool))
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        sizes[0] = 0
        big = lab == int(sizes.argmax())
        out["dist_strong_cc"] = _dist_full(big) / norm
        cy, cx = ndi.center_of_mass(big)
        out["dist_strong_centroid"] = (np.hypot(yy - cy, xx - cx) / norm).astype(np.float32)
    else:
        out["dist_strong_cc"] = np.full((h, w), 2.0, np.float32)
        out["dist_strong_centroid"] = np.full((h, w), 2.0, np.float32)
    wgt = np.clip(dnbr_med3, 0.0, 1.0) * (rule_sev > 0)
    tot = float(wgt.sum())
    if tot > 0:
        cy = float((wgt * yy).sum() / tot)
        cx = float((wgt * xx).sum() / tot)
        out["dist_dnbr_centroid"] = (np.hypot(yy - cy, xx - cx) / norm).astype(np.float32)
    else:
        out["dist_dnbr_centroid"] = np.full((h, w), 2.0, np.float32)
    lab1, n1 = ndi.label(rule_sev > 0, structure=np.ones((3, 3), bool))
    if n1:
        sizes = np.bincount(lab1.ravel(), minlength=n1 + 1).astype(np.float32)
        sizes[0] = 0
        order = np.argsort(-sizes[1:], kind="stable")
        rank = np.zeros(n1 + 1, np.float32)
        rank[1 + order] = np.minimum(np.arange(1, n1 + 1), 50)
        out["cc_rank"] = rank[lab1]
        out["cc_area_rel"] = (sizes / sizes.max())[lab1]
        out["cc_area_log"] = np.log1p(sizes)[lab1]
    else:
        for k in ("cc_rank", "cc_area_rel", "cc_area_log"):
            out[k] = np.zeros((h, w), np.float32)
    return out


def _dist_full(mask: np.ndarray) -> np.ndarray:
    """Евклидово расстояние (px) до ближайшего True без насыщения."""
    if cv2 is not None:
        return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    return ndi.distance_transform_edt(~mask).astype(np.float32)


def compute_bs_features(chip: dict, lc_medians: dict[int, float] | None = None,
                        groups: list[str] | None = None) -> np.ndarray:
    """chip — результат `read_bs_chip`. Возвращает (F, H, W) float32 без NaN/inf;
    порядок каналов — `feature_names(groups)`. Пиксели без данных → 0 + флаги `invalid`, `sar_nodata`."""
    want = set(FEATURE_GROUPS) if groups is None else set(groups)
    f: dict[str, np.ndarray] = {}

    pre = chip["s2_pre"][:9].astype(np.float32) / 10000.0
    post = chip["s2_post"][:9].astype(np.float32) / 10000.0
    scl_pre, scl_post = chip["s2_pre"][SCL], chip["s2_post"][SCL]
    invalid = (scl_pre == 0) | (scl_post == 0)
    ipre, ipost = _indices(pre), _indices(post)
    dnbr = ipre["NBR"] - ipost["NBR"]
    dndvi = ipre["NDVI"] - ipost["NDVI"]

    if "spectral" in want:
        for i, b in enumerate(S2):
            f[f"pre_{b}"], f[f"post_{b}"] = pre[i], post[i]
        for k in _INDICES:
            f[f"pre_{k}"], f[f"post_{k}"] = ipre[k], ipost[k]
        bai = 1.0 / ((0.1 - post[B4]) ** 2 + (0.06 - post[B8A]) ** 2 + 1e-4)
        f["post_BAI"] = np.log1p(bai)
    if "diff" in want:
        f["dNBR"] = dnbr
        f["RdNBR"] = dnbr / np.sqrt(np.abs(ipre["NBR"]) + 0.001)
        f["RBR"] = dnbr / (ipre["NBR"] + 1.001)
        f["dNDVI"] = dndvi
        f["dNBR2"] = ipre["NBR2"] - ipost["NBR2"]
        f["dB12"] = post[B12] - pre[B12]
        f["dB8A"] = post[B8A] - pre[B8A]
        f["dBright"] = ipost["bright"] - ipre["bright"]
        f["dRedEdge"] = (post[B7] - post[B5]) - (pre[B7] - pre[B5])
        f["dMIRBI"] = ipost["MIRBI"] - ipre["MIRBI"]
        f["dNDWI"] = ipost["NDWI"] - ipre["NDWI"]
    if "scl" in want:
        for tag, scl in (("pre", scl_pre), ("post", scl_post)):
            for k, codes in SCL_CODES.items():
                f[f"scl_{tag}_{k}"] = np.isin(scl, codes)
        cloud = np.isin(scl_pre, (3, 8, 9, 10)) | np.isin(scl_post, (3, 8, 9, 10))
        f["scl_bad_any"] = cloud
        f["dist_cloud"] = _dist_to(cloud) / DIST_CAP
        f["invalid"] = invalid
    if "sar" in want:
        s1 = {}
        for tag in ("pre", "post"):
            raw = chip[f"s1_{tag}"].astype(np.float32) / 100.0
            s1[tag] = (_median5(raw[0]), _median5(raw[1]))
        nod = ((chip["s1_pre"][0] == 0) & (chip["s1_pre"][1] == 0)) | \
              ((chip["s1_post"][0] == 0) & (chip["s1_post"][1] == 0))
        f["pre_VV"], f["pre_VH"] = s1["pre"]
        f["post_VV"], f["post_VH"] = s1["post"]
        f["dVV"] = f["post_VV"] - f["pre_VV"]
        f["dVH"] = f["post_VH"] - f["pre_VH"]
        f["pre_VHVV"] = f["pre_VH"] - f["pre_VV"]
        f["post_VHVV"] = f["post_VH"] - f["post_VV"]
        f["dVHVV"] = f["post_VHVV"] - f["pre_VHVV"]
        f["sar_nodata"] = nod
    lc = chip["aux"][2]
    if "aux" in want:
        dem = chip["aux"][0].astype(np.float32)
        f["dem_norm"] = (dem - dem.mean()) / (dem.std() + 1.0)
        f["slope"] = chip["aux"][1].astype(np.float32)
    if "lc" in want:
        f["landcover"] = lc.astype(np.float32)
        for c in LC_CODES:
            f[f"lc_{c}"] = lc == c
    if "ctx" in want:
        for k in (3, 7, 15):
            f[f"dNBR_m{k}"] = _box_mean(dnbr, k)
            f[f"dNDVI_m{k}"] = _box_mean(dndvi, k)
        var = _box_mean(dnbr * dnbr, 7) - f["dNBR_m7"] ** 2
        f["dNBR_s7"] = np.sqrt(np.maximum(var, 0.0))
        # EDA: степень в эталоне на 97 % воспроизводится порогами по МЕДИАННОМУ dNBR 3×3 (+ landcover)
        f["dNBR_med3"] = _median(dnbr, 3)
        f["dNBR_med5"] = _median(dnbr, 5)
    if "lcrel" in want:
        med = lc_medians or DEFAULT_LC_MEDIANS
        lut = np.full(256, 0.25, np.float32)
        for c, v in med.items():
            if 0 <= int(c) < 256:
                lut[int(c)] = max(float(v), 0.02)
        f["dNBR_rel"] = dnbr / lut[np.clip(lc, 0, 255)]

    if "rule" in want:
        f["rule_sev"] = rule_severity(_median(dnbr, 3), lc, scl_pre, scl_post)

    if "med2" in want:
        f["dMIRBI_med3"] = _median(ipost["MIRBI"] - ipre["MIRBI"], 3)
        f["dNBR2_med3"] = _median(ipre["NBR2"] - ipost["NBR2"], 3)
        f["RBR_med3"] = _median(dnbr / (ipre["NBR"] + 1.001), 3)
    if "geom" in want:
        # эталон обрезан буфером события (~4 км); буфер не центрирован по чипу, но слабый радиальный приор есть
        h, w = dnbr.shape
        yy, xx = np.ogrid[:h, :w]
        f["dist_center"] = (np.hypot(yy - (h - 1) / 2.0, xx - (w - 1) / 2.0) / np.hypot(h / 2.0, w / 2.0)).astype(np.float32)
        f["coord_x"] = np.broadcast_to((xx / (w - 1.0) * 2.0 - 1.0).astype(np.float32), (h, w))
        f["coord_y"] = np.broadcast_to((yy / (h - 1.0) * 2.0 - 1.0).astype(np.float32), (h, w))
    if "event" in want:
        f.update(event_features(rule_severity(_median(dnbr, 3), lc, scl_pre, scl_post), _median(dnbr, 3)))

    names = feature_names(None if groups is None else list(want))
    out = np.empty((len(names),) + dnbr.shape, np.float32)
    for i, n in enumerate(names):
        out[i] = f[n]
    # NaN/inf здесь невозможны: входы целочисленные, все деления защищены (см. _nd, +0.001, +0.5, lut ≥ 0.02) —
    # полный проход nan_to_num по ~100 МБ на чип стоил ~15 % времени признаков. Страховка — только для float-входов.
    if any(chip[k].dtype.kind == "f" for k in ("s2_pre", "s2_post", "s1_pre", "s1_post", "aux") if k in chip):
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # пиксели без данных S2: физические признаки → 0 (флаги SCL/aux/lc остаются)
    if invalid.any():
        phys = [i for i, n in enumerate(names) if GROUP_OF[n] in ("spectral", "diff", "ctx", "lcrel")]
        for i in phys:
            out[i][invalid] = 0.0
    return out


def select_channels(feats: np.ndarray, names: list[str], wanted: list[str]) -> np.ndarray:
    idx = [names.index(n) for n in wanted]
    return feats[idx]
