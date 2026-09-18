"""Признаки AF на пиксель (контекстный детектор по мотивам VNP14 + признаки ложных срабатываний).

Две точки входа с ОДНИМ кодом расчёта (значения совпадают побитово):
  * `compute_af_features(chip) -> (F, H, W) float32` — полная карта (тесты, визуализация);
  * `af_feature_rows(chip, pixel_mask) -> (n, F) float32` — только для выбранных пикселей
    (обучение и инференс: модель применяется к пикселям-кандидатам, см. `candidate_mask`).

Оконные суммы считаются через интегральные изображения (одна cumsum на канал, любое окно —
4 обращения), с нормировкой на число валидных пикселей окна: NaN/невалидные не портят фон.
Используется только содержимое чипа (viirs + aux) — никаких дат/координат (правила, CLAUDE.md §7).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

WINDOWS = (5, 7, 11, 15)
MEDIAN_WINDOWS = (7, 15)
ROBUST_MIN_BG = 8        # минимум «не горячих» соседей, иначе робастный фон = обычному
HOT_DI = 10.0            # «горячий сосед»: I4−I5 > 10 K
EPS = 1e-3
DIST_CAP = 64.0          # расстояния (в пикселях, chessboard) обрезаются: дальше — всё равно

# Ослабленные пороги кандидатов (шаг 2, AF-17): запас от минимума огня train ≥ 5 K по I4 (день 319.25 K, ночь 295.04 K)
# и ≥ 5 K по I4−I5 ночью (минимум 10.04 K); днём I4−I5: 1-й перцентиль огня 25 K, хвост добирает «дорастание».
CAND = {"day_i4": 314.0, "day_di": 17.0, "night_i4": 290.0, "night_di": 5.0, "grow_di": 25.0}

LC_CODES = {"built": 50, "water": 80, "crop": 40, "bare": 60}


def _names() -> list[str]:
    n = ["I1", "I2", "I3", "I4", "I5", "di45", "ratio45", "solar_zenith", "sensor_zenith", "is_night",
         "landcover", "lc_built", "lc_water", "lc_crop", "lc_bare", "dem", "t2m", "rh2m", "wind_speed"]
    for w in WINDOWS:
        for ch in ("i4", "di"):
            n += [f"{ch}_mean{w}", f"{ch}_std{w}", f"{ch}_z{w}", f"{ch}_rmean{w}", f"{ch}_rz{w}", f"{ch}_dmax{w}"]
        n += [f"hotfrac{w}", f"validfrac{w}"]
    for w in MEDIAN_WINDOWS:
        n += [f"i4_dmed{w}", f"di_dmed{w}"]
    n += ["i4_dmean3", "di_dmean3", "i5_d7", "i5_z7", "i5_d15", "i5_z15", "i4_dmax3", "di_dmax3",
          "i4_m_p50", "i4_m_p90", "i4_m_p99", "di_m_p50", "di_m_p99", "chip_i4_p50", "chip_i4_std", "chip_cand_frac",
          "g_i3", "g_i3_i2", "g_vis_sum", "g_ndvi", "g_i3_anom7",
          "comp_size", "comp_max_i4", "comp_max_di", "n_cand3",
          "dist_built", "dist_water", "dist_invalid", "frac_built7", "frac_water7", "was_nan"]
    return n


FEATURE_NAMES: list[str] = _names()
feature_names = FEATURE_NAMES  # псевдоним из ТЗ задачи
CATEGORICAL = ["landcover"]

CONTEXT_FEATURES = [f for f in FEATURE_NAMES if any(f.startswith(p) for p in ("i4_mean", "i4_std", "i4_z", "i4_rmean",
                    "i4_rz", "i4_dmax", "di_mean", "di_std", "di_z", "di_rmean", "di_rz", "di_dmax", "hotfrac",
                    "validfrac", "i4_dmed", "di_dmed", "i4_m_p", "di_m_p", "chip_", "i4_dmean3", "di_dmean3", "i5_d", "i5_z"))]
AUX_FEATURES = ["landcover", "lc_built", "lc_water", "lc_crop", "lc_bare", "dem", "t2m", "rh2m", "wind_speed",
                "dist_built", "dist_water", "frac_built7", "frac_water7"]
GLINT_FEATURES = ["I1", "I2", "I3", "g_i3", "g_i3_i2", "g_vis_sum", "g_ndvi", "g_i3_anom7"]


def chip_valid(chip: dict) -> np.ndarray:
    v = chip["viirs"]
    if "valid" in chip and chip["valid"] is not None:
        return chip["valid"]
    return (v[7] > 0.5) & np.isfinite(v[3]) & np.isfinite(v[4])


def night_mask(viirs: np.ndarray) -> np.ndarray:
    """is_night по пикселям (solar_zenith > 90); NaN зенита → по медиане чипа."""
    sz = viirs[5]
    fin = np.isfinite(sz)
    med = float(np.median(sz[fin])) if fin.any() else 0.0
    return np.where(fin, sz, med) > 90.0


def candidate_mask(viirs: np.ndarray, valid: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Пиксели-кандидаты: ослабленное правило день/ночь + «дорастание» вокруг сильных кандидатов.

    Днём внутри крупных очагов I5 тоже растёт (до 340 K) и I4−I5 падает ниже порога — такие пиксели
    добираются как соседи (3×3) сильного кандидата (I4−I5 > grow_di) при I4 выше дневного порога.
    """
    p = CAND if params is None else params
    i4, i5 = viirs[3], viirs[4]
    night = night_mask(viirs)
    with np.errstate(invalid="ignore"):
        di = i4 - i5
        hot4 = np.where(night, i4 > p["night_i4"], i4 > p["day_i4"]) & valid
        base = hot4 & np.where(night, di > p["night_di"], di > p["day_di"])
        strong = base & (di > p["grow_di"])
    return base | (hot4 & ndi.binary_dilation(strong, structure=np.ones((3, 3), bool)))


def _fill(a: np.ndarray, bad: np.ndarray | None = None) -> np.ndarray:
    """NaN (и пиксели `bad`) → медиана валидной части чипа (0, если валидных нет)."""
    nan = ~np.isfinite(a)
    if bad is not None:
        nan = nan | bad
    if not nan.any():
        return a.astype(np.float32, copy=False)
    good = a[~nan]
    med = np.float32(np.median(good)) if good.size else np.float32(0)
    return np.where(nan, med, a).astype(np.float32)


class _Box:
    """Суммы по окну (2r+1)² в точках (ys, xs) через интегральное изображение; вне чипа — нули."""

    def __init__(self, channels: np.ndarray, rmax: int, origin: tuple[int, int] = (0, 0)):
        """`channels` может быть вырезкой чипа с левым верхним углом `origin` (координаты sums — по чипу);
        вырезка обязана покрывать все окна запрашиваемых точек в пределах чипа."""
        c, h, w = channels.shape
        self.p = rmax
        self.oy, self.ox = origin
        integ = np.zeros((c, h + 2 * rmax + 1, w + 2 * rmax + 1), np.float64)
        integ[:, rmax + 1:rmax + 1 + h, rmax + 1:rmax + 1 + w] = channels
        np.cumsum(integ, axis=1, out=integ)
        np.cumsum(integ, axis=2, out=integ)
        self.integ = integ

    def sums(self, r: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        i, p = self.integ, self.p
        ys, xs = ys - self.oy, xs - self.ox
        y0, y1, x0, x1 = ys + p - r, ys + p + r + 1, xs + p - r, xs + p + r + 1
        return i[:, y1, x1] - i[:, y0, x1] - i[:, y1, x0] + i[:, y0, x0]


def _window_order_stats(a: np.ndarray, ys: np.ndarray, xs: np.ndarray, chunk: int = 4096) -> dict[str, np.ndarray]:
    """Максимумы (окна WINDOWS и кольцо 3×3 без центра) и медианы (MEDIAN_WINDOWS) только в точках (ys, xs):
    один сбор патчей 15×15 вместо десятка полнокадровых фильтров (кандидатов обычно десятки–сотни на чип)."""
    wmax = max(WINDOWS)
    r0 = wmax // 2
    view = np.lib.stride_tricks.sliding_window_view(np.pad(a, r0, mode="edge"), (wmax, wmax))
    n = len(ys)
    res = {f"max{w}": np.empty(n, np.float32) for w in WINDOWS}
    res.update({f"med{w}": np.empty(n, np.float32) for w in MEDIAN_WINDOWS})
    res["max3ring"] = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        patches = view[ys[sl], xs[sl]]                                  # копия (m, 15, 15)
        for w in set(WINDOWS) | set(MEDIAN_WINDOWS):
            r = w // 2
            sub = patches[:, r0 - r:r0 + r + 1, r0 - r:r0 + r + 1].reshape(len(patches), -1)
            if w in WINDOWS:
                res[f"max{w}"][sl] = sub.max(axis=1)
            if w in MEDIAN_WINDOWS:
                res[f"med{w}"][sl] = np.median(sub, axis=1)
        ring = patches[:, r0 - 1:r0 + 2, r0 - 1:r0 + 2].reshape(len(patches), 9).copy()
        ring[:, 4] = -np.inf
        res["max3ring"][sl] = ring.max(axis=1)
    return res


def _cdt(target: np.ndarray) -> np.ndarray:
    """Chessboard-расстояние до ближайшего пикселя `target` (нет таких → DIST_CAP)."""
    if not target.any():
        return np.full(target.shape, DIST_CAP, np.float32)
    d = ndi.distance_transform_cdt(~target, metric="chessboard")
    return np.minimum(d, DIST_CAP).astype(np.float32)


def af_feature_rows(chip: dict, pixel_mask: np.ndarray | None = None) -> np.ndarray:
    """Признаки (n, F) float32 для пикселей `pixel_mask` (row-major порядок); None → все пиксели."""
    viirs, aux = chip["viirs"], chip["aux"]
    h, w_ = viirs.shape[1:]
    valid = chip_valid(chip)
    if pixel_mask is None:
        pixel_mask = np.ones((h, w_), bool)
    ys, xs = np.nonzero(pixel_mask)
    out: dict[str, np.ndarray] = {}

    def put(name: str, arr2d: np.ndarray) -> None:
        out[name] = arr2d[ys, xs]

    night = night_mask(viirs)
    day = ~night
    inval = ~valid
    i1, i2, i3 = _fill(viirs[0]), _fill(viirs[1]), _fill(viirs[2])
    i4, i5 = _fill(viirs[3], inval), _fill(viirs[4], inval)
    di = i4 - i5
    for name, a in (("I1", i1), ("I2", i2), ("I3", i3), ("I4", i4), ("I5", i5), ("di45", di)):
        put(name, a)
    put("ratio45", i4 / np.maximum(i5, 1.0))
    put("solar_zenith", _fill(viirs[5]))
    put("sensor_zenith", _fill(viirs[6]))
    put("is_night", night.astype(np.float32))
    lc = np.where(np.isfinite(aux[0]), aux[0], 0).astype(np.float32)
    put("landcover", lc)
    for k, code in LC_CODES.items():
        put(f"lc_{k}", (lc == code).astype(np.float32))
    for j, name in enumerate(("dem", "t2m", "rh2m", "wind_speed"), start=1):
        put(name, _fill(aux[j]))

    # --- контекст: интегральные изображения по центрированным каналам (точность сумм квадратов)
    c4, cd = float(np.median(i4)), float(np.median(di))
    x4, xd = (i4 - c4).astype(np.float64), (di - cd).astype(np.float64)
    x5 = (i5 - float(np.median(i5))).astype(np.float64)
    vf = valid.astype(np.float64)
    cand = candidate_mask(viirs, valid)
    # оконные суммы нужны только вокруг запрошенных точек → интегральные изображения по вырезке (bbox + rmax)
    rmax = max(WINDOWS) // 2
    if len(ys):
        cy0, cy1 = max(int(ys.min()) - rmax, 0), min(int(ys.max()) + rmax + 1, h)
        cx0, cx1 = max(int(xs.min()) - rmax, 0), min(int(xs.max()) + rmax + 1, w_)
    else:
        cy0, cy1, cx0, cx1 = 0, 1, 0, 1
    crop = (slice(cy0, cy1), slice(cx0, cx1))
    a4, ad, a5, av = x4[crop], xd[crop], x5[crop], vf[crop]
    bg = np.zeros((h, w_), np.float64)
    bg[crop] = valid[crop] & ~cand[crop]                  # фон без потенциального огня (как в VNP14)
    hot = np.zeros((h, w_), np.float64)
    hot[crop] = valid[crop] & (di[crop] > HOT_DI)
    ab = bg[crop]
    i3d = np.where(day, i3, 0).astype(np.float64)
    lcc = lc[crop]
    box = _Box(np.stack([av, a4 * av, a4 * a4 * av, ad * av, ad * ad * av,
                         ab, a4 * ab, a4 * a4 * ab, ad * ab, ad * ad * ab,
                         hot[crop], i3d[crop] * av, lcc == 50, lcc == 80, a5 * av, a5 * a5 * av]), rmax, (cy0, cx0))
    p4, pd_, pv, pb, ph = x4[ys, xs], xd[ys, xs], vf[ys, xs], bg[ys, xs], hot[ys, xs]
    p5 = x5[ys, xs]

    c_i4, c_di = i4[ys, xs], di[ys, xs]
    os4, osd = _window_order_stats(i4, ys, xs), _window_order_stats(di, ys, xs)

    def stats(n, s1, s2):
        n_ = np.maximum(n, 1.0)
        mean = s1 / n_
        return mean, np.sqrt(np.maximum(s2 / n_ - mean * mean, 0.0))

    s = box.sums(1, ys, xs)
    n3 = np.maximum(s[0] - pv, 1.0)
    out["i4_dmean3"] = p4 - (s[1] - p4 * pv) / n3
    out["di_dmean3"] = pd_ - (s[3] - pd_ * pv) / n3
    for w in WINDOWS:
        s = box.sums(w // 2, ys, xs)
        n = s[0] - pv                                             # окно без центра
        if w in (7, 15):
            mean5, std5 = stats(n, s[14] - p5 * pv, s[15] - p5 * p5 * pv)
            out[f"i5_d{w}"] = p5 - mean5
            out[f"i5_z{w}"] = (p5 - mean5) / (std5 + EPS)
        nb = s[5] - pb
        use_r = nb >= ROBUST_MIN_BG
        for ch, px, k, center in (("i4", p4, 1, c4), ("di", pd_, 3, cd)):
            mean, std = stats(n, s[k] - px * pv, s[k + 1] - px * px * pv)
            rmean, rstd = stats(nb, s[k + 5] - px * pb, s[k + 6] - px * px * pb)
            rmean, rstd = np.where(use_r, rmean, mean), np.where(use_r, rstd, std)
            out[f"{ch}_mean{w}"] = mean + center
            out[f"{ch}_std{w}"] = std
            out[f"{ch}_z{w}"] = (px - mean) / (std + EPS)
            out[f"{ch}_rmean{w}"] = rmean + center
            out[f"{ch}_rz{w}"] = (px - rmean) / (rstd + EPS)
        out[f"i4_dmax{w}"] = c_i4 - os4[f"max{w}"]
        out[f"di_dmax{w}"] = c_di - osd[f"max{w}"]
        out[f"hotfrac{w}"] = (s[10] - ph) / np.maximum(n, 1.0)
        out[f"validfrac{w}"] = s[0] / float(w * w)
        if w == 7:
            out["g_i3_anom7"] = (i3d[ys, xs] - s[11] / np.maximum(s[0], 1.0)) * day[ys, xs]
            out["frac_built7"] = s[12] / 49.0
            out["frac_water7"] = s[13] / 49.0
    for w in MEDIAN_WINDOWS:
        out[f"i4_dmed{w}"] = c_i4 - os4[f"med{w}"]
        out[f"di_dmed{w}"] = c_di - osd[f"med{w}"]
    out["i4_dmax3"] = c_i4 - os4["max3ring"]
    out["di_dmax3"] = c_di - osd["max3ring"]

    # --- нормировка на чип
    v4, vd = (i4[valid], di[valid]) if valid.any() else (i4.ravel(), di.ravel())
    q4 = np.percentile(v4, [50, 90, 99])
    qd = np.percentile(vd, [50, 99])
    for name, q in (("i4_m_p50", q4[0]), ("i4_m_p90", q4[1]), ("i4_m_p99", q4[2])):
        out[name] = i4[ys, xs] - np.float32(q)
    out["di_m_p50"] = di[ys, xs] - np.float32(qd[0])
    out["di_m_p99"] = di[ys, xs] - np.float32(qd[1])
    const = np.ones(len(ys), np.float32)
    out["chip_i4_p50"] = const * np.float32(q4[0])
    out["chip_i4_std"] = const * np.float32(v4.std())
    out["chip_cand_frac"] = const * np.float32(cand.mean())

    # --- блик (только днём; ночью I1–I3 = NaN → 0)
    dayf = day.astype(np.float32)
    put("g_i3", i3 * dayf)
    put("g_i3_i2", i3 / np.maximum(i2, 0.01) * dayf)
    put("g_vis_sum", (i1 + i2 + i3) * dayf)
    put("g_ndvi", (i2 - i1) / np.maximum(i2 + i1, 0.01) * dayf)

    # --- морфология кандидатов и расстояния
    labels, n_comp = ndi.label(cand, structure=np.ones((3, 3), bool))
    size = np.bincount(labels.ravel(), minlength=n_comp + 1).astype(np.float32)
    size[0] = 0
    lab = labels[ys, xs]
    out["comp_size"] = size[lab]
    if n_comp:
        lab_c = labels[cand]
        m4, md = np.full(n_comp + 1, -np.inf, np.float32), np.full(n_comp + 1, -np.inf, np.float32)
        np.maximum.at(m4, lab_c, i4[cand])
        np.maximum.at(md, lab_c, di[cand])
        out["comp_max_i4"] = np.where(lab > 0, m4[lab], i4[ys, xs])
        out["comp_max_di"] = np.where(lab > 0, md[lab], di[ys, xs])
    else:
        out["comp_max_i4"], out["comp_max_di"] = i4[ys, xs], di[ys, xs]
    put("n_cand3", ndi.uniform_filter(cand.astype(np.float32), size=3, mode="constant") * 9.0 - cand)
    put("dist_built", _cdt(lc == 50))
    put("dist_water", _cdt(lc == 80))
    put("dist_invalid", _cdt(inval))
    nan_any = inval | ~np.isfinite(aux).all(axis=0) | (day & ~np.isfinite(viirs[:3]).all(axis=0))
    put("was_nan", nan_any.astype(np.float32))

    rows = np.stack([out[n] for n in FEATURE_NAMES], axis=1).astype(np.float32)
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)


def compute_af_features(chip: dict) -> np.ndarray:
    """Полная карта признаков (F, H, W) float32 без NaN; порядок каналов — `FEATURE_NAMES`."""
    h, w = chip["viirs"].shape[1:]
    return np.ascontiguousarray(af_feature_rows(chip, None).T).reshape(len(FEATURE_NAMES), h, w)
