"""Признаки связных компонент предсказанной гари — для классификатора «своя / чужая» компонента.

Эталон размечает только «свой» пожар в буфере события; геометрическое правило (event_zero_r) выбирает компоненты грубо.
Здесь каждая 8-связная компонента маски-кандидата описывается вектором признаков (только содержимое чипа и вероятности модели).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ...features.bs_features import RULE_THR, RULE_THR_DEFAULT, _median
from .postprocess import cloud_mask, event_distance
from .rules import nbr

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

S8 = np.ones((3, 3), bool)
LC_GROUPS = {"lc_forest": (10, 20), "lc_grass": (30,), "lc_crop": (40,), "lc_wet": (90,)}
RING = 10

FEATURE_NAMES = (["area", "area_log", "area_rel", "rank", "n_comp", "chip_burn_frac",
                  "pb_mean", "pb_max", "pb_med", "p1_mean", "p2_mean", "p3_mean",
                  "dnbr_mean", "dnbr_p90", "strong_frac"] + list(LC_GROUPS) + ["lc_other",
                  "dist_event", "dist_dnbr_centroid", "in_r200", "in_r300", "touch_edge",
                  "bbox_fill", "perim2_area", "cloud_in", "cloud_ring", "pb_ring", "pb_contrast", "dist_nearest",
                  "event_pb_g40"])


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(m.astype(np.uint8), k) > 0
    return ndi.binary_dilation(m, iterations=r)


def _edt(mask_false_is_target: np.ndarray) -> np.ndarray:
    """Расстояние до ближайшего False."""
    if cv2 is not None:
        return cv2.distanceTransform(mask_false_is_target.astype(np.uint8), cv2.DIST_L2, 3)
    return ndi.distance_transform_edt(mask_false_is_target)


def component_features(mask: np.ndarray, probs: np.ndarray, chip: dict):
    """mask (H,W) 0..3 — кандидат; probs (4,H,W). → (lab, n, X (n, F) float32). Компонента k ↔ строка k−1."""
    lab, n = ndi.label(mask > 0, structure=S8)
    if n == 0:
        return lab, 0, np.zeros((0, len(FEATURE_NAMES)), np.float32)
    h, w = mask.shape
    p = np.asarray(probs, np.float32)
    pb = 1.0 - p[0]
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(pb), lab, idx)
    order = np.argsort(-area, kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(1, n + 1)

    dn = _median(nbr(chip["s2_pre"]) - nbr(chip["s2_post"]), 3)
    lc = chip["aux"][2]
    t3 = np.full(lc.shape, RULE_THR_DEFAULT[2], np.float32)
    for code, thr in RULE_THR.items():
        t3[lc == code] = thr[2]
    strong = (dn >= t3).astype(np.float32)
    cloud = cloud_mask(chip).astype(np.float32)
    d_event = event_distance(p)
    wgt = np.clip(dn, 0, 1) * (mask > 0)
    yy, xx = np.mgrid[:h, :w]
    tot = float(wgt.sum()) or 1.0
    cy, cx = float((wgt * yy).sum() / tot), float((wgt * xx).sum() / tot)
    cents = np.array(ndi.center_of_mass(np.ones_like(pb), lab, idx))
    ci = np.clip(np.round(cents).astype(int), 0, [h - 1, w - 1])
    from ...features.bs_features import _gauss

    g40 = _gauss(pb, 40.0)

    def mean(a):
        return ndi.mean(a, lab, idx)

    X = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    col = {name: j for j, name in enumerate(FEATURE_NAMES)}
    X[:, col["area"]] = area
    X[:, col["area_log"]] = np.log1p(area)
    X[:, col["area_rel"]] = area / area.max()
    X[:, col["rank"]] = rank
    X[:, col["n_comp"]] = n
    X[:, col["chip_burn_frac"]] = area.sum() / (h * w)
    X[:, col["pb_mean"]] = mean(pb)
    X[:, col["pb_max"]] = ndi.maximum(pb, lab, idx)
    X[:, col["p1_mean"]], X[:, col["p2_mean"]], X[:, col["p3_mean"]] = mean(p[1]), mean(p[2]), mean(p[3])
    X[:, col["dnbr_mean"]] = mean(dn)
    X[:, col["strong_frac"]] = mean(strong)
    other = np.ones(n)
    for name, codes in LC_GROUPS.items():
        f = mean(np.isin(lc, codes).astype(np.float32))
        X[:, col[name]] = f
        other -= f
    X[:, col["lc_other"]] = other
    X[:, col["dist_event"]] = d_event[ci[:, 0], ci[:, 1]] / 256.0
    X[:, col["dist_dnbr_centroid"]] = np.hypot(cents[:, 0] - cy, cents[:, 1] - cx) / 256.0
    X[:, col["in_r200"]] = mean((d_event <= 200).astype(np.float32))
    X[:, col["in_r300"]] = mean((d_event <= 300).astype(np.float32))
    X[:, col["cloud_in"]] = mean(cloud)
    X[:, col["event_pb_g40"]] = g40[ci[:, 0], ci[:, 1]]

    slices = ndi.find_objects(lab)
    pad = RING + 1
    for k, sl in enumerate(slices, start=1):
        r = k - 1
        m_tight = lab[sl] == k
        X[r, col["pb_med"]] = np.median(pb[sl][m_tight])
        X[r, col["dnbr_p90"]] = np.percentile(dn[sl][m_tight], 90)
        X[r, col["touch_edge"]] = float(sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == h or sl[1].stop == w)
        X[r, col["bbox_fill"]] = area[r] / m_tight.size
        perim = area[r] - np.count_nonzero(ndi.binary_erosion(np.pad(m_tight, 1), structure=S8)[1:-1, 1:-1])
        X[r, col["perim2_area"]] = perim * perim / area[r]
        big = (slice(max(sl[0].start - pad, 0), min(sl[0].stop + pad, h)),
               slice(max(sl[1].start - pad, 0), min(sl[1].stop + pad, w)))
        m = lab[big] == k
        ring = _dilate(m, RING) & ~m
        if ring.any():
            X[r, col["pb_ring"]] = pb[big][ring].mean()
            X[r, col["cloud_ring"]] = cloud[big][ring].mean()
        X[r, col["pb_contrast"]] = X[r, col["pb_mean"]] - X[r, col["pb_ring"]]
    if n > 1:
        for k in range(1, n + 1):
            d = _edt(lab != k)
            others = np.delete(idx, k - 1)
            X[k - 1, col["dist_nearest"]] = float(np.min(ndi.minimum(d, lab, others))) / 256.0
    else:
        X[:, col["dist_nearest"]] = 2.0
    return lab, n, X
