"""Постобработка масок. BS: маскирование по SCL/landcover, мелкие компоненты, дыры, сглаживание степеней.

Каждый шаг включается конфигом (`bs.postprocess` в configs/bs_*.yaml) — оставлены только шаги, давшие
прирост OOF на обоих сплитах (см. reports/bs_module.md).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_S8 = np.ones((3, 3), bool)  # 8-связность


def scl_zero_mask(chip: dict, pre_codes=(0, 3, 6, 11), post_codes=(0, 3, 9, 11)) -> np.ndarray:
    """Пиксели, где эталон по построению = 0 (EDA: доля гари 0.0000 для этих кодов SCL на train)."""
    return np.isin(chip["s2_pre"][9], pre_codes) | np.isin(chip["s2_post"][9], post_codes)


def cloud_mask(chip: dict, codes=(3, 8, 9, 10)) -> np.ndarray:
    """Облако/тень по SCL в любой из дат (для сравнения политик: как есть / заполнить из соседей / обнулить)."""
    return np.isin(chip["s2_pre"][9], codes) | np.isin(chip["s2_post"][9], codes)


def event_distance(probs: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """Расстояние (px) до оценённого центра события — пика ожидаемой степени, сглаженной гауссианом σ (≈0.8 км).
    Эталон размечает только «свой» пожар внутри буфера события (~4 км), буфер НЕ центрирован по чипу (EDA B4),
    поэтому центр оценивается по содержимому чипа (вероятности модели), а не по геометрии/геопривязке."""
    cy, cx = event_center(probs, sigma)
    h, w = np.asarray(probs).shape[1:]
    yy, xx = np.ogrid[:h, :w]
    return np.hypot(yy - cy, xx - cx)


def event_center(probs: np.ndarray, sigma: float = 40.0, mode: str = "argmax", r_box: float | None = None,
                 t_burn: float = 0.5) -> tuple[float, float]:
    """Оценка центра события (y, x). mode:
      argmax — пик ожидаемой степени, сглаженной гауссианом σ (текущий вариант);
      clip   — то же, но координаты обрезаются рамкой ±r_box от центра чипа;
      boxmax — пик ищется только внутри рамки;
      cc     — p_burn-взвешенный центроид крупнейшей компоненты p_burn > t_burn внутри рамки (нет компонент → boxmax).
    Рамочный приор: по EDA B4c центр буфера события смещён от центра чипа не более чем на ≈ ±170 px (q95 230)."""
    from ...features.bs_features import _gauss

    p = np.asarray(probs, np.float32)
    h, w = p.shape[1:]
    g = _gauss(p[1] + 2.0 * p[2] + 3.0 * p[3], sigma)
    c0y, c0x = (h - 1) / 2.0, (w - 1) / 2.0
    if mode == "argmax" or r_box is None:
        cy, cx = np.unravel_index(int(g.argmax()), g.shape)
        return float(cy), float(cx)
    y0, y1 = int(max(np.ceil(c0y - r_box), 0)), int(min(np.floor(c0y + r_box), h - 1))
    x0, x1 = int(max(np.ceil(c0x - r_box), 0)), int(min(np.floor(c0x + r_box), w - 1))
    if mode == "clip":
        cy, cx = np.unravel_index(int(g.argmax()), g.shape)
        return float(np.clip(cy, y0, y1)), float(np.clip(cx, x0, x1))
    sub = g[y0:y1 + 1, x0:x1 + 1]
    by, bx = np.unravel_index(int(sub.argmax()), sub.shape)
    if mode == "boxmax":
        return float(by + y0), float(bx + x0)
    if mode == "cc":
        pb = (1.0 - p[0])[y0:y1 + 1, x0:x1 + 1]
        lab, n = ndi.label(pb > t_burn, structure=_S8)
        if n == 0:
            return float(by + y0), float(bx + x0)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        sizes[0] = 0
        k = int(sizes.argmax())
        cy, cx = ndi.center_of_mass(pb * (lab == k))
        return float(cy + y0), float(cx + x0)
    raise ValueError(f"event_center: неизвестный mode {mode!r}")


def event_distance_cfg(probs: np.ndarray, cfg: dict, decode: dict | None) -> np.ndarray:
    ec = cfg.get("event_center") or {}
    cy, cx = event_center(probs, float(ec.get("sigma", 40.0)), ec.get("mode", "argmax"), ec.get("r_box"),
                          float((decode or {}).get("t_burn") or 0.5))
    h, w = np.asarray(probs).shape[1:]
    yy, xx = np.ogrid[:h, :w]
    return np.hypot(yy - cy, xx - cx)


def keep_components_touching(mask: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Оставляет 8-связные компоненты гари, пересекающие область `inside`."""
    lab, n = ndi.label(mask > 0, structure=_S8)
    if n == 0:
        return mask
    ok = np.zeros(n + 1, bool)
    ok[np.unique(lab[inside & (lab > 0)])] = True
    ok[0] = False
    return np.where(ok[lab], mask, 0).astype(np.uint8)


def _components(mask: np.ndarray):
    lab, n = ndi.label(mask > 0, structure=_S8)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    return lab, n, sizes


def keep_near_largest(mask: np.ndarray, radius: float) -> np.ndarray:
    """Крупнейшая компонента + все компоненты, имеющие пиксели в пределах `radius` от её центроида."""
    lab, n, sizes = _components(mask)
    if n <= 1:
        return mask
    big = int(sizes.argmax())
    cy, cx = ndi.center_of_mass(lab == big)
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    return keep_components_touching(mask, (np.hypot(yy - cy, xx - cx) <= radius) | (lab == big))


def keep_relative_area(mask: np.ndarray, pct: float) -> np.ndarray:
    """Компоненты площадью ≥ pct % от крупнейшей."""
    lab, n, sizes = _components(mask)
    if n <= 1:
        return mask
    ok = sizes >= sizes.max() * pct / 100.0
    ok[0] = False
    return np.where(ok[lab], mask, 0).astype(np.uint8)


def keep_inside_largest_circle(mask: np.ndarray, radius: float) -> np.ndarray:
    """Всё внутри минимальной охватывающей окружности крупнейшей компоненты, расширенной до `radius`."""
    lab, n, sizes = _components(mask)
    if n <= 1:
        return mask
    pts = np.argwhere(lab == int(sizes.argmax())).astype(np.float32)
    try:
        import cv2

        (cy, cx), r = cv2.minEnclosingCircle(pts)
    except ImportError:  # pragma: no cover
        cy, cx = pts.mean(0)
        r = float(np.hypot(pts[:, 0] - cy, pts[:, 1] - cx).max())
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    return np.where(np.hypot(yy - cy, xx - cx) <= max(float(r), radius), mask, 0).astype(np.uint8)


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Удаляет 8-связные компоненты гари (mask>0) меньше min_px."""
    if min_px <= 1:
        return mask
    lab, n = ndi.label(mask > 0, structure=_S8)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    small = sizes < min_px
    small[0] = False
    out = mask.copy()
    out[small[lab]] = 0
    return out


def fill_holes(mask: np.ndarray, max_px: int, forbid: np.ndarray | None = None) -> np.ndarray:
    """Заполняет дыры (компоненты фона, не касающиеся края) меньше max_px классом ближайшего пикселя гари."""
    if max_px <= 0:
        return mask
    bg = mask == 0
    lab, n = ndi.label(bg)  # 4-связность для фона
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    ok = sizes < max_px
    ok[0] = False
    ok[border] = False
    holes = ok[lab]
    if forbid is not None:
        holes &= ~forbid
    if not holes.any():
        return mask
    idx = ndi.distance_transform_edt(mask == 0, return_distances=False, return_indices=True)
    out = mask.copy()
    out[holes] = mask[idx[0][holes], idx[1][holes]]
    return out


def median_labels(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Медиана по меткам степени ВНУТРИ гари (контур не меняется): классы упорядочены → медиана осмысленна."""
    if size <= 1:
        return mask
    burn = mask > 0
    if not burn.any():
        return mask
    # фон заменяем ближайшей степенью, чтобы он не тянул медиану к 0 на границе
    idx = ndi.distance_transform_edt(~burn, return_distances=False, return_indices=True)
    filled = mask[idx[0], idx[1]]
    sm = ndi.median_filter(filled, size=size, mode="nearest")
    return np.where(burn, sm, 0).astype(np.uint8)


def nonempty_prior(out: np.ndarray, probs: np.ndarray, forbid: np.ndarray, decode: dict | None,
                   min_px: int) -> np.ndarray:
    """Приор «событие есть в каждом чипе»: BS-чип по построению набора нарезан вокруг пожара (в train нет чипов без
    гари, минимум ≈ 27 px). Срабатывает ТОЛЬКО если после всей постобработки гари < min_px: берётся крупнейшая
    8-связная компонента p_burn > t_burn/2, если она ≥ min_px; степень — обычное декодирование (argmax p[1:]·w)."""
    if int(np.count_nonzero(out)) >= min_px:
        return out
    decode = decode or {}
    t = float(decode.get("t_burn") or 0.5) / 2.0
    p = np.asarray(probs, np.float32)
    lab, n, sizes = _components(((1.0 - p[0]) > t) & ~forbid)
    if n == 0 or sizes.max() < min_px:
        return out
    comp = lab == int(sizes.argmax())
    w = np.asarray(decode.get("w") or (1.0, 1.0, 1.0), np.float32).reshape(3, 1, 1)
    cls = (p[1:] * w).argmax(axis=0).astype(np.uint8) + 1
    return np.where(comp, cls, 0).astype(np.uint8)


def postprocess_bs(mask: np.ndarray, chip: dict, cfg: dict | None, probs: np.ndarray | None = None,
                   decode: dict | None = None) -> np.ndarray:
    """cfg: scl_zero, scl_post_zero, water_lc_zero, builtup_zero (bool), cloud_policy, event_keep_r / event_zero_r
    (px, нужны вероятности `probs` (4,H,W)), largest_keep_r / rel_area_pct / largest_circle_r (компоненты
    относительно крупнейшей), min_component_px, fill_holes_px, median_size, nonempty_min_px (приор «событие есть
    в каждом чипе», нужны `probs` и `decode` = {t_burn, w})."""
    cfg = cfg or {}
    out = mask.astype(np.uint8, copy=True)
    forbid = np.zeros(out.shape, bool)
    if cfg.get("scl_zero", False):
        forbid |= scl_zero_mask(chip)
    if cfg.get("scl_post_zero", False):
        forbid |= np.isin(chip["s2_post"][9], (0, 3, 9, 11))
    if cfg.get("water_lc_zero", False):
        forbid |= chip["aux"][2] == 80
    if cfg.get("builtup_zero", False):
        forbid |= chip["aux"][2] == 50
    out[forbid] = 0
    policy = cfg.get("cloud_policy", "as_is")
    if policy != "as_is":
        cloud = cloud_mask(chip) & ~forbid
        if cloud.any() and policy == "zero":
            out[cloud] = 0
        elif cloud.any() and policy == "fill" and not cloud.all():
            idx = ndi.distance_transform_edt(cloud, return_distances=False, return_indices=True)
            out[cloud] = out[idx[0][cloud], idx[1][cloud]]
            out[forbid] = 0
    if probs is not None and (cfg.get("event_keep_r") or cfg.get("event_zero_r")) and (out > 0).any():
        dist = event_distance_cfg(probs, cfg, decode)
        if cfg.get("event_keep_r"):
            out = keep_components_touching(out, dist <= float(cfg["event_keep_r"]))
        if cfg.get("event_zero_r"):
            out[dist > float(cfg["event_zero_r"])] = 0
    if cfg.get("largest_keep_r"):
        out = keep_near_largest(out, float(cfg["largest_keep_r"]))
    if cfg.get("rel_area_pct"):
        out = keep_relative_area(out, float(cfg["rel_area_pct"]))
    if cfg.get("largest_circle_r"):
        out = keep_inside_largest_circle(out, float(cfg["largest_circle_r"]))
    out = remove_small(out, int(cfg.get("min_component_px", 0)))
    out = fill_holes(out, int(cfg.get("fill_holes_px", 0)), forbid)
    out = median_labels(out, int(cfg.get("median_size", 0)))
    if probs is not None and cfg.get("nonempty_min_px"):
        out = nonempty_prior(out, probs, forbid, decode, int(cfg["nonempty_min_px"]))
    return out
