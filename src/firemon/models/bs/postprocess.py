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


def postprocess_bs(mask: np.ndarray, chip: dict, cfg: dict | None) -> np.ndarray:
    """cfg: scl_zero (bool), water_lc_zero (bool), builtup_zero (bool), min_component_px, fill_holes_px, median_size."""
    cfg = cfg or {}
    out = mask.astype(np.uint8, copy=True)
    forbid = np.zeros(out.shape, bool)
    if cfg.get("scl_zero", False):
        forbid |= scl_zero_mask(chip)
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
    out = remove_small(out, int(cfg.get("min_component_px", 0)))
    out = fill_holes(out, int(cfg.get("fill_holes_px", 0)), forbid)
    out = median_labels(out, int(cfg.get("median_size", 0)))
    return out
