"""Чтение чипов AF/BS → dict с массивами (C, H, W) и описаниями каналов.

`data_dir` — корень выборки (data/train, data/test): внутри af/ и bs/. meta.csv — либо в каждом
модуле (`<kind>/meta.csv`, train), либо один общий в корне с колонкой `kind` (тест).

ВАЖНО (Windows): каталог данных называется `aux` — это зарезервированное имя
устройства DOS. Файлы внутри открываются нормально, но сам каталог нельзя
перечислить/создать обычным путём — для таких операций есть `win_path()`
(префикс \\\\?\\ отключает разбор зарезервированных имён).
"""
from __future__ import annotations

import ntpath
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

NODATA_MASK = 255

# ключ результата → (подкаталог, суффикс файла)
AF_LAYERS = {"viirs": ("viirs", "VIIRS_I1-I5"), "aux": ("aux", "AUX")}
BS_LAYERS = {
    "s2_pre": ("sentinel2_pre", "Sentinel-2_pre"),
    "s2_post": ("sentinel2_post", "Sentinel-2_post"),
    "s1_pre": ("sentinel1_pre", "Sentinel-1_pre"),
    "s1_post": ("sentinel1_post", "Sentinel-1_post"),
    "aux": ("aux", "AUX"),
}
LAYERS = {"af": AF_LAYERS, "bs": BS_LAYERS}
MASK_LAYER = ("masks", "MASK")

AF_VIIRS_BANDS = ["I1", "I2", "I3", "I4", "I5", "solar_zenith", "sensor_zenith", "valid"]
AF_AUX_BANDS = ["landcover", "dem", "t2m", "rh2m", "wind_speed"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12", "SCL"]
S1_BANDS = ["VV", "VH"]
BS_AUX_BANDS = ["dem", "slope", "landcover"]


def win_path(path: str | Path) -> str:
    """Путь, безопасный для каталогов с зарезервированными именами (aux) в Windows.

    На любой другой ОС — no-op (организаторы и Docker работают на Linux). На Windows путь
    приводится к абсолютному (префикс \\\\?\\ допустим только для абсолютных путей) и получает
    префикс; уже имеющийся префикс и UNC-пути (\\\\server\\share) не трогаются.
    """
    if os.name != "nt":
        return str(path)
    p = str(path)
    if p.startswith("\\\\"):  # уже \\?\…, \\.\… или UNC
        return p
    # не os.path.abspath: в Windows он превращает «…\aux» в устройство «\\.\aux»
    p = ntpath.normpath(p if ntpath.isabs(p) else ntpath.join(os.getcwd(), p))
    return "\\\\?\\" + p


def chip_path(data_dir: str | Path, kind: str, layer: tuple[str, str], chip_id: str) -> Path:
    return Path(data_dir) / kind / layer[0] / f"{chip_id}_{layer[1]}.tif"


def meta_path(data_dir: str | Path, kind: str) -> tuple[Path, bool]:
    """Путь к meta.csv и признак «общий файл в корне».

    Две раскладки: train — `<data-dir>/<kind>/meta.csv`; тест организаторов — один
    `<data-dir>/meta.csv` на оба модуля (нужен фильтр по колонке `kind`).
    """
    per_kind = Path(data_dir) / kind / "meta.csv"
    if per_kind.exists():
        return per_kind, False
    root = Path(data_dir) / "meta.csv"
    if root.exists():
        return root, True
    raise FileNotFoundError(per_kind)


@lru_cache(maxsize=16)
def _read_meta_cached(path: str, mtime: float, kind: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if kind is None:
        return df
    if "kind" in df.columns:
        sel = df["kind"].astype(str).str.lower() == kind
    else:  # на всякий случай: общий meta без колонки kind — по префиксу chip_id
        sel = df["chip_id"].astype(str).str[:2].str.lower() == kind
    return df[sel].reset_index(drop=True)


def read_meta(data_dir: str | Path, kind: str) -> pd.DataFrame:
    """meta.csv модуля (кэшируется; train и test имеют разный набор колонок и раскладку)."""
    path, shared = meta_path(data_dir, kind)
    return _read_meta_cached(str(path), path.stat().st_mtime, kind if shared else None)


@lru_cache(maxsize=16)
def _meta_index(path: str, mtime: float, kind: str | None) -> dict[str, dict]:
    df = _read_meta_cached(path, mtime, kind)
    return {r["chip_id"]: r for r in df.to_dict("records")}


def _meta_row(data_dir: str | Path, kind: str, chip_id: str) -> dict:
    try:
        path, shared = meta_path(data_dir, kind)
    except FileNotFoundError:
        return {"chip_id": chip_id, "kind": kind}
    index = _meta_index(str(path), path.stat().st_mtime, kind if shared else None)
    return index.get(chip_id, {"chip_id": chip_id, "kind": kind})


def list_chips(data_dir: str | Path, kind: str) -> list[str]:
    """Список chip_id из meta.csv; если meta нет — по файлам первого слоя."""
    try:
        return read_meta(data_dir, kind)["chip_id"].astype(str).tolist()
    except FileNotFoundError:
        sub, suffix = next(iter(LAYERS[kind].values()))
        tail = f"_{suffix}.tif"
        folder = win_path(Path(data_dir) / kind / sub)
        return sorted(f[: -len(tail)] for f in os.listdir(folder) if f.endswith(tail))


def has_masks(data_dir: str | Path, kind: str) -> bool:
    return os.path.isdir(win_path(Path(data_dir) / kind / MASK_LAYER[0]))


def read_tif(path: str | Path, with_georef: bool = False):
    """Один open, все каналы одним вызовом → (C, H, W). Опционально (arr, names, transform, crs)."""
    with rasterio.open(path) as ds:
        arr = ds.read()
        if not with_georef:
            return arr
        names = [d if d else f"band_{i + 1}" for i, d in enumerate(ds.descriptions)]
        return arr, names, ds.transform, ds.crs


def read_mask(data_dir: str | Path, kind: str, chip_id: str) -> np.ndarray:
    """Маска (H, W) uint8 как есть (255 = nodata)."""
    return read_tif(chip_path(data_dir, kind, MASK_LAYER, chip_id))[0]


def _read_chip(data_dir: str | Path, kind: str, chip_id: str, with_mask: bool, layers: tuple[str, ...] | None) -> dict:
    out: dict = {"chip_id": chip_id, "kind": kind, "band_names": {}, "transform": None, "crs": None}
    for key, layer in LAYERS[kind].items():
        if layers is not None and key not in layers:
            continue
        arr, names, transform, crs = read_tif(chip_path(data_dir, kind, layer, chip_id), with_georef=True)
        out[key] = arr
        out["band_names"][key] = names
        if out["transform"] is None:
            out["transform"], out["crs"] = transform, crs
    out["mask"] = out["mask_valid"] = None
    if with_mask:
        mpath = chip_path(data_dir, kind, MASK_LAYER, chip_id)
        if mpath.exists():
            mask = read_tif(mpath)[0]
            out["mask"] = mask
            out["mask_valid"] = mask != NODATA_MASK
    out["meta"] = _meta_row(data_dir, kind, chip_id)
    return out


def read_af_chip(data_dir: str | Path, chip_id: str, with_mask: bool = True,
                 layers: tuple[str, ...] | None = None) -> dict:
    """AF-чип: viirs (8,256,256) f32, aux (5,256,256) f32, mask (256,256) u8|None, mask_valid,
    valid (H,W) bool — канал `valid`==1 и конечные I4/I5 (NaN в viirs сохраняются как есть),
    band_names {слой: [имена]}, transform, crs, meta (строка meta.csv).
    `layers` — читать только перечисленные слои (ускорение инференса)."""
    out = _read_chip(data_dir, "af", chip_id, with_mask, layers)
    if "viirs" in out:
        v = out["viirs"]
        out["valid"] = (v[7] > 0.5) & np.isfinite(v[3]) & np.isfinite(v[4])
    return out


def read_bs_chip(data_dir: str | Path, chip_id: str, with_mask: bool = True,
                 layers: tuple[str, ...] | None = None) -> dict:
    """BS-чип: s2_pre/s2_post (10,512,512) u16, s1_pre/s1_post (2,512,512) i16, aux (3,512,512) i16,
    mask (512,512) u8|None, mask_valid, valid (H,W) bool — SCL pre и post != 0 (есть данные S2),
    band_names, transform, crs, meta."""
    out = _read_chip(data_dir, "bs", chip_id, with_mask, layers)
    if "s2_pre" in out and "s2_post" in out:
        out["valid"] = (out["s2_pre"][9] != 0) & (out["s2_post"][9] != 0)
    return out
