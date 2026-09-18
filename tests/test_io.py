from __future__ import annotations

import os

import numpy as np

from firemon.io import (AF_AUX_BANDS, AF_VIIRS_BANDS, BS_AUX_BANDS, S1_BANDS, S2_BANDS, has_masks, list_chips,
                        read_af_chip, read_bs_chip, read_mask, read_meta, win_path)

PREFIX = "\\\\?\\"


def test_win_path_is_noop_off_windows(monkeypatch):
    # только строки: pathlib нельзя инстанцировать при подменённом os.name
    monkeypatch.setattr(os, "name", "posix")
    for p in ("data/train/af/aux", "/mnt/test/bs/aux", "aux", "C:\\data\\aux"):
        assert win_path(p) == p


def test_win_path_prefix_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "getcwd", lambda: "C:\\work")
    assert win_path("C:\\data\\train\\af\\aux") == PREFIX + "C:\\data\\train\\af\\aux"
    assert win_path("C:/data/train/af/aux") == PREFIX + "C:\\data\\train\\af\\aux"  # \\?\ не терпит «/»
    # относительный путь сначала становится абсолютным: префикс допустим только для абсолютных
    assert win_path("data\\train\\..\\test\\aux") == PREFIX + "C:\\work\\data\\test\\aux"
    # без дублирования префикса, UNC не трогаем
    once = win_path("C:\\data\\aux")
    assert win_path(once) == once and once.count(PREFIX) == 1
    assert win_path("\\\\server\\share\\aux") == "\\\\server\\share\\aux"


def test_list_chips(train_dir):
    af, bs = list_chips(train_dir, "af"), list_chips(train_dir, "bs")
    assert len(af) == 420 and len(bs) == 224
    assert af[0] == "AF_tr_000001" and bs[0] == "BS_tr_000001"
    assert has_masks(train_dir, "af") and has_masks(train_dir, "bs")


def test_read_af_chip(train_dir):
    chip = read_af_chip(train_dir, "AF_tr_000001")
    assert chip["viirs"].shape == (8, 256, 256) and chip["viirs"].dtype == np.float32
    assert chip["aux"].shape == (5, 256, 256) and chip["aux"].dtype == np.float32
    assert chip["mask"].shape == (256, 256) and chip["mask"].dtype == np.uint8
    assert chip["mask_valid"].dtype == bool and chip["valid"].shape == (256, 256) and chip["valid"].dtype == bool
    assert chip["band_names"] == {"viirs": AF_VIIRS_BANDS, "aux": AF_AUX_BANDS}
    assert chip["crs"] is not None and chip["transform"].a == 375.0
    assert chip["meta"]["chip_id"] == "AF_tr_000001" and chip["meta"]["satellite"] in ("SNPP", "NOAA20", "NOAA21")
    assert set(np.unique(chip["mask"])) <= {0, 1, 255}
    assert int((chip["mask"] == 1).sum()) == int(chip["meta"]["n_fire_px"])


def test_af_nan_preserved_and_valid_mask(train_dir):
    meta = read_meta(train_dir, "af")
    cid = meta.sort_values("valid_frac")["chip_id"].iloc[0]  # valid_frac ≈ 0.46
    chip = read_af_chip(train_dir, cid)
    i4 = chip["viirs"][3]
    assert np.isnan(i4).any(), "NaN в VIIRS должны сохраняться как есть"
    assert not chip["valid"].all() and chip["valid"].any()
    assert np.isfinite(i4[chip["valid"]]).all()


def test_read_bs_chip(train_dir):
    chip = read_bs_chip(train_dir, "BS_tr_000001")
    for key in ("s2_pre", "s2_post"):
        assert chip[key].shape == (10, 512, 512) and chip[key].dtype == np.uint16
        assert chip["band_names"][key] == S2_BANDS
    for key in ("s1_pre", "s1_post"):
        assert chip[key].shape == (2, 512, 512) and chip[key].dtype == np.int16
        assert chip["band_names"][key] == S1_BANDS
    assert chip["aux"].shape == (3, 512, 512) and chip["aux"].dtype == np.int16
    assert chip["band_names"]["aux"] == BS_AUX_BANDS
    assert chip["mask"].shape == (512, 512) and chip["mask"].dtype == np.uint8
    assert set(np.unique(chip["mask"])) <= {0, 1, 2, 3, 255}
    assert chip["transform"].a == 20.0
    m = chip["meta"]
    assert [(chip["mask"] == k).sum() for k in (1, 2, 3)] == [m["sev1_px"], m["sev2_px"], m["sev3_px"]]


def test_without_mask_and_layer_subset(train_dir, fake_test_dir):
    chip = read_af_chip(train_dir, "AF_tr_000001", with_mask=False, layers=("viirs",))
    assert chip["mask"] is None and chip["mask_valid"] is None and "aux" not in chip
    # тестовая структура: masks/ нет, meta урезана
    assert not has_masks(fake_test_dir, "bs")
    cid = list_chips(fake_test_dir, "bs")[0]
    chip = read_bs_chip(fake_test_dir, cid)  # with_mask=True не должен падать без masks/
    assert chip["mask"] is None and chip["s2_post"].shape == (10, 512, 512)
    assert set(chip["meta"]) == {"chip_id", "kind", "width", "height", "gsd", "valid_frac", "cloud_frac"}
    assert np.array_equal(read_mask(train_dir, "bs", cid), read_bs_chip(train_dir, cid)["mask"])


def _write_root_meta(root):
    (root / "meta.csv").write_text(
        "chip_id,kind,width,height,gsd,valid_frac,cloud_frac\n"
        "AF_te_000001,af,256,256,375,1.0,nan\n"
        "AF_te_000002,af,256,256,375,0.9,nan\n"
        "BS_te_000001,bs,512,512,20,1.0,0.02\n", encoding="utf-8")


def test_read_meta_from_root(tmp_path):
    """Раскладка теста: один <data-dir>/meta.csv с колонкой kind вместо <kind>/meta.csv."""
    _write_root_meta(tmp_path)
    af, bs = read_meta(tmp_path, "af"), read_meta(tmp_path, "bs")
    assert af["chip_id"].tolist() == ["AF_te_000001", "AF_te_000002"] and set(af["kind"]) == {"af"}
    assert bs["chip_id"].tolist() == ["BS_te_000001"] and bs.index.tolist() == [0]
    assert list_chips(tmp_path, "af") == ["AF_te_000001", "AF_te_000002"]
    assert list_chips(tmp_path, "bs") == ["BS_te_000001"]


def test_per_kind_meta_has_priority_over_root(tmp_path):
    _write_root_meta(tmp_path)
    (tmp_path / "af").mkdir()
    (tmp_path / "af" / "meta.csv").write_text("chip_id,kind\nAF_tr_000009,af\n", encoding="utf-8")
    assert list_chips(tmp_path, "af") == ["AF_tr_000009"]  # сначала <kind>/meta.csv
    assert list_chips(tmp_path, "bs") == ["BS_te_000001"]  # для bs своего нет → корневой


def test_read_meta_missing(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        read_meta(tmp_path, "af")


def test_real_test_layout(test_dir):
    af, bs = list_chips(test_dir, "af"), list_chips(test_dir, "bs")
    assert len(af) == 180 and len(bs) == 89
    assert af[0] == "AF_te_000001" and bs[0] == "BS_te_000001"
    assert not has_masks(test_dir, "af") and not has_masks(test_dir, "bs")

    chip = read_af_chip(test_dir, af[0])
    assert chip["viirs"].shape == (8, 256, 256) and chip["viirs"].dtype == np.float32
    assert chip["aux"].shape == (5, 256, 256)
    assert chip["band_names"] == {"viirs": AF_VIIRS_BANDS, "aux": AF_AUX_BANDS}
    assert chip["mask"] is None and chip["meta"]["kind"] == "af" and chip["meta"]["width"] == 256

    chip = read_bs_chip(test_dir, bs[0])
    assert chip["s2_pre"].shape == chip["s2_post"].shape == (10, 512, 512) and chip["s2_pre"].dtype == np.uint16
    assert chip["s1_pre"].shape == (2, 512, 512) and chip["s1_pre"].dtype == np.int16
    assert chip["aux"].shape == (3, 512, 512) and chip["aux"].dtype == np.int16
    assert chip["band_names"]["s2_post"] == S2_BANDS and chip["band_names"]["s1_post"] == S1_BANDS
    assert chip["band_names"]["aux"] == BS_AUX_BANDS
    assert chip["mask"] is None and chip["meta"]["kind"] == "bs" and chip["meta"]["height"] == 512
