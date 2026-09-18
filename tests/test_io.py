from __future__ import annotations

import numpy as np

from firemon.io import (AF_AUX_BANDS, AF_VIIRS_BANDS, BS_AUX_BANDS, S1_BANDS, S2_BANDS, has_masks, list_chips,
                        read_af_chip, read_bs_chip, read_mask, read_meta)


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
