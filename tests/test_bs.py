"""BS-модуль: признаки, декодирование, постобработка, BSPredictor (CPU, без данных и без сети)."""
from __future__ import annotations

import numpy as np
import pytest

from firemon.features.bs_features import (FEATURE_NAMES, UNET_CHANNELS, compute_bs_features, feature_names)
from firemon.models.bs.common import decode_probs, fit_decode
from firemon.models.bs.postprocess import fill_holes, median_labels, postprocess_bs, remove_small
from firemon.models.bs.rules import fit_lc_thresholds, predict_bs_rules


def make_chip(seed: int = 0, size: int = 512, with_mask: bool = False) -> dict:
    """Синтетический BS-чип с облаками, nodata S2 (SCL=0, нули в каналах) и nodata S1."""
    rng = np.random.RandomState(seed)
    def s2():
        a = rng.randint(100, 5000, (10, size, size)).astype(np.uint16)
        a[9] = rng.choice([4, 5, 5, 5, 6, 8, 9, 10, 3, 2, 11], (size, size))
        return a
    pre, post = s2(), s2()
    post[:, :40, :60] = 0            # nodata: нулевые отражения → деление на 0 в индексах
    pre[:9, 100:120, 100:120] = 0    # нули без флага SCL
    post[9, 200:300, 200:300] = 9    # облако
    s1_pre = rng.randint(-3000, 0, (2, size, size)).astype(np.int16)
    s1_post = rng.randint(-3000, 0, (2, size, size)).astype(np.int16)
    s1_post[:, -50:, :] = 0
    aux = np.stack([rng.randint(0, 300, (size, size)), rng.randint(0, 13, (size, size)),
                    rng.choice([10, 30, 40, 50, 80, 90, 0], (size, size))]).astype(np.int16)
    chip = {"chip_id": f"BS_syn_{seed}", "kind": "bs", "s2_pre": pre, "s2_post": post, "s1_pre": s1_pre,
            "s1_post": s1_post, "aux": aux, "valid": (pre[9] != 0) & (post[9] != 0), "mask": None}
    if with_mask:
        chip["mask"] = rng.randint(0, 4, (size, size)).astype(np.uint8)
    return chip


def test_features_shape_finite_names():
    f = compute_bs_features(make_chip())
    assert f.shape == (len(FEATURE_NAMES), 512, 512) and f.dtype == np.float32
    assert np.isfinite(f).all()
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert set(UNET_CHANNELS) <= set(FEATURE_NAMES)


def test_features_groups_subset_consistent():
    chip = make_chip(1)
    full = compute_bs_features(chip)
    groups = ["diff", "lc"]
    sub = compute_bs_features(chip, groups=groups)
    names = feature_names(groups)
    assert sub.shape[0] == len(names)
    for i, n in enumerate(names):
        np.testing.assert_array_equal(sub[i], full[FEATURE_NAMES.index(n)])


def test_features_deterministic_and_nodata_zeroed():
    chip = make_chip(2)
    a, b = compute_bs_features(chip), compute_bs_features(chip)
    np.testing.assert_array_equal(a, b)
    assert a[FEATURE_NAMES.index("invalid"), :40, :60].all()
    assert (a[FEATURE_NAMES.index("dNBR"), :40, :60] == 0).all()


def test_decode_probs():
    p = np.zeros((4, 2, 2), np.float32)
    p[:, 0, 0] = [0.7, 0.1, 0.1, 0.1]
    p[:, 0, 1] = [0.45, 0.30, 0.20, 0.05]
    p[:, 1, 0] = [0.1, 0.2, 0.3, 0.4]
    p[:, 1, 1] = [0.4, 0.25, 0.3, 0.05]
    np.testing.assert_array_equal(decode_probs(p), [[0, 0], [3, 0]])
    np.testing.assert_array_equal(decode_probs(p, 0.5), [[0, 1], [3, 2]])
    np.testing.assert_array_equal(decode_probs(p, 0.5, (1.0, 1.0, 2.0)), [[0, 1], [3, 2]])
    np.testing.assert_array_equal(decode_probs(p, 0.5, (1.5, 1.0, 1.0)), [[0, 1], [3, 1]])


def test_fit_decode_recovers_threshold():
    rng = np.random.RandomState(0)
    gt = (rng.random_sample((2, 64, 64)) < 0.3).astype(np.uint8)
    pb = np.where(gt > 0, 0.45, 0.35) + rng.normal(0, 0.01, gt.shape)  # идеальный порог ≈ 0.40
    probs = np.stack([1 - pb, pb, np.zeros_like(pb), np.zeros_like(pb)], axis=1)
    dec = fit_decode(probs, gt)
    assert 0.36 <= dec["t_burn"] <= 0.44 and dec["iou_burn"] > 0.99


def test_postprocess_ops():
    m = np.zeros((32, 32), np.uint8)
    m[2:4, 2:4] = 1                 # компонента 4 px
    m[10:25, 10:25] = 2
    m[15:17, 15:17] = 0             # дыра 4 px
    m[12, 12] = 3                   # одиночный пиксель другой степени
    out = remove_small(m, 5)
    assert out[2:4, 2:4].sum() == 0 and out[11, 11] == 2
    out = fill_holes(m, 10)
    assert (out[15:17, 15:17] == 2).all() and out[0, 0] == 0
    out = median_labels(m, 3)
    assert out[12, 12] == 2 and ((out > 0) == (m > 0)).all()
    chip = make_chip(3, size=32)
    chip["s2_post"][9] = 5
    chip["s2_pre"][9] = 5
    chip["s2_post"][9, 10:25, 10:18] = 9
    out = postprocess_bs(m, chip, {"scl_zero": True, "min_component_px": 5, "fill_holes_px": 10})
    assert out[10:25, 10:18].sum() == 0 and out[20, 20] == 2 and set(np.unique(out)) <= {0, 1, 2, 3}


def test_event_postprocess():
    from firemon.models.bs.postprocess import (event_distance, keep_inside_largest_circle, keep_near_largest,
                                               keep_relative_area)

    m = np.zeros((128, 128), np.uint8)
    m[10:40, 10:40] = 2          # крупнейшая компонента (900 px)
    m[50:56, 50:56] = 1          # близкая малая (36 px)
    m[110:120, 110:120] = 3      # далёкая (100 px)
    probs = np.zeros((4, 128, 128), np.float32)
    probs[0] = 1.0
    probs[2, 10:40, 10:40], probs[0, 10:40, 10:40] = 0.9, 0.1
    d = event_distance(probs, sigma=5.0)
    assert d[25, 25] < 5 and d[115, 115] > 100
    chip = make_chip(4, size=128)
    out = postprocess_bs(m, chip, {"event_keep_r": 60}, probs=probs)
    assert out[20, 20] == 2 and out[52, 52] == 1 and out[115, 115] == 0
    out = postprocess_bs(m, chip, {"event_zero_r": 30}, probs=probs)
    assert out[52, 52] == 0 and out[25, 25] == 2
    np.testing.assert_array_equal(postprocess_bs(m, chip, {"event_keep_r": 60}, probs=None), m)  # без probs — no-op
    assert keep_near_largest(m, 50)[115, 115] == 0 and keep_near_largest(m, 50)[52, 52] == 1
    rel = keep_relative_area(m, 10)
    assert rel[52, 52] == 0 and rel[115, 115] == 3 and rel[20, 20] == 2
    circ = keep_inside_largest_circle(m, 45)
    assert circ[20, 20] == 2 and circ[115, 115] == 0


def test_deterministic_ce_matches_torch():
    """Детерминированная CE (one-hot) численно равна F.cross_entropy с весами и ignore_index=255."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    import torch.nn.functional as F

    from firemon.models.bs.unet import ce_loss

    torch.manual_seed(0)
    lo, t = torch.randn(2, 4, 16, 16), torch.randint(0, 4, (2, 16, 16))
    t[0, 0, :3] = 255
    w = torch.tensor([0.5, 1.0, 1.0, 1.2])
    assert abs(float(ce_loss(lo, t, w)) - float(F.cross_entropy(lo, t, weight=w, ignore_index=255))) < 1e-5


def test_component_features():
    """Признаки связных компонент: форма, конечность, площадь/ранг, пустая маска."""
    from firemon.models.bs.components import FEATURE_NAMES as CF, component_features

    chip = make_chip(11, size=128)
    mask = np.zeros((128, 128), np.uint8)
    mask[10:40, 10:40] = 2          # 900 px, касается… нет, внутри
    mask[0:6, 100:110] = 1          # 60 px, касается края
    probs = np.zeros((4, 128, 128), np.float32)
    probs[0] = 0.9
    probs[0][mask > 0], probs[2][mask > 0] = 0.2, 0.8
    lab, n, X = component_features(mask, probs, chip)
    assert n == 2 and X.shape == (2, len(CF)) and np.isfinite(X).all()
    big = int(lab[20, 20]) - 1
    assert X[big, CF.index("area")] == 900 and X[big, CF.index("rank")] == 1 and X[big, CF.index("area_rel")] == 1.0
    assert X[1 - big, CF.index("touch_edge")] == 1.0 and X[big, CF.index("touch_edge")] == 0.0
    assert X[big, CF.index("pb_contrast")] > 0.5
    _, n0, X0 = component_features(np.zeros_like(mask), probs, chip)
    assert n0 == 0 and X0.shape == (0, len(CF))


def test_nonempty_prior():
    """Приор «событие есть в каждом чипе»: срабатывает только при < 25 px гари и только если есть компонента ≥ 25 px."""
    chip = make_chip(9, size=64)
    chip["s2_pre"][9] = 5
    chip["s2_post"][9] = 5
    probs = np.zeros((4, 64, 64), np.float32)
    probs[0] = 0.95
    probs[0, 10:20, 10:20], probs[2, 10:20, 10:20] = 0.7, 0.3   # p_burn 0.3: ниже t_burn=0.42, выше t_burn/2
    probs[0, 40:43, 40:43], probs[1, 40:43, 40:43] = 0.7, 0.3   # 9 px — меньше MMU
    dec = {"t_burn": 0.42, "w": [1.0, 1.0, 1.0]}
    empty = decode_probs(probs, dec["t_burn"], dec["w"])
    assert not empty.any()
    out = postprocess_bs(empty, chip, {"nonempty_min_px": 25}, probs=probs, decode=dec)
    assert (out[10:20, 10:20] == 2).all() and out.sum() == 2 * 100          # крупнейшая компонента, степень по argmax
    np.testing.assert_array_equal(postprocess_bs(empty, chip, {}, probs=probs, decode=dec), empty)  # выключен → как было
    full = empty.copy()
    full[30:36, 30:36] = 1                                                   # уже есть 36 px гари → приор молчит
    np.testing.assert_array_equal(postprocess_bs(full, chip, {"nonempty_min_px": 25}, probs=probs, decode=dec), full)
    small = probs.copy()
    small[0, 10:20, 10:20], small[2, 10:20, 10:20] = 0.95, 0.0               # остаётся только компонента 9 px
    assert not postprocess_bs(empty, chip, {"nonempty_min_px": 25}, probs=small, decode=dec).any()
    chip["s2_post"][9, 10:20, 10:20] = 9                                     # под запретной маской SCL — не берём
    assert not postprocess_bs(empty, chip, {"nonempty_min_px": 25, "scl_post_zero": True}, probs=probs, decode=dec).any()


def test_event_features_localize_largest_strong_component():
    from firemon.features.bs_features import event_features

    rule = np.zeros((64, 64), np.float32)
    rule[5:25, 5:25] = 3
    rule[50:54, 50:54] = 2
    rule[40:42, 10:12] = 1
    f = event_features(rule, rule / 4.0)
    assert f["dist_strong_cc"][10, 10] == 0 and f["dist_strong_cc"][52, 52] > 0
    assert f["cc_rank"][10, 10] == 1 and f["cc_rank"][52, 52] == 2 and f["cc_rank"][0, 63] == 0
    assert f["cc_area_rel"][10, 10] == 1.0 and 0 < f["cc_area_rel"][41, 11] < 0.05
    empty = event_features(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32))
    assert all(np.isfinite(v).all() for v in empty.values())


def test_rules_and_lc_thresholds():
    chips = [make_chip(s, size=64, with_mask=True) for s in range(3)]
    thr = fit_lc_thresholds(chips, n_rounds=1)
    assert all(t[0] < t[1] < t[2] for t in thr.values())
    pred = predict_bs_rules(chips[0], lc_thresholds=thr)
    assert pred.shape == (64, 64) and pred.dtype == np.uint8 and pred.max() <= 3
    assert (pred[~chips[0]["valid"]] == 0).all()


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from firemon.models.bs.unet import build_unet

    torch.manual_seed(0)
    model = build_unet("resnet18", len(UNET_CHANNELS), pretrained=False)
    path = tmp_path_factory.mktemp("w") / "bs_unet_tiny.pt"
    norm = {"channels": UNET_CHANNELS, "mean": [0.0] * len(UNET_CHANNELS), "std": [1.0] * len(UNET_CHANNELS)}
    torch.save({"state_dict": {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()},
                "encoder": "resnet18", "arch": "unet", "loss": "ce_dice", "channels": UNET_CHANNELS, "norm": norm}, path)
    return path


def test_predictor_cpu(tiny_checkpoint, monkeypatch):
    """CPU-режим обязан работать без CUDA: принудительно device=cpu и «выключенная» CUDA."""
    import torch

    from firemon.models.bs.predictor import BSPredictor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = {"device": "cpu", "unet_weights": [str(tiny_checkpoint)], "use_lgbm": False, "num_threads": 2,
           "decode": {"t_burn": 0.3, "w": [1.0, 1.0, 1.0]}, "postprocess": {"scl_zero": True, "min_component_px": 4}}
    pred = BSPredictor(cfg).load()
    assert pred.device.type == "cpu" and pred.models
    chip = make_chip(5)
    a = pred.predict(chip)
    b = pred.predict(chip)
    assert a.shape == (512, 512) and a.dtype == np.uint8 and set(np.unique(a)) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(a, b)  # детерминизм
    assert (a[:40, :60] == 0).all()      # nodata S2 → 0 (scl_zero)
    # три бинарные маски классов не пересекаются по построению
    assert sum((a == k).astype(int) for k in (1, 2, 3)).max() <= 1
    # батч = по одному
    c2 = make_chip(6)
    batch = pred.predict_batch([chip, c2])
    np.testing.assert_array_equal(batch[0], a)


def test_predictor_stack_cpu(tiny_checkpoint, tmp_path, monkeypatch):
    """Полный путь UNet → стек-LGBM на CPU: форма, значения 0..3, детерминизм."""
    import torch

    lgb = pytest.importorskip("lightgbm")
    from firemon.models.bs.predictor import BSPredictor
    from firemon.models.bs.stack import stack_feature_names

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.RandomState(0)
    n_feat = len(stack_feature_names())
    X, y = rng.normal(size=(400, n_feat)).astype(np.float32), rng.randint(0, 4, 400)
    booster = lgb.train(dict(objective="multiclass", num_class=4, num_leaves=4, verbosity=-1, deterministic=True,
                             num_threads=1, seed=42), lgb.Dataset(X, y), num_boost_round=3)
    path = tmp_path / "bs_stack.txt"
    booster.save_model(str(path))
    cfg = {"device": "cpu", "unet_weights": [str(tiny_checkpoint)], "use_stack": True, "stack_weights": str(path),
           "stack_gate": 0.0, "num_threads": 2, "decode": {"t_burn": 0.5, "w": [1.0, 1.0, 1.0]}, "postprocess": {}}
    pred = BSPredictor(cfg).load()
    assert pred.stack is not None
    chip = make_chip(8)
    a, b = pred.predict(chip), pred.predict(chip)
    assert a.shape == (512, 512) and a.dtype == np.uint8 and set(np.unique(a)) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(a, b)


def test_bs_variant_switch():
    """bs.variant: по умолчанию v4; ms — пресет кандидата v5 из configs/bs_inference.yaml; явные ключи сильнее пресета."""
    from firemon.models.bs.infer import load_bs_config

    v4 = load_bs_config({})
    assert v4["variant"] == "v4" and v4["stack_weights"] == "weights/bs_stack.txt" and float(v4["stack_gate"]) == 0.03
    assert len(v4["unet_weights"]) == 2 and not v4.get("aux_unet_weights") and "variants" not in v4
    ms = load_bs_config({"variant": "ms", "model": "unet"})
    assert ms["variant"] == "ms" and ms["stack_weights"] == "weights/bs_stack_ms.txt" and float(ms["stack_gate"]) == 0.01
    assert ms["decode_file"] == "weights/bs_decode_ms.json" and len(ms["unet_weights"]) == 3
    assert ms["aux_unet_weights"] == ["weights/bs_unet_r34_v4_s050.pt"]
    assert ms["tta"] is False and ms["gpu_half"] is False            # базовые настройки v4 сохраняются
    over = load_bs_config({"variant": "ms", "stack_gate": 0.05})
    assert float(over["stack_gate"]) == 0.05
    with pytest.raises(ValueError):
        load_bs_config({"variant": "nope"})
    # общий конфиг по умолчанию — v4, отдельный — ms
    from firemon.utils import REPO_ROOT, load_config

    assert "variant" not in load_config(REPO_ROOT / "configs" / "inference.yaml")["bs"]
    assert load_config(REPO_ROOT / "configs" / "inference_bs_ms.yaml")["bs"]["variant"] == "ms"


def test_predictor_multiscale_aux_cpu(tiny_checkpoint, tmp_path, monkeypatch):
    """Вариант ms на CPU: член базы на масштабе 0.75, доп. сеть 0.5× → признаки aux_* стека; форма, 0..3, детерминизм."""
    import torch

    lgb = pytest.importorskip("lightgbm")
    from firemon.models.bs.predictor import BSPredictor
    from firemon.models.bs.stack import stack_feature_names

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ck = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    paths = {}
    for name, scale in (("s075", 0.75), ("s050", 0.5)):
        paths[name] = tmp_path / f"bs_unet_tiny_{name}.pt"
        torch.save({**ck, "scale": scale}, paths[name])
    n_feat = len(stack_feature_names(n_aux=1))
    assert n_feat == len(stack_feature_names()) + 4
    rng = np.random.RandomState(0)
    booster = lgb.train(dict(objective="multiclass", num_class=4, num_leaves=4, verbosity=-1, deterministic=True,
                             num_threads=1, seed=42),
                        lgb.Dataset(rng.normal(size=(400, n_feat)).astype(np.float32), rng.randint(0, 4, 400)), 3)
    stack_path = tmp_path / "bs_stack_ms.txt"
    booster.save_model(str(stack_path))
    cfg = {"device": "cpu", "unet_weights": [str(tiny_checkpoint), str(paths["s075"])],
           "aux_unet_weights": [str(paths["s050"])], "use_stack": True, "stack_weights": str(stack_path),
           "stack_gate": 0.0, "num_threads": 2, "decode": {"t_burn": 0.5, "w": [1.0, 1.0, 1.0]}, "postprocess": {}}
    pred = BSPredictor(cfg).load()
    assert len(pred.models) == 2 and len(pred.aux_models) == 1 and pred.models[1]["scale"] == 0.75
    chip = make_chip(12)
    a, b = pred.predict(chip), pred.predict(chip)
    assert a.shape == (512, 512) and a.dtype == np.uint8 and set(np.unique(a)) <= {0, 1, 2, 3}
    np.testing.assert_array_equal(a, b)
    with pytest.raises(FileNotFoundError):
        BSPredictor({**cfg, "aux_unet_weights": [str(tmp_path / "nope.pt")]}).load()


def test_predictor_cuda_request_falls_back(tiny_checkpoint, monkeypatch):
    import torch

    from firemon.models.bs.predictor import BSPredictor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pred = BSPredictor({"device": "cuda", "unet_weights": [str(tiny_checkpoint)]}).load()
    assert pred.device.type == "cpu"


def test_predictor_without_weights_uses_rules(tmp_path):
    from firemon.models.bs.predictor import BSPredictor

    pred = BSPredictor({"unet_weights": [str(tmp_path / "nope.pt")], "use_lgbm": True,
                        "lgbm_weights": str(tmp_path / "nope.txt")}).load()
    chip = make_chip(7)
    np.testing.assert_array_equal(pred.predict(chip), predict_bs_rules(chip))
