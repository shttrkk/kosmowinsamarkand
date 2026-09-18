from __future__ import annotations

import numpy as np
import pytest

from firemon.metrics import af_f1, bs_metrics, evaluate, score


def test_af_f1_manual_3x3():
    gt = np.array([[1, 1, 0], [0, 0, 0], [0, 1, 0]], np.uint8)
    pred = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    r = af_f1({"a": pred}, {"a": gt})
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_af_f1_is_micro_not_macro():
    gt1, pred1 = np.ones((3, 3), np.uint8), np.ones((3, 3), np.uint8)  # 9 TP
    gt2 = np.zeros((3, 3), np.uint8)
    gt2[0, 0] = 1
    pred2 = np.zeros((3, 3), np.uint8)
    pred2[0, 1] = 1  # 1 FP + 1 FN
    r = af_f1({"a": pred1, "b": pred2}, {"a": gt1, "b": gt2})
    assert (r["tp"], r["fp"], r["fn"]) == (9, 1, 1)
    assert r["f1"] == pytest.approx(18 / 20)  # макро дало бы (1 + 0) / 2


def test_af_nodata_excluded():
    gt = np.array([[1, 255, 0], [0, 255, 0], [0, 0, 0]], np.uint8)
    pred = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
    r = af_f1({"a": pred}, {"a": gt})
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 0)
    assert af_f1({"a": pred}, {"a": gt}, ignore_nodata=False)["fp"] == 2


def test_af_empty_everything_is_one():
    z = np.zeros((3, 3), np.uint8)
    r = af_f1({"a": z}, {"a": z})
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0


def test_af_missed_class_is_zero():
    gt = np.zeros((3, 3), np.uint8)
    gt[1, 1] = 1
    assert af_f1({"a": np.zeros((3, 3), np.uint8)}, {"a": gt})["f1"] == 0.0


def test_bs_manual_3x3():
    gt = np.array([[1, 1, 2], [2, 3, 3], [0, 0, 0]], np.uint8)
    pred = np.array([[1, 2, 2], [2, 3, 0], [1, 0, 0]], np.uint8)
    r = bs_metrics({"a": pred}, {"a": gt})
    # burn: gt 6 px, pred 6 px, пересечение 5, объединение 7
    assert r["iou_burn"] == pytest.approx(5 / 7)
    assert r["iou_1"] == pytest.approx(1 / 3)   # tp=1, fp=1 (2,0), fn=1 (0,1)
    assert r["iou_2"] == pytest.approx(2 / 3)   # tp=2, fp=1 (0,1), fn=0
    assert r["iou_3"] == pytest.approx(1 / 2)   # tp=1, fp=0, fn=1
    assert r["miou_sev"] == pytest.approx((1 / 3 + 2 / 3 + 1 / 2) / 3)


def test_bs_absent_class_rule():
    gt = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]], np.uint8)
    r = bs_metrics({"a": gt.copy()}, {"a": gt})
    # классов 2 и 3 нет ни в эталоне, ни в предсказании → IoU = 1
    assert r["iou_1"] == r["iou_2"] == r["iou_3"] == r["miou_sev"] == r["iou_burn"] == 1.0
    # класс есть в эталоне, но не предсказан → IoU = 0
    r = bs_metrics({"a": np.zeros_like(gt)}, {"a": gt})
    assert r["iou_1"] == 0.0 and r["iou_burn"] == 0.0 and r["iou_2"] == 1.0
    # класс предсказан, но в эталоне его нет → IoU = 0 (знаменатель > 0)
    pred = gt.copy()
    pred[2, 2] = 3
    assert bs_metrics({"a": pred}, {"a": gt})["iou_3"] == 0.0


def test_bs_nodata_excluded():
    gt = np.array([[1, 255], [0, 2]], np.uint8)
    pred = np.array([[1, 3], [0, 2]], np.uint8)
    r = bs_metrics({"a": pred}, {"a": gt})
    assert r["iou_burn"] == 1.0 and r["iou_3"] == 1.0


def test_score_weights_and_evaluate():
    assert score(1, 1, 1) == pytest.approx(1.0)
    assert score(0.5, 0.4, 0.2) == pytest.approx(0.35 * 0.5 + 0.35 * 0.4 + 0.30 * 0.2)
    gt = np.array([[1, 0], [0, 0]], np.uint8)
    r = evaluate({"AF_1": gt}, {"AF_1": gt}, {"BS_1": gt}, {"BS_1": gt})
    assert r["score"] == pytest.approx(1.0)


def test_missing_prediction_raises():
    with pytest.raises(KeyError):
        af_f1({}, {"a": np.zeros((2, 2), np.uint8)})
    with pytest.raises(ValueError):
        af_f1({"a": np.zeros((3, 3), np.uint8)}, {"a": np.zeros((2, 2), np.uint8)})

