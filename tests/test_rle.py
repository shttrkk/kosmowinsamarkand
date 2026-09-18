from __future__ import annotations

import numpy as np
import pytest

from firemon.rle import rle_decode, rle_encode, validate_rle

from conftest import BASELINE_DIR


def test_known_example():
    mask = np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 1, 0, 0]], np.uint8)
    # row-major, с 1: пиксели 2,3 | 5 | 8,9,10 (серия переходит через конец строки)
    assert rle_encode(mask) == "2 2 5 1 8 3"
    assert np.array_equal(rle_decode("2 2 5 1 8 3", mask.shape), mask)


def test_roundtrip_random_masks():
    rng = np.random.RandomState(42)
    for i in range(200):
        h, w = rng.randint(1, 70, size=2)
        density = rng.choice([0.01, 0.1, 0.5, 0.9, 0.99])
        mask = (rng.rand(h, w) < density).astype(np.uint8)
        rle = rle_encode(mask)
        validate_rle(rle, (h, w))
        assert np.array_equal(rle_decode(rle, (h, w)), mask), f"mask #{i} {h}x{w}"


def test_bool_and_uint8_equal():
    rng = np.random.RandomState(0)
    mask = rng.rand(32, 32) < 0.3
    assert rle_encode(mask) == rle_encode(mask.astype(np.uint8)) == rle_encode(mask.astype(np.uint8) * 255)


@pytest.mark.parametrize("shape", [(5, 7), (256, 256), (1, 9), (9, 1)])
def test_edge_cases(shape):
    h, w = shape
    empty = np.zeros(shape, np.uint8)
    assert rle_encode(empty) == ""
    assert np.array_equal(rle_decode("", shape), empty)

    full = np.ones(shape, np.uint8)
    assert rle_encode(full) == f"1 {h * w}"
    assert np.array_equal(rle_decode(f"1 {h * w}", shape), full)

    last = empty.copy()
    last[-1, -1] = 1
    assert rle_encode(last) == f"{h * w} 1"
    assert np.array_equal(rle_decode(f"{h * w} 1", shape), last)

    first = empty.copy()
    first[0, 0] = 1
    assert rle_encode(first) == "1 1"

    row = empty.copy()
    row[h // 2, :] = 1
    assert rle_encode(row) == f"{(h // 2) * w + 1} {w}"


@pytest.mark.parametrize("rle, fragment", [
    ("1", "нечётное"),
    ("0 3", "start < 1"),
    ("1 0", "length < 1"),
    ("-2 3", "start < 1"),
    ("1 2.5", "нецелое"),
    ("a b", "нецелое"),
    ("10 5 3 2", "не возрастает"),
    ("1 5 4 3", "пересекаются"),
    ("1 3 4 2", "соприкасаются"),
    ("30 8", "выходит за пределы"),
    ("37 1", "выходит за пределы"),
])
def test_validate_rejects(rle, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_rle(rle, (6, 6))


def test_validate_rejects_non_string():
    with pytest.raises(ValueError):
        validate_rle(float("nan"), (6, 6))


def test_validate_accepts():
    validate_rle("", (6, 6))
    validate_rle("1 36", (6, 6))
    validate_rle("1 1 3 1 36 1", (6, 6))


def test_official_rle_compat():
    """Посимвольное совпадение с официальным кодеком организаторов (50 случайных масок)."""
    if not any(BASELINE_DIR.glob("*.py")):
        pytest.skip("baseline/ с официальным RLE-скриптом ещё не выдан — снять skip в задаче 1b")
    pytest.fail("baseline появился: подключите официальный encode/decode в этот тест (задача 1b)")
