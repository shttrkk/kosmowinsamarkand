"""RLE-кодек буквально по правилам ТЗ (постановка кейса, раздел «Формат submission.csv») на ручных примерах 3×3 и 4×5.

Официального скрипта RLE организаторы не выдают, поэтому ожидаемые строки выписаны
вручную по тексту ТЗ. Нумерация пикселей (с 1, row-major):

    3×3:  1 2 3        4×5:   1  2  3  4  5
          4 5 6               6  7  8  9 10
          7 8 9              11 12 13 14 15
                             16 17 18 19 20
"""
from __future__ import annotations

import numpy as np
import pytest

from firemon.rle import rle_decode, rle_encode, validate_rle

M33 = np.array([[1, 0, 1],
                [1, 0, 0],
                [0, 1, 1]], np.uint8)   # пиксели 1 | 3,4 | 8,9
RLE33 = "1 1 3 2 8 2"

M45 = np.array([[0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1]], np.uint8)  # пиксели 2,3 | 10,11,12 | 18 | 20
RLE45 = "2 2 10 3 18 1 20 1"


@pytest.mark.parametrize("mask, rle", [(M33, RLE33), (M45, RLE45)])
def test_manual_examples(mask, rle):
    assert rle_encode(mask) == rle
    assert np.array_equal(rle_decode(rle, mask.shape), mask)


def test_numbering_starts_from_one():
    m = np.zeros((3, 3), np.uint8)
    m[0, 0] = 1
    assert rle_encode(m) == "1 1"  # не "0 1"
    with pytest.raises(ValueError, match="start < 1"):
        validate_rle("0 1", (3, 3))


def test_row_major_not_column_major():
    m = np.zeros((4, 5), np.uint8)
    m[0, 1] = 1   # строка 0, столбец 1 → пиксель 2 (column-major дало бы 5)
    assert rle_encode(m) == "2 1"
    m[:] = 0
    m[1, 0] = 1   # строка 1, столбец 0 → пиксель 6 = W + 1 (column-major дало бы 2)
    assert rle_encode(m) == "6 1"
    m[:] = 0
    m[:, 2] = 1   # вертикальная линия → 4 отдельные серии с шагом W
    assert rle_encode(m) == "3 1 8 1 13 1 18 1"
    m[:] = 0
    m[2, :] = 1   # горизонтальная линия → одна серия
    assert rle_encode(m) == "11 5"
    # неквадратная маска: транспонирование меняет строку
    assert rle_encode(M45.T) != rle_encode(M45)


def test_starts_strictly_increase():
    for mask in (M33, M45):
        starts = [int(t) for t in rle_encode(mask).split()[0::2]]
        assert starts == sorted(set(starts))
    with pytest.raises(ValueError, match="не возрастает"):
        validate_rle("8 2 1 1", (3, 3))


def test_runs_do_not_overlap():
    with pytest.raises(ValueError, match="пересекаются"):
        validate_rle("1 3 3 2", (3, 3))   # [1..3] и [3..4]
    with pytest.raises(ValueError, match="не возрастает"):
        validate_rle("2 2 2 1", (3, 3))   # одинаковый start


def test_runs_do_not_touch_adjacent_are_merged():
    # серия переходит через конец строки и остаётся ОДНОЙ серией
    m = np.zeros((3, 3), np.uint8)
    m[0, 2] = m[1, 0] = 1
    assert rle_encode(m) == "3 2"          # не "3 1 4 1"
    m = np.zeros((4, 5), np.uint8)
    m[1, 3:] = 1
    m[2, :2] = 1
    assert rle_encode(m) == "9 4"          # не "9 2 11 2"
    with pytest.raises(ValueError, match="соприкасаются"):
        validate_rle("3 1 4 1", (3, 3))
    with pytest.raises(ValueError, match="соприкасаются"):
        validate_rle("9 2 11 2", (4, 5))
    validate_rle("3 1 5 1", (3, 3))        # зазор в один пиксель — допустимо


def test_empty_mask_is_empty_string():
    for shape in ((3, 3), (4, 5)):
        assert rle_encode(np.zeros(shape, np.uint8)) == ""
        validate_rle("", shape)
        assert not rle_decode("", shape).any()
        assert rle_decode("", shape).shape == shape


def test_bounds():
    full33, full45 = np.ones((3, 3), np.uint8), np.ones((4, 5), np.uint8)
    assert rle_encode(full33) == "1 9"
    assert rle_encode(full45) == "1 20"
    last = np.zeros((4, 5), np.uint8)
    last[3, 4] = 1
    assert rle_encode(last) == "20 1"      # последний пиксель = H*W
    validate_rle("9 1", (3, 3))
    validate_rle("19 2", (4, 5))
    for rle, shape in [("10 1", (3, 3)), ("9 2", (3, 3)), ("1 10", (3, 3)), ("21 1", (4, 5)), ("19 3", (4, 5))]:
        with pytest.raises(ValueError, match="выходит за пределы"):
            validate_rle(rle, shape)


def test_format_pairs_single_spaces():
    for mask in (M33, M45):
        rle = rle_encode(mask)
        tokens = rle.split(" ")
        assert len(tokens) % 2 == 0 and all(t.isdigit() for t in tokens)  # только целые, один пробел
        assert rle == rle.strip()
        assert sum(int(t) for t in tokens[1::2]) == int(mask.sum())       # сумма длин = число пикселей
    with pytest.raises(ValueError, match="нечётное"):
        validate_rle("1 1 3", (3, 3))
    with pytest.raises(ValueError, match="length < 1"):
        validate_rle("1 0", (3, 3))


@pytest.mark.parametrize("shape", [(3, 3), (4, 5)])
def test_roundtrip_exhaustive_or_random(shape):
    """3×3 — все 512 масок; 4×5 — 2000 случайных. decode(encode(m)) == m и encode(decode(s)) == s."""
    n = shape[0] * shape[1]
    if n <= 9:
        codes = range(2 ** n)
    else:
        codes = np.random.RandomState(42).randint(0, 2 ** n, size=2000).tolist()
    for code in codes:
        mask = np.array([(code >> i) & 1 for i in range(n)], np.uint8).reshape(shape)
        rle = rle_encode(mask)
        validate_rle(rle, shape)
        decoded = rle_decode(rle, shape)
        assert np.array_equal(decoded, mask)
        assert rle_encode(decoded) == rle
