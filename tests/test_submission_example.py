"""Сверка с примером организаторов `baseline/example_submission.csv`: RLE и текстовый формат файла.

Пример и шаблон лежат в `baseline/` (не в git) — без них тесты пропускаются.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from firemon.rle import rle_decode, rle_encode
from firemon.submission import (DEFAULT_SHAPES, build_submission, chip_kind, read_submission,
                                validate_submission)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "baseline" / "example_submission.csv"
TEMPLATE = ROOT / "baseline" / "sample_submission.csv"

pytestmark = pytest.mark.skipif(not (EXAMPLE.exists() and TEMPLATE.exists()),
                                reason="нет baseline/example_submission.csv или шаблона")


@pytest.fixture(scope="module")
def example():
    return read_submission(EXAMPLE)


def _raw_lines(path: Path) -> list[str]:
    raw = path.read_bytes().decode("utf-8-sig")
    assert raw.endswith("\r\n")
    return raw.split("\r\n")[:-1]


def test_example_af_block_position(example):
    """`32640 3 32896 3 33152 3` — нумерация с 1, row-major: квадрат 3×3 в строках и столбцах 127–129 (0-based)."""
    rles = {c: r for c, k, r in zip(example["chip_id"], example["class_id"], example["rle"]) if chip_kind(c) == "af"}
    assert "32640 3 32896 3 33152 3" in rles.values()
    expected = np.zeros((256, 256), np.uint8)
    expected[127:130, 127:130] = 1
    n = 0
    for cid, rle in rles.items():
        if rle == "32640 3 32896 3 33152 3":
            assert np.array_equal(rle_decode(rle, (256, 256)), expected), cid
            n += 1
    assert n > 0


def test_example_decodes_and_bs_classes_disjoint(example):
    union: dict[str, np.ndarray] = {}
    for cid, cls, rle in zip(example["chip_id"], example["class_id"], example["rle"]):
        mask = rle_decode(rle, DEFAULT_SHAPES[chip_kind(cid)])  # validate=True: любая ошибка формата → ValueError
        if chip_kind(cid) == "bs":
            seen = union.setdefault(cid, np.zeros_like(mask))
            assert not np.any(seen & mask), f"{cid}: класс {cls} пересекается с предыдущими"
            seen |= mask
    assert validate_submission(EXAMPLE, TEMPLATE) == []


def test_example_roundtrip_is_char_exact(example):
    """decode → encode возвращает строку примера посимвольно (порядок серий, слияние смежных, пустая маска = '')."""
    for cid, cls, rle in zip(example["chip_id"], example["class_id"], example["rle"]):
        assert rle_encode(rle_decode(rle, DEFAULT_SHAPES[chip_kind(cid)])) == rle, (cid, cls)


def test_our_file_matches_example_text_format(example, tmp_path):
    """Наш файл, собранный из масок примера, побайтно равен примеру: заголовок, порядок строк шаблона, без кавычек, CRLF."""
    af, bs = {}, {}
    for cid, cls, rle in zip(example["chip_id"], example["class_id"], example["rle"]):
        mask = rle_decode(rle, DEFAULT_SHAPES[chip_kind(cid)])
        if chip_kind(cid) == "af":
            af[cid] = mask
        else:
            bs.setdefault(cid, np.zeros_like(mask))[mask == 1] = int(cls)
    out = tmp_path / "ours.csv"
    assert build_submission(af, bs, TEMPLATE, out) == 447

    ours, theirs, tpl = _raw_lines(out), _raw_lines(EXAMPLE), TEMPLATE.read_text(encoding="utf-8-sig").splitlines()
    assert ours[0] == theirs[0] == tpl[0] == "chip_id,class_id,rle"
    assert len(ours) == len(tpl) == 448
    for line_o, line_t in zip(ours[1:], tpl[1:]):  # построчно: те же chip_id/class_id в том же порядке
        assert line_o.split(",")[:2] == line_t.split(",")[:2]
    assert '"' not in "\n".join(ours)
    assert ours == theirs
    assert out.read_bytes() == EXAMPLE.read_bytes()


def test_reader_accepts_both_styles(tmp_path):
    quoted, plain = tmp_path / "q.csv", tmp_path / "p.csv"
    quoted.write_text('chip_id,class_id,rle\nAF_te_000001,1,"1 3 10 2"\nAF_te_000002,1,""\n', encoding="utf-8")
    plain.write_bytes(b"chip_id,class_id,rle\r\nAF_te_000001,1,1 3 10 2\r\nAF_te_000002,1,\r\n")
    assert read_submission(quoted) == read_submission(plain)
    assert read_submission(plain)["rle"] == ["1 3 10 2", ""]
