from __future__ import annotations

import numpy as np
import pytest

from firemon.rle import rle_decode
from firemon.submission import build_submission, read_submission, validate_submission, write_submission

AF_IDS = ["AF_te_000001", "AF_te_000002"]
BS_IDS = ["BS_te_000001", "BS_te_000002"]
SHAPES = {**{c: (16, 16) for c in AF_IDS}, **{c: (32, 32) for c in BS_IDS}}


def _template_rows():
    return [(c, 1, "") for c in AF_IDS] + [(c, k, "") for c in BS_IDS for k in (1, 2, 3)]


@pytest.fixture()
def template(tmp_path):
    path = tmp_path / "sample_submission.csv"
    write_submission(_template_rows(), path)
    return path


@pytest.fixture()
def preds():
    rng = np.random.RandomState(1)
    af = {c: (rng.rand(16, 16) < 0.1).astype(np.uint8) for c in AF_IDS}
    af[AF_IDS[1]][:] = 0  # пустая маска
    bs = {c: rng.randint(0, 4, size=(32, 32)).astype(np.uint8) for c in BS_IDS}
    return af, bs


@pytest.fixture()
def good(tmp_path, template, preds):
    out = tmp_path / "submission.csv"
    build_submission(*preds, template, out)
    return out


def _rewrite(path, mutate):
    sub = read_submission(path)
    rows = [list(r) for r in zip(sub["chip_id"], sub["class_id"], sub["rle"])]
    rows = mutate(rows)
    write_submission([(c, int(k), r) for c, k, r in rows], path)


def test_build_matches_template_and_roundtrips(good, template, preds):
    af, bs = preds
    assert validate_submission(good, template, SHAPES) == []
    sub, tpl = read_submission(good), read_submission(template)
    assert list(zip(sub["chip_id"], sub["class_id"])) == list(zip(tpl["chip_id"], tpl["class_id"]))
    for cid, cls, rle in zip(sub["chip_id"], sub["class_id"], sub["rle"]):
        src = af[cid] == 1 if cid in af else bs[cid] == int(cls)
        assert np.array_equal(rle_decode(rle, SHAPES[cid]), src.astype(np.uint8))


def test_file_format(good):
    lines = good.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "chip_id,class_id,rle"
    assert len(lines) == 1 + 8
    assert all(line.endswith('"') and line.count('"') == 2 for line in lines[1:])  # rle всегда в кавычках
    assert f'{AF_IDS[1]},1,""' in lines  # пустая маска → ""
    assert "nan" not in good.read_text(encoding="utf-8").lower()


def test_build_strict_missing_chip(tmp_path, template, preds):
    af, bs = preds
    del bs[BS_IDS[0]]
    with pytest.raises(KeyError):
        build_submission(af, bs, template, tmp_path / "s.csv")
    build_submission(af, bs, template, tmp_path / "s.csv", strict=False)
    assert validate_submission(tmp_path / "s.csv", template, SHAPES) == []


def test_detects_extra_row(good, template):
    _rewrite(good, lambda rows: rows + [["AF_te_999999", "1", ""]])
    errors = " | ".join(validate_submission(good, template, SHAPES))
    assert "число строк" in errors and "лишние" in errors


def test_detects_missing_pair(good, template):
    _rewrite(good, lambda rows: rows[:-1])
    errors = " | ".join(validate_submission(good, template, SHAPES))
    assert "число строк" in errors and "пропущено" in errors


def test_detects_duplicate_pair(good, template):
    _rewrite(good, lambda rows: rows[:-1] + [rows[0]])
    errors = " | ".join(validate_submission(good, template, SHAPES))
    assert "дубликаты" in errors and "пропущено" in errors


def test_detects_class_overlap(good, template):
    def mutate(rows):
        for r in rows:
            if r[0] == BS_IDS[0]:
                r[2] = "5 10" if r[1] == "1" else ("10 10" if r[1] == "2" else "")
        return rows
    _rewrite(good, mutate)
    errors = validate_submission(good, template, SHAPES)
    assert len(errors) == 1 and "пересекаются" in errors[0] and BS_IDS[0] in errors[0]


def test_detects_out_of_bounds(good, template):
    def mutate(rows):
        rows[0][2] = "250 10"  # 16*16 = 256
        return rows
    _rewrite(good, mutate)
    errors = validate_submission(good, template, SHAPES)
    assert len(errors) == 1 and "выходит за пределы" in errors[0]


@pytest.mark.parametrize("bad", ["nan", "NaN", "None"])
def test_detects_nan(good, template, bad):
    def mutate(rows):
        rows[1][2] = bad
        return rows
    _rewrite(good, mutate)
    errors = validate_submission(good, template, SHAPES)
    assert len(errors) == 1 and "NaN" in errors[0]


def test_detects_touching_runs_and_bad_header(good, template, tmp_path):
    def mutate(rows):
        rows[0][2] = "1 3 4 2"
        return rows
    _rewrite(good, mutate)
    assert "соприкасаются" in validate_submission(good, template, SHAPES)[0]

    bad = tmp_path / "bad.csv"
    bad.write_text("id,cls,rle\nAF_te_000001,1,\"\"\n", encoding="utf-8")
    assert "заголовок" in validate_submission(bad, template, SHAPES)[0]


def test_row_order_is_free(good, template):
    _rewrite(good, lambda rows: rows[::-1])
    assert validate_submission(good, template, SHAPES) == []
