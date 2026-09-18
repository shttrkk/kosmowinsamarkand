"""Устойчивость inference.py: сломанные и отсутствующие чипы, коды выхода, откат бэкенда, реестр моделей."""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import rasterio

from firemon import pipeline
from firemon.io import AF_LAYERS, BS_LAYERS, chip_path, win_path
from firemon.pipeline import RunContext, run_backend
from firemon.rle import rle_decode
from firemon.submission import read_submission, validate_submission

from conftest import ROOT

ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
AF_BANDS = {"viirs": 8, "aux": 5}
BS_BANDS = {"s2_pre": (10, "uint16"), "s2_post": (10, "uint16"), "s1_pre": (2, "int16"), "s1_post": (2, "int16"),
            "aux": (3, "int16")}


def _write_tif(path, arr: np.ndarray) -> None:
    os.makedirs(win_path(path.parent), exist_ok=True)  # `aux` — зарезервированное имя в Windows
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=str(arr.dtype), compress="deflate") as ds:
        ds.write(arr)


def _af_chip(root, cid: str, fill: float, valid: float = 1.0) -> None:
    for key, n in AF_BANDS.items():
        arr = np.full((n, 256, 256), fill, np.float32)
        if key == "viirs":
            arr[7] = valid
        _write_tif(chip_path(root, "af", AF_LAYERS[key], cid), arr)


def _bs_chip(root, cid: str, fill: int) -> None:
    for key, (n, dtype) in BS_BANDS.items():
        _write_tif(chip_path(root, "bs", BS_LAYERS[key], cid), np.full((n, 512, 512), fill, dtype))


@pytest.fixture(scope="module")
def broken_dir(tmp_path_factory):
    """Тест из «сломанных» чипов: NaN везде, нули везде, valid=0, постоянное значение, отсутствующий на диске чип."""
    root = tmp_path_factory.mktemp("broken")
    _af_chip(root, "AF_te_000001", np.nan)              # NaN везде
    _af_chip(root, "AF_te_000002", 0.0)                 # нули везде
    _af_chip(root, "AF_te_000003", 400.0, valid=0.0)    # «горячо», но valid = 0
    _bs_chip(root, "BS_te_000001", 0)                   # нули везде (SCL = 0 → нет данных)
    _bs_chip(root, "BS_te_000002", 3000)                # константа: dNBR = 0
    rows = ["chip_id,class_id,rle", "AF_te_000001,1,", "AF_te_000002,1,", "AF_te_000003,1,", "AF_te_000404,1,"]
    rows += [f"{c},{k}," for c in ("BS_te_000001", "BS_te_000002", "BS_te_000404") for k in (1, 2, 3)]
    (root / "sample_submission.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "af" / "viirs" / "junk.txt").write_text("лишний файл", encoding="utf-8")
    return root


def _run(*args: str):
    return subprocess.run([sys.executable, str(ROOT / "inference.py"), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, env=ENV, timeout=300)


def test_broken_and_missing_chips_give_empty_masks(broken_dir, tmp_path):
    out = tmp_path / "sub.csv"
    proc = _run("--data-dir", str(broken_dir), "--output", str(out), "--profile", "--device", "cpu", "--workers", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stdout + proc.stderr
    assert "AF_te_000404" in proc.stdout and "BS_te_000404" in proc.stdout  # warning про отсутствующие файлы
    assert "ПОЛНОЕ ВРЕМЯ" in proc.stdout                                     # таблица --profile
    sub = read_submission(out)
    assert len(sub["chip_id"]) == 4 + 9 and set(sub["rle"]) == {""}          # все маски пустые
    assert validate_submission(out, broken_dir / "sample_submission.csv") == []


@pytest.mark.parametrize("args, fragment", [
    (["--data-dir", "no/such/dir"], "каталог не найден"),
    (["--data-dir", "{empty}"], "не найдено ни одного чипа"),  # шаблон найдётся в baseline/, чипов нет
    (["--data-dir", "{empty}", "--sample-submission", "no_such.csv"], "файл не найден"),
    (["--data-dir", "{empty}", "--config", "no_such.yaml"], "файл не найден"),
])
def test_missing_inputs_exit_code_and_message(tmp_path, args, fragment):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _run(*[a.replace("{empty}", str(empty)) for a in args], "--output", str(tmp_path / "s.csv"))
    assert proc.returncode == 2
    assert fragment in proc.stdout and "Traceback" not in proc.stdout + proc.stderr


def test_registry_has_configured_backends():
    from firemon.utils import load_config

    cfg = load_config(ROOT / "configs" / "inference.yaml")
    for kind in ("af", "bs"):
        assert cfg[kind]["model"] in pipeline.BACKENDS[kind] and cfg[kind]["fallback"] in pipeline.BACKENDS[kind]
    with pytest.raises(ValueError, match="нет такого бэкенда"):
        run_backend("bs", ROOT, ["BS_te_000001"], {"model": "no_such_model"}, RunContext())


def test_backend_falls_back_when_weights_missing(broken_dir, tmp_path):
    """Нет весов AF → контекстное правило, а не падение; новый BS-бэкенд подключается одной записью в реестре."""
    ctx = RunContext(workers=2)
    ids = ["AF_te_000001", "AF_te_000002"]
    preds, failed, used = run_backend("af", broken_dir, ids, {"model": "lgbm", "fallback": "rules",
                                                              "weights_dir": str(tmp_path / "no_weights")}, ctx)
    assert used == "rules" and failed == 0 and all(preds[c].shape == (256, 256) and not preds[c].any() for c in ids)


def test_new_bs_backend_plugs_in_via_registry(broken_dir, monkeypatch):
    def fake_backend(data_dir, chip_ids, cfg, ctx):
        return {c: np.full((512, 512), 2, np.uint8) for c in chip_ids}, 0

    monkeypatch.setitem(pipeline.BS_BACKENDS, "fake", fake_backend)
    preds, failed, used = run_backend("bs", broken_dir, ["BS_te_000001"], {"model": "fake"}, RunContext())
    assert used == "fake" and failed == 0 and (preds["BS_te_000001"] == 2).all()


def test_without_template_rows_come_from_chip_list(broken_dir, tmp_path):
    """Нет шаблона → состав чипов по meta.csv, а без него — по именам файлов первого слоя; с шаблоном — ровно шаблон."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("inference_cli", ROOT / "inference.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    chips = cli.chips_by_kind(broken_dir, None)
    assert {"AF_te_000001", "AF_te_000002", "AF_te_000003"} <= set(chips["af"]) and "AF_te_000404" not in chips["af"]
    assert {"BS_te_000001", "BS_te_000002"} <= set(chips["bs"])
    chips = cli.chips_by_kind(broken_dir, broken_dir / "sample_submission.csv")
    assert chips["af"][-1] == "AF_te_000404" and chips["bs"] == ["BS_te_000001", "BS_te_000002", "BS_te_000404"]


def test_stub_prediction_roundtrip(broken_dir, tmp_path):
    """Непустая маска проходит путь backend → RLE → csv → decode без потерь (на константном BS-чипе с «гарью»)."""
    cid = "BS_te_000777"
    _bs_chip(broken_dir, cid, 3000)
    path = chip_path(broken_dir, "bs", BS_LAYERS["s2_post"], cid)
    post = np.full((10, 512, 512), 3000, np.uint16)
    post[6, 100:200, 50:300] = 1000   # B8A упал, B12 вырос → dNBR > 0.44
    post[8, 100:200, 50:300] = 5000
    _write_tif(path, post)
    sample = tmp_path / "sample.csv"
    sample.write_text("chip_id,class_id,rle\n" + "".join(f"{cid},{k},\n" for k in (1, 2, 3)), encoding="utf-8")
    out = tmp_path / "s.csv"
    assert _run("--data-dir", str(broken_dir), "--output", str(out), "--sample-submission", str(sample)).returncode == 0
    sub = read_submission(out)
    masks = {int(k): rle_decode(r, (512, 512)) for k, r in zip(sub["class_id"], sub["rle"])}
    assert masks[3][100:200, 50:300].all() and masks[3].sum() == 100 * 250 and not masks[1].any() and not masks[2].any()
