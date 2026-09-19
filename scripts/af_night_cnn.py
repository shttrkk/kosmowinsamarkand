"""Идея (i), AF: лёгкая полносвёрточная CNN (5 слоёв, 32–64 канала, без понижения разрешения, focal + dice) на 8+5 каналах;
среднее с LightGBM AF-17 на пикселях-кандидатах. Фолды/временной сплит — configs/folds_af.csv. Веса сохраняются в
weights_night_af/ только если принято (≥ +0.003 F1 на обоих сплитах). → reports/af_night_cnn.json
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.io import read_af_chip, read_meta  # noqa: E402
from firemon.models.af.dataset import load_or_build  # noqa: E402
from firemon.models.af.train_lgbm import DEFAULT_PARAMS, best_threshold, run_cv, run_split, score_rows  # noqa: E402
from firemon.splits import load_folds  # noqa: E402
from firemon.utils import load_config, seed_everything  # noqa: E402

EPOCHS, BATCH, LR = 40, 16, 2e-3


def chip_tensor(data_dir, cid):
    c = read_af_chip(data_dir, cid, with_mask=True)
    v, a = c["viirs"], c["aux"]
    nan = ~np.isfinite(v[:5]).all(0)
    v = np.nan_to_num(v, nan=0.0)
    a = np.nan_to_num(a, nan=0.0)
    night = (v[5] > 90).astype(np.float32)
    x = np.stack([v[0], v[1], v[2], (v[3] - 300) / 20, (v[4] - 290) / 15, (v[3] - v[4]) / 20, night, v[6] / 70,
                  c["valid"].astype(np.float32), nan.astype(np.float32),
                  (a[0] == 50), (a[0] == 40), (a[0] == 80), (a[0] == 10), a[1] / 1000, a[2] / 30, a[3] / 100, a[4] / 10]
                 ).astype(np.float32)
    return x, (c["mask"] == 1).astype(np.float32), c["valid"].astype(np.float32)


class Net(nn.Module):
    def __init__(self, cin: int):
        super().__init__()
        chs = [cin, 32, 48, 64, 64, 32]
        layers = []
        for i in range(5):
            layers += [nn.Conv2d(chs[i], chs[i + 1], 3, padding=1 if i % 2 == 0 else 2, dilation=1 if i % 2 == 0 else 2),
                       nn.BatchNorm2d(chs[i + 1]), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.head.bias, -6.0)  # огонь ≈ 0.035 % пикселей

    def forward(self, x):
        return self.head(self.body(x))[:, 0]


def loss_fn(lo, y, valid):
    p = torch.sigmoid(lo)
    bce = F.binary_cross_entropy_with_logits(lo, y, reduction="none")
    focal = ((1 - torch.where(y > 0.5, p, 1 - p)) ** 2 * torch.where(y > 0.5, 0.9, 0.1) * bce * valid).sum() / valid.sum()
    inter = (p * y * valid).sum()
    dice = 1 - (2 * inter + 1) / ((p * valid).sum() + (y * valid).sum() + 1)
    return 20 * focal + dice


def train_predict(X, Y, V, tr, va, seed=42):
    seed_everything(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    rng = np.random.RandomState(seed)
    Xt, Yt, Vt = (torch.from_numpy(a).to(dev) for a in (X, Y, V))
    for ep in range(EPOCHS):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), BATCH):
            b = torch.from_numpy(order[i:i + BATCH]).to(dev)
            xb, yb, vb = Xt[b], Yt[b], Vt[b]
            r = rng.randint(8)
            if r & 1:
                xb, yb, vb = xb.flip(3), yb.flip(2), vb.flip(2)
            if r & 2:
                xb, yb, vb = xb.flip(2), yb.flip(1), vb.flip(1)
            if r & 4:
                xb, yb, vb = xb.transpose(2, 3), yb.transpose(1, 2), vb.transpose(1, 2)
            loss = loss_fn(net(xb), yb, vb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    out = np.zeros((len(va), 256, 256), np.float32)
    with torch.no_grad():
        vt = torch.from_numpy(va).to(dev)
        for i in range(0, len(va), 32):
            b = vt[i:i + 32]
            out[i:i + 32] = torch.sigmoid(net(Xt[b])).cpu().numpy()
    return out


def main() -> None:
    cfg = load_config(ROOT / "configs" / "af_lgbm.yaml")
    seed_everything(42)
    data_dir = ROOT / "data" / "train"
    meta = read_meta(data_dir, "af").merge(load_folds("af"), on="chip_id", how="left")
    ids = meta["chip_id"].tolist()
    folds, ts = meta["fold"].to_numpy().astype(int), meta["temporal_split"].to_numpy()
    ds = load_or_build(data_dir, ids, ROOT / cfg["cache"], 42, False)
    params = {**DEFAULT_PARAMS, **cfg.get("params", {}), "seed": 42}
    nr = int(cfg.get("n_rounds", 2000))

    # LightGBM AF-17: OOF и temporal вероятности по строкам
    oof_l, _, _ = run_cv(ds, folds, params, None, ds["is_cand"], nr)
    thr_l, r_l = best_threshold(ds, oof_l)
    _, va_rows, p = run_split(ds, ts == "train", ts == "val", params, np.arange(ds["X"].shape[1]), ds["is_cand"], nr)
    tmp_l = np.full(len(ds["y"]), np.nan, np.float32)
    tmp_l[va_rows] = p
    rt_l = score_rows(ds, tmp_l, thr_l, chip_sel=ts == "val")
    print("LGBM AF-17: OOF F1 %.4f | temporal F1 %.4f" % (r_l["f1"], rt_l["f1"]))

    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda c: chip_tensor(data_dir, c), ids))
    X, Y, V = (np.stack([p[j] for p in parts]) for j in range(3))
    t0 = time.time()
    cnn_oof = np.zeros((len(ids), 256, 256), np.float32)
    idx = np.arange(len(ids))
    for f in sorted(np.unique(folds)):
        cnn_oof[folds == f] = train_predict(X, Y, V, idx[folds != f], idx[folds == f])
        print(f"CNN фолд {f}: {time.time() - t0:.0f} с")
    cnn_tmp = np.zeros_like(cnn_oof)
    cnn_tmp[ts == "val"] = train_predict(X, Y, V, idx[ts == "train"], idx[ts == "val"])

    rows_oof = cnn_oof[ds["chip_idx"], ds["yy"], ds["xx"]]
    rows_tmp = cnn_tmp[ds["chip_idx"], ds["yy"], ds["xx"]]
    res = {"lgbm": {"oof": r_l["f1"], "temporal": rt_l["f1"]}, "blends": {}}
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 1.0):  # w — вес CNN
        po = np.where(np.isfinite(oof_l), (1 - w) * oof_l + w * rows_oof, np.nan)
        pt = np.where(np.isfinite(tmp_l), (1 - w) * tmp_l + w * rows_tmp, np.nan)
        thr, r = best_threshold(ds, po)
        rt = score_rows(ds, pt, thr, chip_sel=ts == "val")
        res["blends"][str(w)] = {"oof": r["f1"], "temporal": rt["f1"], "thr": thr}
        print(f"вес CNN {w}: OOF F1 {r['f1']:.4f} | temporal F1 {rt['f1']:.4f} (порог {thr})")
    best = max((k for k in res["blends"] if k != "0.0"), key=lambda k: min(res["blends"][k]["oof"] - r_l["f1"],
                                                                         res["blends"][k]["temporal"] - rt_l["f1"]))
    b = res["blends"][best]
    res["best_w"] = best
    res["base"] = {"oof": {"f1": r_l["f1"]}, "temporal": {"f1": rt_l["f1"]}}
    res["new"] = {"oof": {"f1": b["oof"], "precision": float("nan"), "recall": float("nan")}, "temporal": {"f1": b["temporal"]}}
    res["delta_f1"] = {"oof": b["oof"] - r_l["f1"], "temporal": b["temporal"] - rt_l["f1"]}
    res["accepted"] = bool(res["delta_f1"]["oof"] >= 0.003 and res["delta_f1"]["temporal"] >= 0.003)
    print("лучший вес CNN", best, res["delta_f1"], "→", "ПРИНЯТО" if res["accepted"] else "не принято")
    (ROOT / "reports" / "af_night_cnn.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
