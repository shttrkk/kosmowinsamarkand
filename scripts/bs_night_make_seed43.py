"""Проверка устойчивости принятого ночного варианта: база стека с UNet 0.75×, переобученным с seed 43
(вместо seed 42) + доп. канал UNet 0.5×. → data/cache/bs_oof/avg3s075seed43_half_<split>.npy"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "data" / "cache" / "bs_oof"
B = ROOT / "data" / "cache" / "bs_weights_night_3" / "bs_oof"


def f(p):
    return np.load(p).astype(np.float32)


for s in ("group", "temporal"):
    base3 = (f(A / f"r34_cedice_{s}.npy") + f(A / f"r34_v4_{s}.npy") + f(B / f"s075_v4_seed43_{s}.npy")) / 3
    half = (1 - f(B / f"half_v4_{s}.npy")[:, 0])[:, None]
    np.save(A / f"avg3s075seed43_half_{s}.npy", np.concatenate([base3, half], axis=1).astype(np.float16))
    print(s, "ok")
