"""Собирает 5-канальный OOF для стека: 4 канала среднего UNet (avg14nt) + вспомогательный канал (p[1] зоны события).

    python scripts/bs_night_make_aux.py --aux-dir data/cache/bs_weights_night_1/bs_oof --aux-tag zone_v4 --out avg14nt_zone
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "data" / "cache" / "bs_oof"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux-dir", required=True)
    ap.add_argument("--aux-tag", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="avg14nt")
    args = ap.parse_args()
    for s in ("group", "temporal"):
        base = np.load(OOF / f"{args.base}_{s}.npy")
        chans = []
        for tag in args.aux_tag:
            aux = np.load(Path(args.aux_dir) / f"{tag}_{s}.npy")
            chans.append((1.0 - aux[:, 0].astype(np.float32)).astype(np.float16)[:, None])  # P(гарь/зона) = 1 − p0
        out = np.concatenate([base, *chans], axis=1)
        # у temporal заполнены только val-чипы — как и в base
        np.save(OOF / f"{args.out}_{s}.npy", out)
        print(s, out.shape)


if __name__ == "__main__":
    main()
