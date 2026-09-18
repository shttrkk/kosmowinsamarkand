"""Запас пикселей огня до порогов кандидатов: (I4 − T_cand), (I4−I5 − D_cand) и z-score окна 7, день/ночь.
Запуск: python scripts/af/candidate_margin.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_all  # noqa: E402

from firemon.features.af_features import CAND, candidate_mask, night_mask  # noqa: E402
from firemon.models.af.rules import background_z  # noqa: E402

Q = [0, 0.1, 0.5, 1, 5, 25, 50]


def main() -> None:
    meta, chips = load_all()
    rows = {"day": [], "night": []}
    n_cand = miss = 0
    for c in chips:
        v, valid = c["viirs"], c["valid"]
        fire = c["mask"] == 1
        cand = candidate_mask(v, valid)
        n_cand += cand.sum()
        miss += (fire & ~cand).sum()
        if not fire.any():
            continue
        night = night_mask(v)
        i4 = np.where(valid, v[3], 0).astype(np.float32)
        di = np.where(valid, v[3] - v[4], 0).astype(np.float32)
        z4, zd = background_z(i4, valid, 7), background_z(di, valid, 7)
        for p, sel in (("day", fire & ~night), ("night", fire & night)):
            if sel.any():
                rows[p].append(np.c_[i4[sel], di[sel], z4[sel], zd[sel]])
    print(f"пороги кандидатов: {CAND}; кандидатов {n_cand}, огня вне кандидатов {miss}")
    for p in ("day", "night"):
        a = np.concatenate(rows[p])
        print(f"\n{p}: {len(a)} px огня; перцентили {Q}")
        print("  I4 − T_cand     ", np.percentile(a[:, 0] - CAND[f"{p}_i4"], Q).round(2))
        print("  (I4−I5) − D_cand", np.percentile(a[:, 1] - CAND[f"{p}_di"], Q).round(2))
        print("  z7(I4)          ", np.percentile(a[:, 2], Q).round(2))
        print("  z7(I4−I5)       ", np.percentile(a[:, 3], Q).round(2))


if __name__ == "__main__":
    main()
