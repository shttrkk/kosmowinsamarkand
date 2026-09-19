"""Идея (g): многомасштабный инференс UNet (0.75× / 1× / 1.25×, усреднение вероятностей) — OOF по сохранённым
фолд-чекпойнтам основной модели (data/cache/bs_oof/<tag>_<split><k>.pt). → data/cache/bs_oof/avg14ms_<split>.npy

    python scripts/bs_night_multiscale.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.features.bs_features import CHANNEL_SETS  # noqa: E402
from firemon.models.bs.common import get_splits  # noqa: E402
from firemon.models.bs.unet import Normalizer, build_unet, predict_probs, prepare_tensors  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
SCALES = (0.75, 1.25)


@torch.no_grad()
def predict_scaled(model, xb, scale: float) -> torch.Tensor:
    size = int(round(512 * scale / 32) * 32)
    xs = F.interpolate(xb, size=(size, size), mode="bilinear", align_corners=False)
    p = predict_probs(model, xs, "ce_dice", tta=False)
    return F.interpolate(p, size=(512, 512), mode="bilinear", align_corners=False)


def main() -> None:
    dev = torch.device("cuda")
    total = {s: 0.0 for s in ("group", "temporal")}
    for tag, ch in (("r34_cedice", "v1"), ("r34_v4", "v4")):
        ids, x, _, _ = prepare_tensors(ROOT / "data" / "train", None, CHANNEL_SETS[ch])
        splits = get_splits(ids)
        for split in ("group", "temporal"):
            base = np.load(OOF / f"{tag}_{split}.npy").astype(np.float32)
            acc = base.copy()
            t0 = time.time()
            for k, (_, va) in enumerate(splits[split]):
                ck = torch.load(OOF / f"{tag}_{split}{k}.pt", map_location="cpu", weights_only=False)
                model = build_unet(ck["encoder"], len(ck["channels"]), pretrained=False, arch=ck.get("arch", "unet"))
                model.load_state_dict(ck["state_dict"])
                model = model.to(dev).float().eval()
                nz = Normalizer(ck["norm"], dev)
                for i in range(0, len(va), 8):
                    b = va[i:i + 8]
                    xb = nz(x[b].to(dev))
                    with torch.autocast("cuda"):
                        for sc in SCALES:
                            acc[b] += predict_scaled(model, xb, sc).float().cpu().numpy()
                acc[va] /= (1 + len(SCALES))
            np.save(OOF / f"{tag}_ms_{split}.npy", acc.astype(np.float16))
            total[split] = total[split] + acc if isinstance(total[split], np.ndarray) else acc.copy()
            print(tag, split, f"{time.time() - t0:.0f} с")
        del x
    for split in ("group", "temporal"):
        np.save(OOF / f"avg14ms_{split}.npy", (total[split] / 2).astype(np.float16))
    print("ok")


if __name__ == "__main__":
    main()
