"""UNet (segmentation_models_pytorch) для BS: обучение по фолдам / временному сплиту / на всём train.

    python -m firemon.models.bs.unet --split group --tag r34_cedice
    python -m firemon.models.bs.unet --split temporal --tag r34_cedice
    python -m firemon.models.bs.unet --split full --tag r34_cedice     # финальный чекпойнт в weights/

OOF-вероятности (float16, (N,4,512,512)) → data/cache/bs_oof/<tag>_<split>[_tta].npy.
Чекпойнт берётся с ПОСЛЕДНЕЙ эпохи (cosine → конец расписания), без выбора эпохи по валидации:
OOF получается честным, а число эпох переносится на обучение на всём train.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...features.bs_features import (CHANNEL_SETS, UNET_CHANNELS, compute_bs_features, feature_names, groups_for,
                                     select_channels)
from ...utils import REPO_ROOT, get_logger, seed_everything
from .common import cache_dir, weights_dir, evaluate_masks, fmt, get_splits, load_chips

log = get_logger("bs.unet")
_BINARY_PREFIX = ("scl_", "lc_", "invalid", "sar_nodata")
CACHE = cache_dir() / "bs_oof"


def unet_inputs(chip: dict, channels: list[str] | None = None) -> np.ndarray:
    """(C, H, W) float32 — ненормированные каналы `channels` (по умолчанию UNET_CHANNELS)."""
    channels = channels or UNET_CHANNELS
    groups = groups_for(channels)
    feats = compute_bs_features(chip, groups=groups)
    return select_channels(feats, feature_names(groups), channels)


def build_unet(encoder: str, in_channels: int, pretrained: bool = True, arch: str = "unet"):
    import segmentation_models_pytorch as smp

    cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN}[arch]
    return cls(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
               in_channels=in_channels, classes=4)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, classes=(1, 2, 3), eps: float = 1.0) -> torch.Tensor:
    p = logits.float().softmax(1)
    loss = 0.0
    for k in classes:
        pk, gk = p[:, k], (target == k).float()
        loss = loss + 1.0 - (2.0 * (pk * gk).sum() + eps) / (pk.sum() + gk.sum() + eps)
    return loss / len(classes)


def _weighted_onehot(target: torch.Tensor, weight: torch.Tensor, dtype) -> torch.Tensor:
    """(B,4,H,W): вес класса на месте эталонного класса, 0 для ignore (255)."""
    oh = F.one_hot(target.clamp(max=3), 4).permute(0, 3, 1, 2).to(dtype) * (target != 255).unsqueeze(1)
    return oh * weight.view(1, -1, 1, 1)


def ce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Взвешенная CE через one-hot — то же значение, что F.cross_entropy(weight=…), но ДЕТЕРМИНИРОВАННЫЙ backward:
    у штатного nll_loss2d на CUDA обратный проход недетерминирован, из-за него два обучения с одним seed расходились
    (BS-часть OOF ±0.01). С этой реализацией веса двух прогонов побитно равны (reports/bs_module.md, «Воспроизводимость»)."""
    logp = F.log_softmax(logits.float(), 1)
    wmap = _weighted_onehot(target, weight, logp.dtype)
    return -(wmap * logp).sum() / wmap.sum().clamp_min(1e-6)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), 1)
    wmap = _weighted_onehot(target, weight, logp.dtype)
    ce = -(wmap * logp).sum(1)
    pt = (F.one_hot(target.clamp(max=3), 4).permute(0, 3, 1, 2).to(logp.dtype) * logp).sum(1).exp()
    return ((1 - pt) ** gamma * ce).mean()


def make_loss(name: str, class_weights: torch.Tensor):
    def ce_dice(lo, t):
        return ce_loss(lo, t, class_weights) + dice_loss(lo, t)

    def focal_dice(lo, t):
        return focal_loss(lo, t, class_weights) + dice_loss(lo, t)

    def ordinal(lo, t):
        # каналы 1..3 — логиты «степень ≥ k»; канал 0 не используется
        tgt = torch.stack([(t >= k).float() for k in (1, 2, 3)], 1)
        return F.binary_cross_entropy_with_logits(lo[:, 1:].float(), tgt)

    return {"ce_dice": ce_dice, "focal_dice": focal_dice, "ordinal": ordinal}[name]


def logits_to_probs(logits: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name != "ordinal":
        return logits.float().softmax(1)
    c = torch.sigmoid(logits[:, 1:].float())  # P(≥1), P(≥2), P(≥3)
    c = torch.cummin(c, dim=1).values         # монотонность
    return torch.stack([1 - c[:, 0], c[:, 0] - c[:, 1], c[:, 1] - c[:, 2], c[:, 2]], 1)


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, loss_name: str = "ce_dice", tta: bool = False) -> torch.Tensor:
    """x (B,C,H,W) на устройстве модели → вероятности (B,4,H,W) float32. TTA: 4 отражения."""
    flips = [(), (3,), (2,), (2, 3)] if tta else [()]
    out = 0.0
    for fl in flips:
        xx = torch.flip(x, fl) if fl else x
        p = logits_to_probs(model(xx), loss_name)
        out = out + (torch.flip(p, fl) if fl else p)
    return out / len(flips)


def _augment(x: torch.Tensor, y: torch.Tensor, gen: torch.Generator):
    r = torch.randint(0, 8, (1,), generator=gen).item()
    if r & 1:
        x, y = x.flip(3), y.flip(2)
    if r & 2:
        x, y = x.flip(2), y.flip(1)
    if r & 4:
        x, y = x.transpose(2, 3), y.transpose(1, 2)
    return x.contiguous(), y.contiguous()


def train_one(x: torch.Tensor, y: torch.Tensor, tr: np.ndarray, va: np.ndarray, args, name: str, norm: dict):
    """x (N,C,H,W) float16 CPU pinned, y (N,H,W) uint8. Возвращает (model, val_probs, val_probs_tta, history)."""
    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(args.encoder, x.shape[1], arch=args.arch).to(dev)
    nz = Normalizer(norm, dev)
    cw = torch.tensor(args.class_weights, dtype=torch.float32, device=dev)
    loss_fn = make_loss(args.loss, cw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos")
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    hist = []
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.permutation(tr)
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            if len(b) < 2:
                continue
            xb = nz(x[b].to(dev, non_blocking=True))
            yb = y[b].to(dev, non_blocking=True).long()
            xb, yb = _augment(xb, yb, gen)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                lo = model(xb)
            loss = loss_fn(lo, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(b)
        rec = {"epoch": ep + 1, "loss": tot / len(tr), "sec": time.time() - t0}
        if len(va) and ((ep + 1) % args.val_every == 0 or ep + 1 == args.epochs):
            probs = _predict_set(model, x, va, dev, nz, args.loss, tta=False)
            m = evaluate_masks({i: p.argmax(0).astype(np.uint8) for i, p in zip(va, probs)},
                               {i: y[i].numpy() for i in va})
            rec.update(iou_burn=m["iou_burn"], miou_sev=m["miou_sev"])
        hist.append(rec)
        log.info("%s ep %d/%d loss %.4f %s (%.0f с)", name, ep + 1, args.epochs, rec["loss"],
                 f"val IoU_burn {rec['iou_burn']:.4f} mIoU_sev {rec['miou_sev']:.4f}" if "iou_burn" in rec else "",
                 rec["sec"])
    vp = vpt = None
    if len(va):
        vp = _predict_set(model, x, va, dev, nz, args.loss, tta=False)
        vpt = _predict_set(model, x, va, dev, nz, args.loss, tta=True)
    return model, vp, vpt, hist


def _predict_set(model, x, idx, dev, nz, loss_name, tta, batch: int = 8) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        xb = nz(x[idx[i:i + batch]].to(dev))
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = predict_probs(model, xb, loss_name, tta)
        out.append(p.half().cpu().numpy())
    return np.concatenate(out)


def save_checkpoint(model, path: Path, args, norm: dict) -> None:
    """fp16-веса (resnet34-UNet ≈ 47 МБ) + всё нужное для сборки модели на инференсе."""
    state = {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "encoder": args.encoder, "arch": args.arch, "loss": args.loss,
                "channels": norm["channels"], "norm": norm}, path)


def prepare_tensors(data_dir: str | Path, norm_path: Path | None = None, channels: list[str] | None = None):
    """Сырые каналы float16 (N,C,H,W) в pinned-памяти + статистика нормализации (применяется на GPU)."""
    from concurrent.futures import ThreadPoolExecutor

    chips = load_chips(data_dir)
    ids = [c["chip_id"] for c in chips]
    t0 = time.time()
    channels = list(channels or UNET_CHANNELS)
    x = np.empty((len(chips), len(channels), 512, 512), np.float16)
    s1 = np.zeros(len(channels), np.float64)
    s2 = np.zeros_like(s1)

    def work(i: int):
        a = np.clip(unet_inputs(chips[i], channels), -60000, 60000)
        x[i] = a
        return a.sum(axis=(1, 2), dtype=np.float64), (a.astype(np.float64) ** 2).sum(axis=(1, 2))

    with ThreadPoolExecutor(8) as ex:
        for a, b in ex.map(work, range(len(chips))):  # порядок фиксирован → суммы детерминированы
            s1 += a
            s2 += b
    y = np.stack([c["mask"] for c in chips])
    n = len(chips) * 512 * 512
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    for i, name in enumerate(channels):
        if name.startswith(_BINARY_PREFIX):
            mean[i], std[i] = 0.0, 1.0
    norm = {"channels": channels, "mean": [float(v) for v in mean], "std": [float(v) for v in std]}
    if norm_path is not None:
        norm_path.write_text(json.dumps(norm, indent=1), encoding="utf-8")
    log.info("входы UNet: %s за %.0f с", x.shape, time.time() - t0)
    xt = torch.from_numpy(x)
    if torch.cuda.is_available():
        try:
            xt = xt.pin_memory()
        except RuntimeError:  # 5.5 ГБ pinned может не выделиться, если память занята — работаем без pin
            log.warning("pin_memory не удался — обычная память")
    return ids, xt, torch.from_numpy(y), norm


class Normalizer:
    def __init__(self, norm: dict, dev):
        self.mean = torch.tensor(norm["mean"], dtype=torch.float32, device=dev).view(1, -1, 1, 1)
        self.std = torch.tensor(norm["std"], dtype=torch.float32, device=dev).view(1, -1, 1, 1)

    def __call__(self, xb: torch.Tensor) -> torch.Tensor:
        return (xb.float() - self.mean) / self.std


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "train"))
    ap.add_argument("--split", choices=["group", "temporal", "full"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--loss", default="ce_dice", choices=["ce_dice", "focal_dice", "ordinal"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--class-weights", type=float, nargs=4, default=[0.5, 1.0, 1.0, 1.2])
    ap.add_argument("--channels", default="v1", choices=sorted(CHANNEL_SETS))
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="только эти фолды (отладка)")
    args = ap.parse_args(argv)

    CACHE.mkdir(parents=True, exist_ok=True)
    weights = weights_dir()
    weights.mkdir(parents=True, exist_ok=True)
    ids, x, y, norm = prepare_tensors(args.data_dir, weights / f"bs_norm_{args.tag}.json" if args.split == "full" else None,
                                      CHANNEL_SETS[args.channels])
    result: dict = {"tag": args.tag, "split": args.split, "args": vars(args)}

    if args.split == "full":
        model, _, _, hist = train_one(x, y, np.arange(len(ids)), np.array([], int), args, "full", norm)
        save_checkpoint(model, weights / f"bs_unet_{args.tag}.pt", args, norm)
        result["history"] = [hist]
    else:
        folds = get_splits(ids)[args.split]
        oof = np.zeros((len(ids), 4, 512, 512), np.float16)
        oof_tta = np.zeros_like(oof)
        done = np.zeros(len(ids), bool)
        hists = []
        for k, (tr, va) in enumerate(folds):
            if args.folds is not None and k not in args.folds:
                continue
            model, vp, vpt, hist = train_one(x, y, tr, va, args, f"{args.split}{k}", norm)
            oof[va], oof_tta[va], done[va] = vp, vpt, True
            hists.append(hist)
            save_checkpoint(model, CACHE / f"{args.tag}_{args.split}{k}.pt", args, norm)
            del model
            torch.cuda.empty_cache()
        np.save(CACHE / f"{args.tag}_{args.split}.npy", oof)
        np.save(CACHE / f"{args.tag}_{args.split}_tta.npy", oof_tta)
        yn = y.numpy()
        val = np.nonzero(done)[0]
        for name, arr in (("argmax", oof), ("argmax+TTA", oof_tta)):
            m = evaluate_masks({i: arr[i].argmax(0).astype(np.uint8) for i in val}, {i: yn[i] for i in val})
            result[name] = m
            log.info("%s %s %s: %s", args.tag, args.split, name, fmt(m))
        result["history"] = hists
    (CACHE / f"{args.tag}_{args.split}.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
