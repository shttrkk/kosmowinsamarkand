"""Общий каркас ночных экспериментов BS по кэшированным OOF: загрузка, оценка на group / temporal / облачных,
запись результата в reports/bs_night.json и строки в reports/experiments.md (нумерация с BS-109)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.models.bs.common import (decode_probs, evaluate_masks, fit_decode, get_splits, load_chips,  # noqa: E402
                                      log_experiment)
from firemon.models.bs.postprocess import postprocess_bs  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
NIGHT = ROOT / "reports" / "bs_night.json"
VAL = "GroupKFold-5 (`folds_bs.csv`) / temporal val=2024"
BASE = {"group": 0.4912, "temporal": 0.5008, "cloudy": 0.4401}


class Setup:
    """Финальная модель ветки task/bs: стек avg14nt_event, декодирование и постобработка из weights/bs_decode.json."""

    def __init__(self, probs_tag: str = "stack_avg14nt_event", oof_dir: Path = OOF, fit: bool = False):
        self.chips = load_chips(ROOT / "data" / "train")
        self.ids = [c["chip_id"] for c in self.chips]
        self.gts = np.stack([c["mask"] for c in self.chips])
        self.splits = get_splits(self.ids)
        self.t_tr, self.t_va = self.splits["temporal"][0]
        self.idx = {"group": np.arange(len(self.ids)), "temporal": self.t_va}
        self.cloudy = [i for i in self.idx["group"] if float(self.chips[i]["meta"].get("cloud_frac") or 0) > 0.25]
        self.post = json.loads((ROOT / "weights" / "bs_decode.json").read_text(encoding="utf-8"))["postprocess"]
        self.load_probs(probs_tag, oof_dir, fit)

    def load_probs(self, tag: str, oof_dir: Path = OOF, fit: bool = False) -> None:
        self.probs = {s: np.load(Path(oof_dir) / f"{tag}_{s}.npy", mmap_mode="r") for s in ("group", "temporal")}
        if fit:
            self.decs = {"group": fit_decode(self.probs["group"][:], self.gts),
                         "temporal": fit_decode(self.probs["group"][self.t_tr], self.gts[self.t_tr])}
        else:
            ens = json.loads((OOF / "ensemble_r34_cedice_main_avg14nt_event.json").read_text(encoding="utf-8"))
            self.decs = {"group": ens["final"]["decode_group"], "temporal": ens["final"]["decode_temporal"]}
        self._cache: dict = {}

    def p(self, split: str, i: int) -> np.ndarray:
        key = (split, int(i))
        if key not in self._cache:
            self._cache[key] = np.asarray(self.probs[split][i], np.float32)
        return self._cache[key]

    def evaluate(self, fn) -> dict:
        """fn(split, i, probs(4,H,W), dec, chip) → маска 0..3. Метрики group / temporal / cloudy."""
        out = {}
        for s in ("group", "temporal"):
            preds = {i: fn(s, i, self.p(s, i), self.decs[s], self.chips[i]) for i in self.idx[s]}
            out[s] = evaluate_masks(preds, {i: self.gts[i] for i in self.idx[s]})
            if s == "group":
                out["cloudy"] = evaluate_masks({i: preds[i] for i in self.cloudy}, {i: self.gts[i] for i in self.cloudy})
        return out

    def baseline_fn(self, post: dict | None = None):
        post = self.post if post is None else post

        def fn(s, i, p, dec, chip):
            return postprocess_bs(decode_probs(p, dec["t_burn"], dec["w"]), chip, post, probs=p, decode=dec)
        return fn


def gains(m: dict, base: dict) -> dict:
    return {s: m[s]["bs_part"] - base[s]["bs_part"] for s in ("group", "temporal", "cloudy")}


def accepted(g: dict, cloudy_min: float | None = None) -> bool:
    ok = g["group"] >= 0.005 and g["temporal"] >= 0.005
    return bool(ok and (cloudy_min is None or g["cloudy"] >= cloudy_min))


def line(name: str, m: dict, g: dict | None = None) -> str:
    def c(s):
        d = f" ({g[s]:+.4f})" if g else ""
        return f"{m[s]['iou_burn']:.4f}/{m[s]['miou_sev']:.4f}/{m[s]['bs_part']:.4f}{d}"
    return f"{name:46s} | {c('group')} | {c('temporal')} | {c('cloudy')}"


def record(key: str, title: str, config: str, best: dict, g: dict, ok: bool, why: str, extra: dict | None = None,
           inference_time: str = "—") -> None:
    """→ reports/bs_night.json[key] и строка BS-N в experiments.md (один раз на key)."""
    data = json.loads(NIGHT.read_text(encoding="utf-8")) if NIGHT.exists() else {}
    data[key] = {"title": title, "metrics": best, "gain": g, "accepted": ok, "why": why, **(extra or {})}
    NIGHT.write_text(json.dumps(data, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    path = ROOT / "reports" / "experiments.md"
    mark = f"[bs:{key}]"
    if mark in path.read_text(encoding="utf-8"):
        return
    cl = best["cloudy"]
    log_experiment(f"{title}; облачные чипы (>0.25): {cl['iou_burn']:.4f} / {cl['miou_sev']:.4f} / BS-часть {cl['bs_part']:.4f} {mark}",
                   config, VAL, best["group"], best["temporal"], inference_time,
                   f"Δ к финалу (0.4912 / 0.5008 / обл. 0.4401): {g['group']:+.4f} / {g['temporal']:+.4f} / {g['cloudy']:+.4f} → "
                   f"{'ПРИНЯТО' if ok else 'не принято'}. {why}")
    text = path.read_text(encoding="utf-8")
    nums = [int(x) for x in re.findall(r"^\| BS-(\d+) \|", text, flags=re.M)]
    if max(nums) < 109:  # BS-105…108 заняты ветками ens4 и components
        text = text.replace(f"| BS-{max(nums)} |", "| BS-109 |")
        path.write_text(text, encoding="utf-8", newline="\n")
