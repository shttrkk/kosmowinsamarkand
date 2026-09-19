"""Фильтр связных компонент гари вместо геометрического правила event_zero_r (ветка task/bs-components).

    python scripts/bs_components.py --step 0     # потолок: оракул «оставить компоненты с долей эталона > 0.5»
    python scripts/bs_components.py --step 1     # признаки компонент → LightGBM (групповые фолды / temporal) → подбор порога

Маска-кандидат: декодирование финального стека (p_burn > t_burn) + запретные маски SCL/воды + MMU 25 px, БЕЗ event_*.
После фильтра компонент — остальная постобработка (дыры < 32 px, приор непустого чипа).
Результаты → reports/bs_components.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.features.bs_features import RULE_THR, RULE_THR_DEFAULT, _median  # noqa: E402
from firemon.models.bs.common import decode_probs, evaluate_masks, fmt, get_splits, load_chips  # noqa: E402
from firemon.models.bs.postprocess import cloud_mask, event_distance, postprocess_bs  # noqa: E402
from firemon.models.bs.rules import nbr  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
S8 = np.ones((3, 3), bool)
OUT = ROOT / "reports" / "bs_components.json"


def load_setup(data_dir, probs_tag, ens_name):
    ens = json.loads((OOF / ens_name).read_text(encoding="utf-8"))
    post = json.loads((ROOT / "weights" / "bs_decode.json").read_text(encoding="utf-8"))["postprocess"]
    decs = {"group": ens["final"]["decode_group"], "temporal": ens["final"]["decode_temporal"]}
    chips = load_chips(data_dir)
    ids = [c["chip_id"] for c in chips]
    splits = get_splits(ids)
    probs = {s: np.load(OOF / f"{probs_tag}_{s}.npy", mmap_mode="r") for s in ("group", "temporal")}
    return chips, ids, splits, probs, decs, post


def split_post(post: dict) -> tuple[dict, dict]:
    """(до фильтра компонент, после): event_* убираются — их заменяет фильтр."""
    pre = {k: v for k, v in post.items() if k in ("scl_post_zero", "scl_zero", "water_lc_zero", "builtup_zero",
                                                    "min_component_px")}
    rest = {k: v for k, v in post.items() if k in ("fill_holes_px", "median_size", "nonempty_min_px")}
    return pre, rest


def candidate(chip, p, dec, pre):
    m = decode_probs(p, dec["t_burn"], dec["w"])
    return postprocess_bs(m, chip, pre, probs=p, decode=dec)


def finish(mask, chip, p, dec, rest):
    return postprocess_bs(mask, chip, rest, probs=p, decode=dec)


def evaluate_sets(preds_by_split: dict, chips, idx: dict) -> dict:
    out = {}
    for split, preds in preds_by_split.items():
        out[split] = evaluate_masks(preds, {i: chips[i]["mask"] for i in idx[split]})
    cl = [i for i in idx["group"] if float(chips[i]["meta"].get("cloud_frac") or 0) > 0.25]
    out["cloudy"] = evaluate_masks({i: preds_by_split["group"][i] for i in cl}, {i: chips[i]["mask"] for i in cl})
    return out


# ───────────────────────── шаг 0: потолок ─────────────────────────

def step0(args) -> dict:
    chips, ids, splits, probs, decs, post = load_setup(args.data_dir, args.probs, args.ensemble)
    pre, rest = split_post(post)
    idx = {"group": np.arange(len(ids)), "temporal": splits["temporal"][0][1]}
    cur, nofilter, oracle = {}, {}, {}
    stats = {"n_components": 0, "n_keep": 0, "area": 0, "area_keep": 0, "mixed_area": 0}
    for split in ("group", "temporal"):
        cur[split], nofilter[split], oracle[split] = {}, {}, {}
        for i in idx[split]:
            p = np.asarray(probs[split][i], np.float32)
            dec, chip, gt = decs[split], chips[i], chips[i]["mask"]
            cur[split][i] = postprocess_bs(decode_probs(p, dec["t_burn"], dec["w"]), chip, post, probs=p, decode=dec)
            cand = candidate(chip, p, dec, pre)
            nofilter[split][i] = finish(cand, chip, p, dec, rest)
            lab, n = ndi.label(cand > 0, structure=S8)
            if n:
                size = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
                inside = np.bincount(lab.ravel(), weights=(gt > 0).ravel().astype(float), minlength=n + 1)
                frac = inside / np.maximum(size, 1)
                keep = frac > 0.5
                keep[0] = False
                cand = np.where(keep[lab], cand, 0).astype(np.uint8)
                if split == "group":
                    stats["n_components"] += n
                    stats["n_keep"] += int(keep.sum())
                    stats["area"] += int(size[1:].sum())
                    stats["area_keep"] += int(size[keep].sum())
                    stats["mixed_area"] += int(size[1:][(frac[1:] > 0.1) & (frac[1:] < 0.9)].sum())
            oracle[split][i] = finish(cand, chip, p, dec, rest)
    res = {"current_rule": evaluate_sets(cur, chips, idx), "no_event_filter": evaluate_sets(nofilter, chips, idx),
           "oracle_components": evaluate_sets(oracle, chips, idx), "component_stats_group": stats}
    for name in ("current_rule", "no_event_filter", "oracle_components"):
        for s in ("group", "temporal", "cloudy"):
            print(f"{name:20s} {s:9s} {fmt(res[name][s])}")
    res["ceiling_gain"] = {s: res["oracle_components"][s]["bs_part"] - res["current_rule"][s]["bs_part"]
                           for s in ("group", "temporal", "cloudy")}
    print("потолок (оракул − текущее правило):", {k: round(v, 4) for k, v in res["ceiling_gain"].items()})
    print("компоненты (group):", stats)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--probs", default="stack_avg14nt_event")
    ap.add_argument("--ensemble", default="ensemble_r34_cedice_main_avg14nt_event.json")
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if args.step == 0:
        prev["step0"] = step0(args)
    else:
        from bs_components_model import step1  # noqa: PLC0415 — тяжёлая часть в отдельном файле

        prev["step1"] = step1(args)
    OUT.write_text(json.dumps(prev, indent=1, default=float, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
