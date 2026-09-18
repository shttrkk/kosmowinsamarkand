"""H1 (запись) и H3: seed + region growing вместо/поверх event_zero_r (Stroppiana et al. 2021, Remote Sens. 13:2214,
https://www.mdpi.com/2072-4292/13/11/2214). По кэшированным OOF финального стека."""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage as ndi

from bs_night_lib import ROOT, Setup, accepted, gains, line, record
from firemon.models.bs.postprocess import event_center, postprocess_bs

S8 = np.ones((3, 3), bool)


def grow(p: np.ndarray, t_high: float, t_low: float, w, seed_mode: str) -> np.ndarray | None:
    """Маска 0..3: рост от seed гистерезисом. seed_mode: best — одна компонента с max score = mean p_burn·√площади
    (при равенстве — ближайшая к центру события); top3near — ближайшая к центру из трёх лучших; all — все компоненты t_high."""
    pb = 1.0 - p[0]
    lab_h, n = ndi.label(pb > t_high, structure=S8)
    if n == 0:
        return None
    idx = np.arange(1, n + 1)
    if seed_mode == "all":
        seeds = lab_h > 0
    else:
        area = ndi.sum(np.ones_like(pb), lab_h, idx)
        score = ndi.mean(pb, lab_h, idx) * np.sqrt(area)
        cy, cx = event_center(p)
        cents = np.array(ndi.center_of_mass(np.ones_like(pb), lab_h, idx))
        dist = np.hypot(cents[:, 0] - cy, cents[:, 1] - cx)
        if seed_mode == "best":
            k = int(np.lexsort((dist, -score))[0])
        else:
            top = np.argsort(-score)[:3]
            k = int(top[np.argmin(dist[top])])
        seeds = lab_h == k + 1
    lab_l, _ = ndi.label(pb > t_low, structure=S8)
    keep = np.unique(lab_l[seeds])
    keep = keep[keep > 0]
    region = np.isin(lab_l, keep)
    cls = (p[1:] * np.asarray(w, np.float32).reshape(3, 1, 1)).argmax(0).astype(np.uint8) + 1
    return np.where(region, cls, 0).astype(np.uint8)


def main() -> None:
    S = Setup()
    base = S.evaluate(S.baseline_fn())
    print(line("текущий финал", base))

    # ---- H1: запись результата scripts/bs_event_center.py ----
    h1 = json.loads((ROOT / "reports" / "bs_event_center.json").read_text(encoding="utf-8"))
    b = h1["variants"][h1["best"]]
    d = h1["diagnostics"]
    record("h1_box_prior", "H1: рамочный приор на центр события (clip / boxmax / центроид крупнейшей компоненты; R_box 170–260; "
           f"event_zero_r 300/250) — лучший вариант «{h1['best']}»", "`scripts/bs_event_center.py`",
           {k: b[k] for k in ("group", "temporal", "cloudy")}, b["gain"], False,
           "Центр сейчас = argmax гауссиана σ=40 от ожидаемой степени; он вне рамки ±170 px у "
           f"{100 * d['est_outside_box170_frac']:.0f} % чипов (у эталона — {100 * d['gt_outside_box170_frac']:.0f} %), медианная ошибка 73 px, "
           "но при zero_r=300 правило почти не зависит от центра: clip/boxmax дают Δ=0.0000, «центроид компоненты» −0.001…−0.005, "
           "zero_r=250 −0.004…−0.010.", {"diagnostics": d})

    # ---- H3 ----
    post_no_event = {k: v for k, v in S.post.items() if not k.startswith("event_")}
    results = {}
    for on_top, post in (("вместо event_zero", post_no_event), ("поверх event_zero", S.post)):
        for seed_mode in ("best", "top3near", "all"):
            for t_high in (0.6, 0.7, 0.8):
                for t_low in (0.2, 0.3, None):
                    def fn(s, i, p, dec, chip, t_high=t_high, t_low=t_low, seed_mode=seed_mode, post=post):
                        tl = dec["t_burn"] if t_low is None else t_low
                        m = grow(p, t_high, tl, dec["w"], seed_mode)
                        if m is None:  # нет уверенного seed — обычное декодирование
                            from firemon.models.bs.common import decode_probs
                            m = decode_probs(p, dec["t_burn"], dec["w"])
                        return postprocess_bs(m, chip, post, probs=p, decode=dec)
                    name = f"{on_top}, seed={seed_mode}, t_high={t_high}, t_low={'t_burn' if t_low is None else t_low}"
                    m = S.evaluate(fn)
                    g = gains(m, base)
                    results[name] = {**m, "gain": g}
                    print(line(name, m, g))
    best = max(results, key=lambda k: min(results[k]["gain"]["group"], results[k]["gain"]["temporal"]))
    r = results[best]
    single = max((k for k in results if "seed=all" not in k), key=lambda k: min(results[k]["gain"]["group"], results[k]["gain"]["temporal"]))
    rs = results[single]["gain"]
    record("h3_region_growing", f"H3: seed + region growing гистерезисом (Stroppiana et al. 2021, RS 13:2214): 54 варианта "
           f"(вместо/поверх event_zero × seed best/top3near/all × t_high 0.6–0.8 × t_low 0.2/0.3/t_burn), лучший «{best}»",
           "`scripts/bs_night_h3.py`", {k: r[k] for k in ("group", "temporal", "cloudy")}, r["gain"], accepted(r["gain"]),
           f"Один seed («{single}») даёт {rs['group']:+.4f} / {rs['temporal']:+.4f}: в эталоне медиана 18 несвязных компонент на чип, "
           "рост от одного seed их теряет; гистерезис от всех seed почти совпадает с текущим порогом, т.к. вероятности стека почти бинарны.",
           {"all_variants": {k: v["gain"] for k, v in results.items()}})
    print("лучший:", best, r["gain"], "принято" if accepted(r["gain"]) else "не принято")


if __name__ == "__main__":
    main()
