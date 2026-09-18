"""Идея (c): степень = смесь вероятностей стека и порогового правила по покрову, вес α подбирается ПО КАЖДОМУ покрову.

q_k = (1−α_lc)·p_k + α_lc·p_burn·[rule_sev == k] для пикселей, где правило даёт гарь; контур не меняется.
α подбирается координатным спуском по mIoU_sev на групповом OOF (для temporal — на чипах не из 2024)."""
from __future__ import annotations

import numpy as np

from bs_night_lib import Setup, accepted, gains, line, record
from firemon.features.bs_features import compute_bs_features
from firemon.models.bs.common import decode_probs
from firemon.models.bs.postprocess import postprocess_bs

LCS = (10, 30, 40, 90, -1)  # -1 — прочие
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def main() -> None:
    S = Setup()
    base = S.evaluate(S.baseline_fn())
    print(line("текущий финал", base))
    rule = [compute_bs_features(c, groups=["rule"])[0].astype(np.uint8) for c in S.chips]
    lcg = []
    for c in S.chips:
        lc = c["aux"][2]
        g = np.full(lc.shape, -1, np.int16)
        for code in (10, 30, 40, 90):
            g[lc == code] = code
        lcg.append(g)

    def mixed(p, i, alpha: dict):
        q = p.copy()
        pb = p[1:].sum(0)
        a = np.zeros(p.shape[1:], np.float32)
        for code, v in alpha.items():
            a[lcg[i] == code] = v
        a = np.where(rule[i] > 0, a, 0.0)
        for k in (1, 2, 3):
            q[k] = (1 - a) * p[k] + a * pb * (rule[i] == k)
        return q

    def fit_alpha(idx) -> dict:
        """Координатный спуск по mIoU_sev (без постобработки — она классы почти не меняет)."""
        alpha = {c: 0.0 for c in LCS}
        dec = S.decs["group"]

        def miou(al):
            inter, union = np.zeros(3), np.zeros(3)
            for i in idx:
                m = decode_probs(mixed(S.p("group", i), i, al), dec["t_burn"], dec["w"])
                gt = S.gts[i]
                for k in (1, 2, 3):
                    inter[k - 1] += np.count_nonzero((m == k) & (gt == k))
                    union[k - 1] += np.count_nonzero((m == k) | (gt == k))
            return float((inter / np.maximum(union, 1)).mean())

        best = miou(alpha)
        for code in LCS:
            for v in ALPHAS[1:]:
                trial = {**alpha, code: v}
                s = miou(trial)
                if s > best + 1e-5:
                    best, alpha = s, trial
        return alpha

    a_group = fit_alpha(S.idx["group"])
    a_temp = fit_alpha(S.t_tr)
    print("α (group):", a_group, "| α (для temporal, подбор на ≤2023):", a_temp)

    def fn(s, i, p, dec, chip):
        al = a_group if s == "group" else a_temp
        q = mixed(p, i, al)
        return postprocess_bs(decode_probs(q, dec["t_burn"], dec["w"]), chip, S.post, probs=q, decode=dec)

    m = S.evaluate(fn)
    g = gains(m, base)
    print(line("смесь стек + правило, α по покрову", m, g))
    record("c_lc_mix", "Идея (c): степень = смесь вероятностей стека и порогового правила по покрову, вес α подбирается отдельно "
           f"для каждого landcover (α group {a_group}, α temporal {a_temp})", "`scripts/bs_night_c.py`",
           m, g, accepted(g), "Правило по медианному dNBR уже является признаком стека (rule_sev, dNBR_med3): подобранные α малы/нулевые, "
           "смесь не добавляет информации.", {"alpha_group": {str(k): v for k, v in a_group.items()},
                                              "alpha_temporal": {str(k): v for k, v in a_temp.items()}})


if __name__ == "__main__":
    main()
