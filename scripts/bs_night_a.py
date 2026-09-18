"""Идея (a): ординальное декодирование — три бинарных стекера P(sev ≥ k) (stack.py --ordinal), класс = число «да»
при РАЗДЕЛЬНЫХ порогах t1, t2, t3 (координатный спуск по BS-части на групповом OOF; temporal — на чипах не из 2024)."""
from __future__ import annotations

import numpy as np

from bs_night_lib import Setup, accepted, gains, line, record
from firemon.models.bs.postprocess import postprocess_bs

GRID = np.round(np.arange(0.2, 0.81, 0.04), 2)


def cum(p):
    return 1.0 - p[0], p[2] + p[3], p[3]


def decode(p, t):
    c1, c2, c3 = cum(p)
    return ((c1 > t[0]).astype(np.uint8) + ((c1 > t[0]) & (c2 > t[1])) + ((c1 > t[0]) & (c2 > t[1]) & (c3 > t[2]))).astype(np.uint8)


def main() -> None:
    S = Setup()
    base = S.evaluate(S.baseline_fn())
    S.load_probs("stack_night_ordinal", fit=True)

    def objective(idx, t):
        inter, union = np.zeros(4), np.zeros(4)
        for i in idx:
            m, gt = decode(S.p("group", i), t), S.gts[i]
            for k in (1, 2, 3):
                inter[k] += np.count_nonzero((m == k) & (gt == k))
                union[k] += np.count_nonzero((m == k) | (gt == k))
            inter[0] += np.count_nonzero((m > 0) & (gt > 0))
            union[0] += np.count_nonzero((m > 0) | (gt > 0))
        iou = inter / np.maximum(union, 1)
        return 0.35 * iou[0] + 0.30 * iou[1:].mean()

    def fit(idx):
        t = [0.46, 0.5, 0.5]
        best = objective(idx, t)
        for _ in range(2):
            for j in range(3):
                for v in GRID:
                    trial = list(t)
                    trial[j] = float(v)
                    s = objective(idx, trial)
                    if s > best + 1e-6:
                        best, t = s, trial
        return t

    sub = S.idx["group"][::2]  # подбор на половине чипов — быстрее, порогов всего три
    t_group = fit(sub)
    t_temp = fit(np.array([i for i in S.t_tr if i % 2 == 0]))
    print("пороги group:", t_group, "| temporal:", t_temp)

    def fn(s, i, p, dec, chip):
        t = t_group if s == "group" else t_temp
        return postprocess_bs(decode(p, t), chip, S.post, probs=p, decode=dec)

    m_sep = S.evaluate(fn)
    m_std = S.evaluate(S.baseline_fn())
    print(line("текущий финал", base))
    print(line("ординальный стек, обычное декодирование", m_std, gains(m_std, base)))
    print(line("ординальный стек, раздельные пороги", m_sep, gains(m_sep, base)))
    best, name = (m_sep, "раздельные пороги") if min(gains(m_sep, base)["group"], gains(m_sep, base)["temporal"]) >= \
        min(gains(m_std, base)["group"], gains(m_std, base)["temporal"]) else (m_std, "обычное декодирование")
    g = gains(best, base)
    g2 = gains(m_std if best is m_sep else m_sep, base)
    record("a_ordinal", f"Идея (a): ординальный стек — три бинарных LGBM P(sev≥k) вместо multiclass; лучший вариант декодирования — «{name}» "
           f"(пороги group {t_group}, temporal {t_temp})", "`stack.py --ordinal`, `scripts/bs_night_a.py`", best, g, accepted(g),
           f"Другой вариант декодирования: {g2['group']:+.4f} / {g2['temporal']:+.4f}. Соседние степени стек уже почти не путает (1→2 1.4 %, 2→3 0.9 %): "
           "ошибка — в контуре, порядок классов её не лечит.")


if __name__ == "__main__":
    main()
