"""Идея (b), шаг 0 — потолок: сколько даст ИДЕАЛЬНАЯ степень внутри предсказанного контура (оракул: там, где и модель,
и эталон говорят «гарь», берём класс эталона). Если потолок < +0.005 — регрессия сглаженного dNBR не имеет смысла."""
from __future__ import annotations

import numpy as np

from bs_night_lib import Setup, accepted, gains, line, record


def main() -> None:
    S = Setup()
    base_fn = S.baseline_fn()
    base = S.evaluate(base_fn)

    def oracle(s, i, p, dec, chip):
        m = base_fn(s, i, p, dec, chip)
        gt = S.gts[i]
        both = (m > 0) & (gt > 0) & (gt <= 3)
        return np.where(both, gt, m).astype(np.uint8)

    m = S.evaluate(oracle)
    g = gains(m, base)
    print(line("текущий финал", base))
    print(line("оракул степени внутри контура", m, g))
    record("b_severity_ceiling", "Идея (b), потолок: ИДЕАЛЬНАЯ степень внутри предсказанного контура (оракул по эталону на пересечении "
           "pred∩GT) — верхняя граница для регрессии сглаженного dNBR и любых других улучшений степени", "`scripts/bs_night_b_ceiling.py`",
           m, g, False, f"Это оракул, не модель: потолок всего {g['group']:+.4f} / {g['temporal']:+.4f}. Регрессия dNBR с порогами по покрову "
           "может забрать лишь часть (правило по dNBR_med3 уже в признаках стека, §4.4 п.4: правило хуже модели на 0.001) → идея (b) не "
           "реализовывалась, ресурс направлен на контур.")


if __name__ == "__main__":
    main()
