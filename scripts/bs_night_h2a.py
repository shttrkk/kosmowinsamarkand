"""H2(а), чистый Lovász: обучение остановлено после UNet v1 (group + temporal) — сравнение на уровне UNet v1
с CE+Dice при одинаковой оценке (декодирование порогами + финальная постобработка, без TTA)."""
from __future__ import annotations

from pathlib import Path

from bs_night_lib import OOF, ROOT, Setup, gains, line, record


def main() -> None:
    S = Setup()
    S.load_probs("r34_cedice", OOF, fit=True)
    base = S.evaluate(S.baseline_fn())
    S.load_probs("r34_cedice", ROOT / "data" / "cache" / "bs_weights_lovasz_pure" / "bs_oof", fit=True)
    m = S.evaluate(S.baseline_fn())
    g = gains(m, base)
    print(line("UNet v1, CE+Dice (основная модель)", base))
    print(line("UNet v1, чистый Lovász", m, g))
    record("h2a_lovasz_pure", "H2(а): чистый Lovász-Softmax. Обучен только UNet v1 (group + temporal), прогон ОСТАНОВЛЕН досрочно; "
           "сравнение на уровне UNet v1 с CE+Dice (декодирование порогами + постобработка, без TTA; цифры — не стек!)",
           "`train.py --module bs --weights-dir weights_lovasz_pure/ --bs-unet-loss lovasz` (прерван после v1)", m, g, False,
           f"Δ здесь — к UNet v1 с CE+Dice при той же оценке ({base['group']['bs_part']:.4f} / {base['temporal']['bs_part']:.4f} / обл. "
           f"{base['cloudy']['bs_part']:.4f}), а не к финалу: {g['group']:+.4f} / {g['temporal']:+.4f} / {g['cloudy']:+.4f}. Чистый Lovász заметно "
           "хуже уже на уровне сети (без CE сеть хуже калибрована, а стек питается вероятностями), поэтому GPU отдана идее (d).")


if __name__ == "__main__":
    main()
