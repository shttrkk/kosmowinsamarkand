"""Оценка OOF-вероятностей стека (любой тег/каталог) в финальном режиме: декодирование подбирается заново
(temporal — на групповом OOF чипов не из 2024), постобработка — из weights/bs_decode.json; запись в журнал.

    python scripts/bs_night_eval_stack.py --tag stack_night_h4_med2 --key h4_med2 --title "…" --why "…" [--oof-dir …] [--cloudy-min 0.003]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bs_night_lib import OOF, Setup, accepted, gains, line, record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--oof-dir", default=str(OOF))
    ap.add_argument("--key", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--config", default="—")
    ap.add_argument("--why", default="")
    ap.add_argument("--cloudy-min", type=float, default=None)
    ap.add_argument("--time", default="—")
    args = ap.parse_args()
    S = Setup()
    base = S.evaluate(S.baseline_fn())
    S.load_probs(args.tag, Path(args.oof_dir), fit=True)
    m = S.evaluate(S.baseline_fn())
    g = gains(m, base)
    print(line("текущий финал", base))
    print(line(args.tag, m, g))
    print("decode:", S.decs)
    ok = accepted(g, args.cloudy_min)
    record(args.key, args.title, args.config, m, g, ok, args.why, {"decode": S.decs}, args.time)
    print("→", "ПРИНЯТО" if ok else "не принято")


if __name__ == "__main__":
    main()
