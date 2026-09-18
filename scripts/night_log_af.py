"""Строка AF-эксперимента (ночной режим) в reports/experiments.md и reports/bs_night.json."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(key: str, json_name: str, title: str, config: str, why: str) -> None:
    r = json.loads((ROOT / "reports" / json_name).read_text(encoding="utf-8"))
    path = ROOT / "reports" / "experiments.md"
    text = path.read_text(encoding="utf-8")
    mark = f"[af:{key}]"
    d = r["delta_f1"]
    if mark not in text:
        lines = text.rstrip("\n").split("\n")
        last = max(i for i, l in enumerate(lines) if l.startswith("|"))
        n = sum(1 for l in lines if l.startswith("| N-AF")) + 1
        row = (f"| N-AF{n} | {dt.date.today().isoformat()} | AF | {title} {mark} | {config} | GroupKFold-5 по тайлу (OOF) + temporal "
               f"(val=2024) | 42 | {r['new']['oof']['f1']:.4f} (P {r['new']['oof']['precision']:.3f} / R {r['new']['oof']['recall']:.3f}); "
               f"temporal {r['new']['temporal']['f1']:.4f} | — | — | — | — | Δ F1 к AF-17 (пересчёт {r['base']['oof']['f1']:.4f} / "
               f"{r['base']['temporal']['f1']:.4f}): {d['oof']:+.4f} / {d['temporal']:+.4f} → "
               f"{'ПРИНЯТО' if r['accepted'] else 'не принято'} (правило: ≥ +0.003 на обоих). {why} |")
        lines.insert(last + 1, row)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    night = ROOT / "reports" / "bs_night.json"
    data = json.loads(night.read_text(encoding="utf-8")) if night.exists() else {}
    data[key] = {"title": title, "module": "AF", "f1_oof": r["new"]["oof"]["f1"], "f1_temporal": r["new"]["temporal"]["f1"],
                 "gain_f1": d, "accepted": r["accepted"], "why": why}
    night.write_text(json.dumps(data, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main(*sys.argv[1:6])
