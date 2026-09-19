"""Сводный отчёт ночного режима: reports/bs_night.json → out/04_bs_night.md (таблица гипотез)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(extra_path: str | None = None) -> None:
    data = json.loads((ROOT / "reports" / "bs_night.json").read_text(encoding="utf-8"))
    extra = Path(extra_path).read_text(encoding="utf-8") if extra_path else ""
    rows = []
    for key, v in data.items():
        if v.get("module") == "AF":
            g = v["gain_f1"]
            rows.append(f"| `{key}` | AF | {v['title'][:160]} | F1 {v['f1_oof']:.4f} / {v['f1_temporal']:.4f} | ΔF1 {g['oof']:+.4f} | "
                        f"ΔF1 {g['temporal']:+.4f} | — | {'**ДА**' if v['accepted'] else 'нет'} | не измерялось | {v['why']} |")
        else:
            g, m = v["gain"], v["metrics"]
            rows.append(f"| `{key}` | BS | {v['title'][:160]} | {m['group']['bs_part']:.4f} / {m['temporal']['bs_part']:.4f} | "
                        f"{g['group']:+.4f} | {g['temporal']:+.4f} | {g['cloudy']:+.4f} | {'**ДА**' if v['accepted'] else 'нет'} | "
                        f"{v.get('inference_time', 'не измерялось (не принято)')} | {v['why']} |")
    n_acc = sum(1 for v in data.values() if v["accepted"])
    head = f"""# Ночной автономный режим BS/AF (ветка task/bs-lovasz) — сводный отчёт

Период: 2026-09-19, ≈ 02:00–09:00 МСК. База сравнения — финал task/bs: BS-часть 0.4912 (group) / 0.5008 (temporal) / 0.4401
(облачные чипы, cloud_frac > 0.25); AF-17: F1 0.9682 / 0.9597. Правило принятия: ≥ +0.005 BS-части (или ≥ +0.003 F1_af) на ОБОИХ
сплитах. `weights/`, конфиги по умолчанию, main и другие worktree не менялись; веса экспериментов — в `weights_lovasz*/`,
`weights_night_*/` (в git не входят). Все строки — также в `reports/experiments.md` (BS-109… и N-AF…), сырые числа —
`reports/bs_night.json`.

**Проверено гипотез: {len(data)}, принято: {n_acc}.**

| Ключ | Модуль | Гипотеза | Метрика (group / temporal) | Δ group | Δ temporal | Δ облачные | Принято | Время инференса | Почему |
|---|---|---|---|---|---|---|---|---|---|
"""
    out = head + "\n".join(rows) + "\n\n" + extra
    (ROOT / "out" / "04_bs_night.md").write_text(out, encoding="utf-8", newline="\n")
    print(f"гипотез {len(data)}, принято {n_acc}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
