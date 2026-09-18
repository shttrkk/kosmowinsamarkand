"""Переносит результаты BS-экспериментов из data/cache/bs_oof/*.json в reports/experiments.md.

    python scripts/bs_log_experiments.py            # добавляет только отсутствующие строки (по метке [bs:<ключ>])

Каждая строка: обе цифры «групповой OOF / временной сплит (val=2024)».
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.models.bs.common import log_experiment  # noqa: E402

OOF = ROOT / "data" / "cache" / "bs_oof"
VAL = "GroupKFold-5 (`folds_bs.csv`) / temporal val=2024"


def load(name: str) -> dict | None:
    p = OOF / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    path = ROOT / "reports" / "experiments.md"
    rows: list[tuple] = []  # (key, description, config, m_group, m_temporal, time, conclusion)

    r = load("rules.json")
    if r:
        rows.append(("rules_usgs", "Шаг 0: dNBR, глобальные пороги USGS 0.10/0.27/0.44", "`models/bs/rules.py`",
                     r["group"]["usgs"], r["temporal"]["usgs"], "<1 с",
                     "Точка отсчёта: один порог заведомо плох (IoU₁ 0.17), 60 % срабатываний — не «свой» пожар."))
        rows.append(("rules_lc", "Шаг 0: пороги dNBR по landcover (координатный спуск по гистограммам, подбор на train-фолдах)",
                     "`rules.fit_lc_thresholds`", r["group"]["by_landcover"], r["temporal"]["by_landcover"], "<1 с",
                     "mIoU_sev +0.09…+0.10 к USGS на обоих сплитах — тезис ТЗ подтверждён; контур почти не улучшается (нужна пространственная модель)."))

    r = load("lgbm_main.json")
    if r:
        rows.append(("lgbm_argmax", f"LightGBM multiclass по пикселям, {r['features']} признаков, веса 1/доля отбора; argmax",
                     "`models/bs/lgbm.py` PARAMS (leaves 127, lr 0.05, early stop)", r["group"]["argmax"], r["temporal"]["argmax"],
                     "—", f"best_iter {r['best_iters']}: переобучение на чипы наступает рано (70–100 деревьев)."))
        rows.append(("lgbm_tuned", "То же + декодирование: t_burn по IoU_burn, множители классов по mIoU_sev",
                     f"decode {r['group']['decode']['t_burn']:.2f} / w {r['group']['decode']['w']}", r["group"]["tuned"],
                     r["temporal"]["tuned"], "—", "Подбор порогов даёт +0.03…+0.04 BS-части к argmax на обоих сплитах — используем везде."))
    names = {"abl_full": "все признаки", "abl_no_sar": "без SAR", "abl_no_scl": "без SCL", "abl_no_lc": "без landcover",
             "abl_no_ctx": "без контекстных (окна, медианы)", "abl_minimal": "только dNBR+RdNBR+landcover",
             "abl_regression": "регрессия порядковой степени + 3 порога"}
    for tag, title in names.items():
        r = load(f"lgbm_{tag}.json")
        if r:
            rows.append((tag, f"Абляция LGBM ({title}); облегчённый режим: 1 млн строк, lr 0.1, leaves 63, ≤150 деревьев, оценка на сетке пикселей с шагом 2",
                         "`lgbm.py --stage ablate`", r["group"]["tuned"], r["temporal"]["tuned"], "—", ""))

    unets = {"r34_cedice": "UNet resnet34 (imagenet), 47 каналов v1, CE(веса 0.5/1/1/1.2)+Dice, 60 эпох, AdamW 3e-4 OneCycle, AMP, флипы/rot90",
             "r34_focal": "UNet resnet34 v1, Focal+Dice", "effb3": "UNet efficientnet-b3 v1, CE+Dice",
             "r34_v2": "UNet resnet34, каналы v2 (+dNBR_med3)", "r34_v3": "UNet resnet34, каналы v3 (+dNBR_med3, rule_sev — пиксельное правило эталона)",
             "r34_v4": "UNet resnet34, каналы v4 (v3 + dist_center)", "r34_v5": "UNet resnet34, каналы v5 (v4 + coord_x/y, dist_strong_cc, cc_area_rel)"}
    for tag, title in unets.items():
        g, t = load(f"{tag}_group.json"), load(f"{tag}_temporal.json")
        if not (g or t):
            continue
        for key, suffix in (("argmax", ""), ("argmax+TTA", " + TTA (4 отражения)")):
            rows.append((f"unet_{tag}_{key}", title + suffix + "; argmax, последняя эпоха (без выбора эпохи по валидации)",
                         f"`unet.py --tag {tag}`", g and g[key], t and t[key], "—", ""))

    for tag, title in {"r34_cedice": "UNet v1 → LGBM 2-го уровня (пиксельные признаки + вероятности UNet, ворота p_burn>0.03)",
                       "v1_geo": "Стек на UNet v1 + rule_sev, dist_center, dist_event (расстояние до оценённого центра события), p_burn на масштабах σ=20/40",
                       "v1_event": "Стек на UNet v1 + признаки локализации события (крупнейшая компонента сильного dNBR, dNBR-центроид, ранг/площадь компоненты)",
                       "v4_event": "Стек на UNet v4 + все признаки локализации"}.items():
        r = load(f"stack_{tag}.json")
        if r:
            rows.append((f"stack_{tag}", title + "; декодирование порогами", f"`stack.py --tag {tag}`", r["group"]["tuned"],
                         r["temporal"]["tuned"], "—", f"best_iter {r['best_iters']}"))

    for f in sorted(OOF.glob("ensemble_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        tag = f.stem.replace("ensemble_", "")
        for w, m in r.get("weights", {}).items():
            if w != "1.0":
                rows.append((f"ens_{tag}_{w}", f"Ансамбль: {w}·UNet(TTA) + {1 - float(w):.1f}·LGBM, декодирование порогами",
                             f"`ensemble.py` ({tag})", m["group"], m["temporal"], "—", ""))
        for step, m in r.get("post", {}).items():
            rows.append((f"post_{tag}_{step}", f"Постобработка финальной модели ({r.get('stack')}): шаг `{step}` поверх уже принятых",
                         f"`ensemble.py` ({tag})", m["group"], m["temporal"], "—",
                         f"Δ group {m['d_group']:+.4f} / temporal {m['d_temporal']:+.4f} / облачные>0.25 {m.get('d_cloudy', 0):+.4f} → "
                         + ("ПРИНЯТ" if m["accepted"] else "отклонён")))
        if "final" in r:
            rows.append((f"final_{tag}", f"ИТОГ: стек {r.get('stack')} + постобработка {r['postprocess']}", f"decode {r['final_decode']['t_burn']:.2f} / w {r['final_decode']['w']}",
                         r["final"]["group"], r["final"]["temporal"], "см. bs_module.md", ""))

    text = path.read_text(encoding="utf-8")
    added = 0
    for key, desc, cfg, mg, mt, tm, concl in rows:
        mark = f"[bs:{key}]"
        if mark in text:
            continue
        log_experiment(f"{desc} {mark}", cfg, VAL, mg, mt, tm, concl, path=path)
        text = path.read_text(encoding="utf-8")
        added += 1
    print(f"добавлено строк: {added}")


if __name__ == "__main__":
    main()
