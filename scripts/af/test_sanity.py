"""Sanity-проверка AFPredictor на реальном тесте: только агрегированная статистика test vs train
(без разметки, без правки предсказаний, без геопривязки/дат). → reports/af_test_sanity.md
Запуск: python scripts/af/test_sanity.py [--test-dir data/test]"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from firemon.features.af_features import night_mask
from firemon.io import list_chips, read_af_chip
from firemon.models.af.predictor import AFPredictor
from firemon.utils import REPO_ROOT

Q = [0, 10, 25, 50, 75, 90, 99, 100]


def collect(data_dir, predictor: AFPredictor) -> dict:
    ids = list_chips(data_dir, "af")

    def work(cid):
        chip = read_af_chip(data_dir, cid, with_mask=False)
        cand, rows = predictor.candidates_and_rows(chip)
        v, valid = chip["viirs"], chip["valid"]
        night = bool(night_mask(v)[valid].mean() > 0.5) if valid.any() else False
        bg = float(np.median(v[3][valid])) if valid.any() else np.nan
        return cand, rows, night, bg, v[3][cand], float(valid.mean())

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(work, ids))
    masks = predictor.masks_from_parts([(r[0], r[1]) for r in res])
    night = np.array([r[2] for r in res])
    return {"n": len(ids), "n_cand": np.array([r[0].sum() for r in res]), "n_pred": np.array([m.sum() for m in masks]),
            "night": night, "bg_i4": np.array([r[3] for r in res]), "valid": np.array([r[5] for r in res]),
            "cand_i4_day": np.concatenate([r[4] for r, n in zip(res, night) if not n] or [np.zeros(0)]),
            "cand_i4_night": np.concatenate([r[4] for r, n in zip(res, night) if n] or [np.zeros(0)])}


def qrow(name: str, a: np.ndarray, fmt: str = "{:.0f}") -> str:
    if len(a) == 0:
        return f"| {name} | — |" + " — |" * (len(Q) - 1)
    return f"| {name} | " + " | ".join(fmt.format(x) for x in np.percentile(a, Q)) + " |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default=str(REPO_ROOT / "data" / "test"))
    ap.add_argument("--train-dir", default=str(REPO_ROOT / "data" / "train"))
    args = ap.parse_args()
    predictor = AFPredictor({"num_threads": 4}).load()
    tr, te = collect(args.train_dir, predictor), collect(args.test_dir, predictor)
    head = "| | " + " | ".join(f"p{q}" for q in Q) + " |\n|---|" + "---|" * len(Q)
    L = ["# AF: sanity-проверка на реальном тесте (`data/test`, только статистика)", "",
         "Сформировано `scripts/af/test_sanity.py`. Предсказания не правились, разметка теста не выполнялась, геопривязка и даты",
         "не восстанавливались — сравниваются только агрегаты по чипам. Train-предсказания здесь **in-sample** (финальные",
         "фолд-модели видели эти чипы), они служат лишь ориентиром масштаба.", "",
         f"Порог {predictor.threshold:.2f}; чипов: train {tr['n']}, test {te['n']}.", "",
         "## Сводка", "", "| Показатель | train | test |", "|---|---|---|"]

    def pct(x):
        return f"{100 * x:.1f} %"

    L += [f"| доля ночных чипов | {pct(tr['night'].mean())} | {pct(te['night'].mean())} |",
          f"| доля чипов с пустой маской | {pct((tr['n_pred'] == 0).mean())} (эталон: 124/420 = 29.5 %) | {pct((te['n_pred'] == 0).mean())} |",
          f"| — среди дневных | {pct((tr['n_pred'][~tr['night']] == 0).mean())} | {pct((te['n_pred'][~te['night']] == 0).mean())} |",
          f"| — среди ночных | {pct((tr['n_pred'][tr['night']] == 0).mean())} | {pct((te['n_pred'][te['night']] == 0).mean())} |",
          f"| доля чипов без кандидатов | {pct((tr['n_cand'] == 0).mean())} | {pct((te['n_cand'] == 0).mean())} |",
          f"| предсказано пикселей огня всего / на чип | {tr['n_pred'].sum()} / {tr['n_pred'].mean():.1f} | {te['n_pred'].sum()} / {te['n_pred'].mean():.1f} |",
          f"| доля кандидатов, ставших огнём | {pct(tr['n_pred'].sum() / max(tr['n_cand'].sum(), 1))} | {pct(te['n_pred'].sum() / max(te['n_cand'].sum(), 1))} |",
          f"| чипов с valid_frac < 1 | {(tr['valid'] < 0.999).sum()} | {(te['valid'] < 0.999).sum()} |", ""]
    for title, key, fmt in (("Кандидатов на чип", "n_cand", "{:.0f}"), ("Предсказанных пикселей огня на чип", "n_pred", "{:.0f}")):
        L += [f"## {title} (квантили)", "", head, qrow("train", tr[key], fmt), qrow("test", te[key], fmt),
              qrow("train, только чипы с огнём" if key == "n_pred" else "train, день", tr[key][tr[key] > 0] if key == "n_pred" else tr[key][~tr["night"]], fmt),
              qrow("test, только чипы с огнём" if key == "n_pred" else "test, день", te[key][te[key] > 0] if key == "n_pred" else te[key][~te["night"]], fmt), ""]
    L += ["## I4 кандидатов, K (квантили по пикселям)", "", head,
          qrow("train, день", tr["cand_i4_day"], "{:.1f}"), qrow("test, день", te["cand_i4_day"], "{:.1f}"),
          qrow("train, ночь", tr["cand_i4_night"], "{:.1f}"), qrow("test, ночь", te["cand_i4_night"], "{:.1f}"), "",
          "## Фон: медиана I4 по чипу, K (квантили по чипам)", "", head,
          qrow("train, день", tr["bg_i4"][~tr["night"]], "{:.1f}"), qrow("test, день", te["bg_i4"][~te["night"]], "{:.1f}"),
          qrow("train, ночь", tr["bg_i4"][tr["night"]], "{:.1f}"), qrow("test, ночь", te["bg_i4"][te["night"]], "{:.1f}"), ""]
    L += ["## Срез по прогретости дневной сцены (медиана I4 чипа)", "",
          "| Срез | выборка | чипов | пустых масок | предсказано px на чип (среднее / медиана) | чипов с 1–2 px |", "|---|---|---|---|---|---|"]
    for lo, hi in ((0, 305), (305, 315), (315, 1000)):
        for nm, d in (("train", tr), ("test", te)):
            sel = ~d["night"] & (d["bg_i4"] >= lo) & (d["bg_i4"] < hi)
            n = int(sel.sum())
            if n == 0:
                L.append(f"| {lo}–{hi} K | {nm} | 0 | — | — | — |")
                continue
            k = d["n_pred"][sel]
            L.append(f"| {lo}–{hi} K | {nm} | {n} | {pct((k == 0).mean())} | {k.mean():.1f} / {np.median(k):.0f} | {int(((k > 0) & (k <= 2)).sum())} |")
    L.append("")
    out = REPO_ROOT / "reports" / "af_test_sanity.md"
    # раздел «Выводы» пишется вручную и при перегенерации статистики сохраняется
    marker = "## Выводы"
    old = out.read_text(encoding="utf-8") if out.exists() else ""
    tail = old[old.index(marker):] if marker in old else marker + "\n\n(заполнить после просмотра статистики)\n"
    out.write_text("\n".join(L) + "\n" + tail, encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
