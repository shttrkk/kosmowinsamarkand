"""Сдвиг распределений test vs train → reports/test_vs_train.md.

    python eda/test_vs_train.py

Используется только содержимое чипов и поля тестового meta.csv (правила §7: никакой
геопривязки/дат теста). AF — все пиксели; BS — каждый 4-й пиксель по обеим осям (для квантилей
этого достаточно, 16 тыс. пикселей на чип).
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.io import list_chips, read_af_chip, read_bs_chip, read_meta  # noqa: E402

OUT = ROOT / "reports" / "test_vs_train.md"
QS = [1, 5, 25, 50, 75, 95, 99, 99.9]
LC_NAMES = {10: "деревья", 20: "кустарник", 30: "трава", 40: "пашня", 50: "застройка", 60: "голая почва",
            70: "снег", 80: "вода", 90: "болото", 95: "мангры", 100: "мох"}
BS_STRIDE = 4

# Выводы по состоянию на 2026-09-18 (тест организаторов fire-test-renamed.tar); при смене данных — перепроверить.
CONCLUSIONS = """## Выводы (наблюдение → решение)

1. **AF: дневной фон в тесте теплее.** I4 днём: p99 +6.8 K, p99.9 +6.4 K (334 K против 328 K), I5 p99 +5 K, при этом
   I4−I5 в хвосте даже ниже (p99 −3.9 K). Абсолютный порог `I4 > 320 K` на тесте срабатывает в среднем на 1141 пикселе
   дневного чипа против 214 в train (×5.3; медиана 67 против 45 — сдвиг сидит в хвосте из нескольких очень жарких сцен).
   → Абсолютные пороги по I4 непереносимы; признаки — только относительно фона чипа/окна (z-score, разность с медианой
   окна), пороги подбирать на temporal-сплите, а не на всём train.
2. **AF: ночных чипов больше** (38.3 % против 30.2 %). Ночью заглушка почти не ошибается (3–4 пикселя на чип), хвост
   I4−I5 ночью в тесте чуть тяжелее (p99.9 21 K против 17 K). → Раздельная логика/калибровка день и ночь обязательна;
   вклад ночных чипов в F1 на тесте выше, чем в train.
3. **Покров смещён от пашни к траве и болоту** (AF: пашня −7.1 п.п., трава +7.4, болото +4.1, лес −5.7; BS: пашня
   −12.1 п.п., трава +9.4, болото +3.8). → Landcover — признак, а не жёсткий фильтр; в валидации смотреть метрики по
   классам покрова, особенно трава/болото (в train их меньше); пороги тяжести по типу покрова — ключевая гипотеза BS.
4. **BS: тест облачнее.** Чипов с cloud_frac > 0.25 — 25.8 % против 10.7 %; SCL-облака pre 14.8 % против 10.9 %, тени
   +0.6–0.7 п.п., valid_frac в среднем 0.879 против 0.919. → Политика по SCL (маскирование/понижение уверенности на
   облаках и тенях post-снимка, опора на S1 там, где S2 закрыт) даст на тесте больше, чем показывает train-валидация;
   в валидации отдельно отчитывать метрики на облачных чипах.
5. **BS: хвост dNBR тяжелее** (p95 0.42 против 0.35, p99 0.66 против 0.56; доля dNBR > 0.10 — 27.0 % против 22.2 % при
   10.1 % гари в эталоне train). Часть сдвига — облака/тени (dNBR p1 тоже ушёл вниз: −0.42 против −0.35), часть — иной
   покров. → Глобальные пороги dNBR на тесте дадут ещё больше ложной «гари», чем на train; нужны RdNBR/нормировка на
   фон чипа и обученная модель с landcover и SCL.

Структура файлов (формы, dtype, имена каналов) в тесте идентична train; NaN/невалидных пикселей AF чуть больше
(9.4 % против 8.3 %), насыщения I4 нет ни там, ни там.
""".splitlines()


def af_stats(data_dir: Path, cid: str) -> dict:
    chip = read_af_chip(data_dir, cid, with_mask=False)
    v, valid = chip["viirs"], chip["valid"]
    i4, i5 = v[3][valid], v[4][valid]
    night = bool(np.nanmedian(v[5]) > 90)
    lc = chip["aux"][0]
    stub = valid & (v[3] > 320.0) & (v[3] - v[4] > 15.0)  # правило-заглушка baseline
    return {"i4": i4, "i5": i5, "d45": i4 - i5, "night": night, "nan_frac": float(np.isnan(v[3]).mean()),
            "invalid_frac": float(1 - valid.mean()), "sat": int((i4 >= 367).sum()), "n": int(valid.sum()),
            "lc": np.bincount(np.nan_to_num(lc, nan=0).astype(np.int64).ravel(), minlength=256)[:256],
            "stub_px": int(stub.sum())}


def bs_stats(data_dir: Path, cid: str) -> dict:
    chip = read_bs_chip(data_dir, cid, with_mask=False, layers=("s2_pre", "s2_post", "aux"))
    s = np.s_[::BS_STRIDE, ::BS_STRIDE]
    pre, post = chip["s2_pre"][:, s[0], s[1]].astype(np.float32), chip["s2_post"][:, s[0], s[1]].astype(np.float32)
    valid = chip["valid"][s]

    def nbr(x):
        return (x[6] - x[8]) / np.maximum(x[6] + x[8], 1.0)

    nbr_pre, nbr_post = nbr(pre)[valid], nbr(post)[valid]
    scl_pre, scl_post = chip["s2_pre"][9], chip["s2_post"][9]
    return {"dnbr": nbr_pre - nbr_post, "nbr_pre": nbr_pre,
            "cloud_pre": float(np.isin(scl_pre, (8, 9, 10)).mean()), "cloud_post": float(np.isin(scl_post, (8, 9, 10)).mean()),
            "shadow_pre": float((scl_pre == 3).mean()), "shadow_post": float((scl_post == 3).mean()),
            "water_post": float((scl_post == 6).mean()), "nodata": float(1 - chip["valid"].mean()),
            "lc": np.bincount(chip["aux"][2].clip(0, 255).astype(np.int64).ravel(), minlength=256)[:256]}


def collect(fn, data_dir: Path, kind: str) -> list[dict]:
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda c: fn(data_dir, c), list_chips(data_dir, kind)))


def q_row(name: str, tr: np.ndarray, te: np.ndarray, fmt: str = "{:.1f}") -> list[str]:
    a, b = np.percentile(tr, QS), np.percentile(te, QS)
    return [f"| {name} p{q:g} | {fmt.format(x)} | {fmt.format(y)} | {fmt.format(y - x)} |" for q, x, y in zip(QS, a, b)]


def v_row(name: str, x: float, y: float, fmt: str = "{:.3f}") -> str:
    return f"| {name} | {fmt.format(x)} | {fmt.format(y)} | {fmt.format(y - x)} |"


def lc_rows(tr: list[dict], te: list[dict]) -> list[str]:
    a, b = sum(r["lc"] for r in tr), sum(r["lc"] for r in te)
    a, b = a / a.sum() * 100, b / b.sum() * 100
    codes = [c for c in range(256) if a[c] > 0.05 or b[c] > 0.05]
    return [v_row(f"landcover {c} ({LC_NAMES.get(c, 'нет данных' if c == 0 else '?')}), % пикселей", a[c], b[c], "{:.1f}")
            for c in codes]


def cat(rows: list[dict], key: str, night: bool | None = None) -> np.ndarray:
    return np.concatenate([r[key] for r in rows if night is None or r["night"] == night])


def main() -> int:
    train, test = ROOT / "data" / "train", ROOT / "data" / "test"
    head = ["| Показатель | train | test | Δ (test − train) |", "|---|---|---|---|"]
    lines = ["# Сдвиг распределений: test vs train", "",
             "Сгенерировано `python eda/test_vs_train.py`. Только содержимое чипов и поля тестового `meta.csv` "
             "(геопривязка и даты теста не используются и не восстанавливаются).", ""]

    tr, te = collect(af_stats, train, "af"), collect(af_stats, test, "af")
    lines += [f"## AF (train {len(tr)} чипов, test {len(te)} чипов; квантили по валидным пикселям)", ""] + head
    lines.append(v_row("доля ночных чипов (медиана solar_zenith > 90)", np.mean([r["night"] for r in tr]),
                       np.mean([r["night"] for r in te])))
    for key, name in (("i4", "I4, K"), ("i5", "I5, K"), ("d45", "I4−I5, K")):
        lines += q_row(name, cat(tr, key), cat(te, key))
    for night, tag in ((False, "день"), (True, "ночь")):
        for key, name in (("i4", "I4"), ("d45", "I4−I5")):
            a, b = cat(tr, key, night), cat(te, key, night)
            for q in (50, 99, 99.9):
                lines.append(v_row(f"{name} {tag} p{q:g}, K", np.percentile(a, q), np.percentile(b, q), "{:.1f}"))
    lines.append(v_row("доля NaN в I4, %", 100 * np.mean([r["nan_frac"] for r in tr]),
                       100 * np.mean([r["nan_frac"] for r in te])))
    lines.append(v_row("доля невалидных пикселей (valid=0 или NaN), %", 100 * np.mean([r["invalid_frac"] for r in tr]),
                       100 * np.mean([r["invalid_frac"] for r in te])))
    lines.append(v_row("чипов с невалидными пикселями, %", 100 * np.mean([r["invalid_frac"] > 0 for r in tr]),
                       100 * np.mean([r["invalid_frac"] > 0 for r in te]), "{:.1f}"))
    lines.append(v_row("пикселей с насыщением I4 ≥ 367 K на чип", np.mean([r["sat"] for r in tr]),
                       np.mean([r["sat"] for r in te]), "{:.2f}"))
    for night, tag in ((False, "дневной"), (True, "ночной")):
        a = [r["stub_px"] for r in tr if r["night"] == night]
        b = [r["stub_px"] for r in te if r["night"] == night]
        lines.append(v_row(f"срабатываний правила-заглушки на {tag} чип (среднее)", np.mean(a), np.mean(b), "{:.1f}"))
        lines.append(v_row(f"срабатываний правила-заглушки на {tag} чип (медиана)", np.median(a), np.median(b), "{:.1f}"))
    lines.append(v_row("доля чипов без срабатываний заглушки, %", 100 * np.mean([r["stub_px"] == 0 for r in tr]),
                       100 * np.mean([r["stub_px"] == 0 for r in te]), "{:.1f}"))
    lines += lc_rows(tr, te)
    af_gt = read_meta(train, "af")
    lines += ["", f"Справочно, эталон train: {af_gt['n_fire_px'].sum()} пикселей огня = "
              f"{af_gt['n_fire_px'].mean():.1f} на чип, {100 * (af_gt['n_fire_px'] == 0).mean():.1f} % чипов без огня.", ""]

    tr, te = collect(bs_stats, train, "bs"), collect(bs_stats, test, "bs")
    m_tr, m_te = read_meta(train, "bs"), read_meta(test, "bs")
    lines += [f"## BS (train {len(tr)} чипов, test {len(te)} чипов; квантили по валидным пикселям, шаг {BS_STRIDE})", ""] + head
    lines += q_row("dNBR", cat(tr, "dnbr"), cat(te, "dnbr"), "{:.3f}")
    for q in (5, 50, 95):
        lines.append(v_row(f"NBR_pre p{q}", np.percentile(cat(tr, "nbr_pre"), q), np.percentile(cat(te, "nbr_pre"), q)))
    for thr in (0.10, 0.27, 0.44):
        lines.append(v_row(f"доля пикселей dNBR > {thr:.2f}, %", 100 * np.mean(cat(tr, "dnbr") > thr),
                           100 * np.mean(cat(te, "dnbr") > thr), "{:.1f}"))
    for key, name in (("cloud_pre", "SCL-облака (8/9/10) pre"), ("cloud_post", "SCL-облака (8/9/10) post"),
                      ("shadow_pre", "SCL-тени (3) pre"), ("shadow_post", "SCL-тени (3) post"),
                      ("water_post", "SCL-вода (6) post"), ("nodata", "нет данных S2 (SCL=0 pre или post)")):
        lines.append(v_row(f"{name}, % пикселей", 100 * np.mean([r[key] for r in tr]), 100 * np.mean([r[key] for r in te]), "{:.2f}"))
    for col in ("cloud_frac", "valid_frac"):
        for stat, fn in (("среднее", np.mean), ("медиана", np.median), ("min", np.min), ("max", np.max)):
            lines.append(v_row(f"meta {col}, {stat}", fn(m_tr[col]), fn(m_te[col])))
    lines.append(v_row("чипов с cloud_frac > 0.25, %", 100 * np.mean(m_tr["cloud_frac"] > 0.25),
                       100 * np.mean(m_te["cloud_frac"] > 0.25), "{:.1f}"))
    lines += lc_rows(tr, te)
    burn = (m_tr["sev1_px"] + m_tr["sev2_px"] + m_tr["sev3_px"]).sum() / (len(m_tr) * 512 * 512)
    lines += ["", f"Справочно, эталон train: гарь занимает {100 * burn:.1f} % пикселей.", ""]

    lines += CONCLUSIONS
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
