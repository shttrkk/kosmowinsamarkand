"""Собирает «фейковый тест» из train-чипов для сквозной проверки инференса.

    python scripts/make_fake_test.py            # 20 AF + 10 BS → data/test_fake/

Структура как у настоящего теста: без masks/, meta.csv только с тестовыми колонками,
плюс sample_submission.csv. Маски для оценки берутся из data/train (chip_id те же).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from firemon.io import LAYERS, chip_path, read_meta, win_path  # noqa: E402
from firemon.submission import write_submission  # noqa: E402

TEST_META_COLUMNS = ["chip_id", "kind", "width", "height", "gsd", "valid_frac", "cloud_frac"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "test_fake"))
    ap.add_argument("--n-af", type=int, default=20)
    ap.add_argument("--n-bs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_dir, out_dir = Path(args.train_dir), Path(args.out_dir)
    rng = np.random.RandomState(args.seed)

    rows = []
    for kind, n in (("af", args.n_af), ("bs", args.n_bs)):
        meta = read_meta(train_dir, kind)
        # худшие по valid_frac чипы — обязательно (проверка устойчивости к NaN/nodata), остальные случайно
        worst = meta.sort_values("valid_frac").head(max(1, n // 5))
        rest = meta.drop(worst.index).sample(n - len(worst), random_state=rng)
        chosen = meta.loc[sorted(worst.index.tolist() + rest.index.tolist())]
        for sub, _ in LAYERS[kind].values():
            os.makedirs(win_path(out_dir / kind / sub), exist_ok=True)  # `aux` — зарезервированное имя в Windows
        for cid in chosen["chip_id"]:
            for layer in LAYERS[kind].values():
                shutil.copyfile(chip_path(train_dir, kind, layer, cid), chip_path(out_dir, kind, layer, cid))
        test_meta = chosen[TEST_META_COLUMNS].copy()
        if kind == "af":
            test_meta["cloud_frac"] = np.nan  # как в настоящем тесте
        test_meta.to_csv(out_dir / kind / "meta.csv", index=False)
        classes = (1,) if kind == "af" else (1, 2, 3)
        rows += [(cid, k, "") for cid in chosen["chip_id"] for k in classes]
        print(f"{kind.upper()}: {len(chosen)} чипов → {out_dir / kind}")

    write_submission(rows, out_dir / "sample_submission.csv")
    print(f"sample_submission.csv: {len(rows)} строк")

    # ФЕЙКОВЫЙ шаблон по всем train-чипам: для OOF-оценки через evaluate_submission и тестов,
    # пока нет настоящего baseline/sample_submission.csv. НЕ является файлом организаторов.
    full = [(c, 1, "") for c in read_meta(train_dir, "af")["chip_id"]]
    full += [(c, k, "") for c in read_meta(train_dir, "bs")["chip_id"] for k in (1, 2, 3)]
    fake = ROOT / "baseline" / "sample_submission_FAKE.csv"
    write_submission(full, fake)
    print(f"{fake}: {len(full)} строк (шаблон по train, только для тестов)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
