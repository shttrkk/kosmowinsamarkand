"""AFPredictor: кандидаты → признаки → среднее 5 фолд-моделей LightGBM → порог. Работает на CPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...features.af_features import FEATURE_NAMES, af_feature_rows, candidate_mask, chip_valid
from ...utils import REPO_ROOT, auto_threads

META_NAME = "af_lgbm_meta.json"


class AFPredictor:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        wd = Path(cfg.get("weights_dir", "weights"))
        self.weights_dir = wd if wd.is_absolute() else REPO_ROOT / wd
        self.num_threads = auto_threads(cfg.get("num_threads"))  # auto → min(число ядер, 16); на результат не влияет
        self.threshold = cfg.get("threshold")            # None → порог из meta (подобран по OOF)
        self.boosters: list = []
        self.cols: np.ndarray | None = None

    def load(self) -> "AFPredictor":
        return self.load_meta().load_boosters()

    def load_meta(self) -> "AFPredictor":
        """Быстрая часть загрузки (json): после неё уже можно считать признаки (`candidates_and_rows`)."""
        meta = json.loads((self.weights_dir / META_NAME).read_text(encoding="utf-8"))
        missing = [n for n in meta["feature_names"] if n not in FEATURE_NAMES]
        if missing:
            raise RuntimeError(f"веса AF обучены на признаках, которых нет в коде: {missing[:5]} — переобучите модель")
        self.cols = np.array([FEATURE_NAMES.index(n) for n in meta["feature_names"]])
        if self.threshold is None:
            self.threshold = float(meta["threshold"])
        self._model_files = [self.weights_dir / f for f in meta["models"]]
        absent = [f.name for f in self._model_files if not f.exists()]
        if absent:
            raise FileNotFoundError(f"нет файлов весов AF: {absent}")
        return self

    def load_boosters(self) -> "AFPredictor":
        """Медленная часть (импорт lightgbm + разбор моделей, ~1 с); inference.py выполняет её в фоновом потоке."""
        import lightgbm as lgb

        self.boosters = [lgb.Booster(model_file=str(f)) for f in self._model_files]
        return self

    # --- этапы по отдельности: inference.py считает признаки в потоках, а predict вызывает одним батчем
    def candidates_and_rows(self, chip: dict) -> tuple[np.ndarray, np.ndarray]:
        cand = candidate_mask(chip["viirs"], chip_valid(chip))
        if not cand.any():
            return cand, np.zeros((0, len(self.cols)), np.float32)
        return cand, af_feature_rows(chip, cand)[:, self.cols]

    def predict_rows(self, rows: np.ndarray) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros(0, np.float64)
        return np.mean([b.predict(rows, num_threads=self.num_threads) for b in self.boosters], axis=0)

    def predict_proba(self, chip: dict) -> np.ndarray:
        cand, rows = self.candidates_and_rows(chip)
        prob = np.zeros(cand.shape, np.float32)
        prob[cand] = self.predict_rows(rows)
        return prob

    def predict(self, chip: dict) -> np.ndarray:
        """Маска uint8 (H, W) 0/1; valid==0 и NaN в I4/I5 → всегда 0 (такие пиксели не бывают кандидатами)."""
        return (self.predict_proba(chip) > self.threshold).astype(np.uint8)

    def predict_many(self, chips: list[dict]) -> list[np.ndarray]:
        return self.masks_from_parts([self.candidates_and_rows(c) for c in chips])

    def masks_from_parts(self, parts: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Один вызов predict на все чипы (накладные расходы LightGBM на вызов ≫ стоимости строки)."""
        prob = self.predict_rows(np.concatenate([r for _, r in parts])) if parts else np.zeros(0)
        out, pos = [], 0
        for cand, rows in parts:
            mask = np.zeros(cand.shape, np.uint8)
            mask[cand] = prob[pos:pos + len(rows)] > self.threshold
            pos += len(rows)
            out.append(mask)
        return out


_DEFAULT: AFPredictor | None = None


def predict_af(chip: dict) -> np.ndarray:
    """Выход модуля AF: маска uint8 (256, 256) 0/1 (ленивая загрузка весов по умолчанию)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = AFPredictor().load()
    return _DEFAULT.predict(chip)
