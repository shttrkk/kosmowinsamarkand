"""Адаптеры моделей для слоя `pred`: финальные AFPredictor/BSPredictor, иначе заглушки на правилах.

Контракт модулей моделей (промпты 3–4): `Predictor(config).load()`, `.predict(chip) -> mask`;
опционально `.predict_proba(chip) -> (H, W)` — используется как `confidence` термоточек.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from firemon.utils import REPO_ROOT, get_logger, load_config

log = get_logger("service.predictors")


def _config(section: str) -> dict:
    try:
        return load_config(REPO_ROOT / "configs" / "inference.yaml").get(section, {}) or {}
    except Exception:  # сервис не должен падать из-за конфига инференса
        return {}


def _load(module: str, cls: str, section: str):
    import importlib

    predictor = getattr(importlib.import_module(module), cls)(_config(section))
    predictor.load()
    return predictor


def get_af_predictor() -> tuple[Callable[[dict], tuple[np.ndarray, np.ndarray | None]], str]:
    """→ (fn(chip) -> (mask, proba|None), имя модели)."""
    try:
        p = _load("firemon.models.af.predictor", "AFPredictor", "af")

        def run(chip):
            proba = p.predict_proba(chip) if hasattr(p, "predict_proba") else None
            return np.asarray(p.predict(chip)), proba

        return run, "AFPredictor"
    except Exception as e:
        log.warning("AFPredictor недоступен (%s) — слой pred строится заглушкой на правилах", e)
        from firemon.models.af.rules import predict_af_rules

        cfg = _config("af")
        return (lambda chip: (predict_af_rules(chip, cfg.get("i4_min", 320.0), cfg.get("di45_min", 15.0)), None),
                "rules-stub")


def get_bs_predictor() -> tuple[Callable[[dict], np.ndarray], str]:
    try:
        # параметры BS-модели (пути к весам, TTA, стек) лежат в configs/bs_inference.yaml — тот же конфиг, что у
        # inference.py; секции `bs` общего inference.yaml недостаточно: без путей к весам предиктор молча уходит в правила
        from firemon.models.bs.infer import load_bs_config
        from firemon.models.bs.predictor import BSPredictor

        p = BSPredictor(load_bs_config(_config("bs"))).load()
        if not p.models and p.booster is None:  # аварийный режим предиктора = те же правила: в манифест — честно
            raise RuntimeError("веса BS-моделей не загружены")
        return (lambda chip: np.asarray(p.predict(chip))), "BSPredictor"
    except Exception as e:
        log.warning("BSPredictor недоступен (%s) — слой pred строится заглушкой на правилах", e)
        from firemon.models.bs.rules import predict_bs_rules

        thr = tuple(_config("bs").get("dnbr_thresholds", (0.10, 0.27, 0.44)))
        return (lambda chip: predict_bs_rules(chip, thr)), "rules-stub"
