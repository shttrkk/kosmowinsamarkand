"""BSPredictor — инференс BS: UNet (GPU/CPU) → [ансамбль с пиксельным LGBM] → [стек-LGBM 2-го уровня]
→ декодирование порогами → постобработка.

    pred = BSPredictor(load_config("configs/bs_inference.yaml")["bs"]).load()
    mask = pred.predict(read_bs_chip(data_dir, chip_id, with_mask=False))      # (512, 512) uint8 0..3

Нет файлов весов → аварийный режим на правилах dNBR (inference.py обязан отработать с кодом 0).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ...features.bs_features import (FEATURE_GROUPS, compute_bs_features, feature_names, groups_for,
                                     load_lc_medians, select_channels)
from ...utils import REPO_ROOT, get_logger
from .common import decode_probs
from .postprocess import postprocess_bs
from .rules import predict_bs_rules

log = get_logger("bs.predictor")

DEFAULTS: dict = {
    "device": "auto",            # auto | cpu | cuda
    "use_unet": True,
    "use_stack": True,           # LGBM 2-го уровня поверх вероятностей UNet (основной режим)
    "use_lgbm": False,           # пиксельный LGBM 1-го уровня в среднем с UNet (вариант «ансамбль»)
    "tta": False,                # TTA (4 отражения): прирост < 0.005 BS-части → выключен, чтобы результат не зависел от устройства
    "tta_cpu": False,            # на CPU TTA по умолчанию выключен (время)
    "gpu_half": False,           # fp32 без TF32 на GPU: маски побайтно совпадают с CPU (fp16 давал ~25 отличающихся px/чип)
    "prob_decimals": None,       # округление вероятностей UNet до N знаков перед стеком (сглаживает различия устройств)
    "batch_size": 8,
    "num_threads": 8,            # потоки чтения/признаков/постобработки и torch на CPU
    "lgbm_threads": 16,          # потоки LightGBM (один вызов predict на батч)
    "unet_weights": ["weights/bs_unet.pt"],
    "stack_weights": "weights/bs_stack.txt",
    "stack_gate": 0.03,
    "lgbm_weights": "weights/bs_lgbm.txt",
    "lc_medians": "weights/bs_lc_medians.json",
    # decode / postprocess / ens_weights: None → из decode_file (пишет ensemble.py по OOF) → значения ниже
    "decode_file": "weights/bs_decode.json",
    "ens_weights": None,
    "decode": None,
    "postprocess": None,
    "rules_thresholds": [0.10, 0.27, 0.44],
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


class BSPredictor:
    def __init__(self, config: dict | None = None):
        self.cfg = {**DEFAULTS, **(config or {})}
        self.models: list = []
        self.channels: list[str] | None = None
        self.booster = None      # пиксельный LGBM
        self.stack = None        # LGBM 2-го уровня
        self.loaded = False
        self.device = None

    # ---------- загрузка ----------
    def _load_booster(self, key: str, what: str):
        p = _resolve(self.cfg[key])
        if not p.exists():
            log.warning("нет весов %s %s — отключено", what, p)
            return None
        import lightgbm as lgb

        return lgb.Booster(model_file=str(p))

    def load(self) -> "BSPredictor":
        cfg = self.cfg
        fitted = {}
        if cfg.get("decode_file") and _resolve(cfg["decode_file"]).exists():
            fitted = json.loads(_resolve(cfg["decode_file"]).read_text(encoding="utf-8"))
        fallback = {"decode": {"t_burn": 0.5, "w": [1.0, 1.0, 1.0]}, "postprocess": {},
                    "ens_weights": {"unet": 0.7, "lgbm": 0.3}}
        for key, default in fallback.items():
            if cfg.get(key) is None:
                cfg[key] = fitted.get(key, default)
        self.lc_medians = load_lc_medians(_resolve(cfg["lc_medians"]))
        if cfg["use_unet"]:
            paths = [_resolve(p) for p in cfg["unet_weights"]]
            missing = [p for p in paths if not p.exists()]
            if missing:
                log.warning("нет весов UNet %s — UNet отключён", [str(m) for m in missing])
            else:
                self._load_unets(paths)
        if cfg["use_lgbm"]:
            self.booster = self._load_booster("lgbm_weights", "LightGBM")
        if cfg["use_stack"] and self.models:
            self.stack = self._load_booster("stack_weights", "стек-LGBM")
        if not self.models and self.booster is None:
            log.warning("BS: модели не загружены — аварийный режим (правила dNBR)")
        # какие группы признаков считать
        if self.booster is not None:
            self.groups = list(FEATURE_GROUPS)
        elif self.stack is not None:
            self.groups = [g for g in FEATURE_GROUPS if g != "lcrel"]
        else:
            self.groups = groups_for(self.channels or [])
        self.names = feature_names(self.groups)
        self._stack_cols = None
        if self.booster is not None:  # пиксельная LGBM: базовые признаки + dNBR_rel (порядок как при обучении)
            from .lgbm import BASE_GROUPS

            self._lgbm_cols = [self.names.index(n) for n in feature_names(BASE_GROUPS) + ["dNBR_rel"]]
        self.loaded = True
        return self

    def _load_unets(self, paths: list[Path]) -> None:
        import torch

        from .unet import build_unet

        want = self.cfg["device"]
        use_cuda = want in ("auto", "cuda") and torch.cuda.is_available()
        half = use_cuda and bool(self.cfg.get("gpu_half", False))
        if use_cuda and not half:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if want == "cuda" and not use_cuda:
            log.warning("device=cuda, но CUDA недоступна — CPU")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if not use_cuda:
            torch.set_num_threads(int(self.cfg["num_threads"]))
        dt = torch.float16 if half else torch.float32
        self.channels = []
        for p in paths:  # чекпойнты ансамбля могут быть обучены на разных наборах каналов
            ck = torch.load(p, map_location="cpu", weights_only=False)
            self.channels += [c for c in ck["channels"] if c not in self.channels]
            model = build_unet(ck["encoder"], len(ck["channels"]), pretrained=False, arch=ck.get("arch", "unet"))
            model.load_state_dict(ck["state_dict"])
            model = model.to(self.device).eval()
            model = model.half() if half else model.float()
            mean = torch.tensor(ck["norm"]["mean"], dtype=dt, device=self.device).view(1, -1, 1, 1)
            std = torch.tensor(ck["norm"]["std"], dtype=dt, device=self.device).view(1, -1, 1, 1)
            self.models.append({"model": model, "loss": ck.get("loss", "ce_dice"), "channels": list(ck["channels"]),
                                "mean": mean, "std": std})

    # ---------- предсказание ----------
    def _features(self, chip: dict) -> np.ndarray:
        return compute_bs_features(chip, lc_medians=self.lc_medians, groups=self.groups)

    def _unet_probs(self, feats: list[np.ndarray], xs: list[np.ndarray] | None = None) -> np.ndarray:
        import torch

        from .unet import predict_probs

        x = np.stack(xs if xs is not None else [self._unet_input(f) for f in feats])
        tta = bool(self.cfg["tta"] if self.device.type == "cuda" else self.cfg["tta_cpu"])
        with torch.inference_mode():
            xb = torch.from_numpy(x).to(self.device)
            out = 0.0
            for m in self.models:
                idx = [self.channels.index(c) for c in m["channels"]]
                xm = xb if idx == list(range(xb.shape[1])) else xb[:, idx]
                xm = (xm.to(m["mean"].dtype) - m["mean"]) / m["std"]
                out = out + predict_probs(m["model"], xm, m["loss"], tta)
            out = (out / len(self.models)).float().cpu().numpy()
        nd = self.cfg.get("prob_decimals")
        if nd is not None:
            out = np.round(out, int(nd))
        return out

    def _lgbm_blend(self, feats: np.ndarray, p_unet: np.ndarray | None) -> np.ndarray:
        """Среднее UNet и пиксельного LGBM. Без стека LGBM считается только там, где ансамбль в принципе
        может дать «гарь»: w_u·pu + w_l·1 > t ⇔ pu > (t − w_l)/w_u (в остальных пикселях ответ 0 в любом случае)."""
        nthr = int(self.cfg["num_threads"])
        f = feats[self._lgbm_cols].reshape(len(self._lgbm_cols), -1)
        if p_unet is None:
            return self.booster.predict(f.T, num_threads=nthr).T.reshape(4, *feats.shape[1:]).astype(np.float32)
        wu, wl = float(self.cfg["ens_weights"]["unet"]), float(self.cfg["ens_weights"]["lgbm"])
        wu, wl = wu / (wu + wl), wl / (wu + wl)
        t = float(self.cfg["stack_gate"] if self.stack is not None else self.cfg["decode"]["t_burn"])
        pu = p_unet.reshape(4, -1)
        sel = (1.0 - pu[0]) > (t - wl) / wu
        out = pu.copy()
        out[0, ~sel], out[1:, ~sel] = 1.0, 0.0
        if sel.any():
            out[:, sel] = wu * pu[:, sel] + wl * self.booster.predict(f[:, sel].T, num_threads=nthr).T.astype(np.float32)
        return out.reshape(p_unet.shape)

    def _stack_rows(self, feats: np.ndarray, p: np.ndarray):
        """Строки 2-го уровня для пикселей-ворот одного чипа → (X (n,F) float32, индексы пикселей)."""
        from .stack import N_UNET_FEATS, stack_feature_names, unet_features

        if self._stack_cols is None:
            base = stack_feature_names()[:-N_UNET_FEATS]
            cols = [self.names.index(n) for n in base]
            self._stack_cols = slice(0, len(cols)) if cols == list(range(len(cols))) else cols
        idx = np.flatnonzero((1.0 - p[0]).ravel() > float(self.cfg["stack_gate"]))
        if not len(idx):
            return None, idx
        f = feats.reshape(feats.shape[0], -1)[self._stack_cols]
        u = unet_features(p).reshape(N_UNET_FEATS, -1)
        X = np.empty((len(idx), f.shape[0] + N_UNET_FEATS), np.float32)
        X[:, :f.shape[0]] = f[:, idx].T
        X[:, f.shape[0]:] = u[:, idx].T
        return X, idx

    def _stack_batch(self, feats: list[np.ndarray], probs: list[np.ndarray], ex: ThreadPoolExecutor) -> list[np.ndarray]:
        rows = list(ex.map(self._stack_rows, feats, probs))
        outs = []
        for p in probs:
            o = np.zeros_like(p)
            o[0] = 1.0
            outs.append(o)
        parts = [X for X, _ in rows if X is not None]
        if parts:
            pred = self.stack.predict(np.concatenate(parts), num_threads=int(self.cfg["lgbm_threads"])).astype(np.float32)
            pos = 0
            for o, (X, idx) in zip(outs, rows):
                if X is not None:
                    o.reshape(4, -1)[:, idx] = pred[pos:pos + len(idx)].T
                    pos += len(idx)
        return outs

    def _stack(self, feats: np.ndarray, p: np.ndarray) -> np.ndarray:
        with ThreadPoolExecutor(1) as ex:
            return self._stack_batch([feats], [p], ex)[0]

    def prepare(self, chip: dict):
        """Признаки чипа + готовый вход UNet (float16). Вызывается из фоновых потоков — numpy/cv2 отпускают GIL."""
        if not self.loaded:
            self.load()
        if not self.models and self.booster is None:
            return None
        feats = self._features(chip)
        return feats, (self._unet_input(feats) if self.models else None)

    def _unet_input(self, feats: np.ndarray) -> np.ndarray:
        return np.clip(select_channels(feats, self.names, self.channels), -60000, 60000).astype(np.float16)

    def first_stage(self, prepared: list) -> list:
        """Этап 1 (GPU/torch): вероятности UNet для батча подготовленных чипов (результаты `prepare`)."""
        feats = [f for f, _ in prepared]
        xs = [x for _, x in prepared]
        return list(self._unet_probs(feats, xs)) if self.models else [None] * len(feats)

    def second_stage(self, chips: list[dict], prepared: list, probs: list, ex: ThreadPoolExecutor) -> list[np.ndarray]:
        """Этап 2 (CPU): [пиксельный LGBM] → стек-LGBM → декодирование → постобработка."""
        feats = [f for f, _ in prepared]
        if self.booster is not None:
            probs = [self._lgbm_blend(f, p) for f, p in zip(feats, probs)]
        if self.stack is not None:
            probs = self._stack_batch(feats, probs, ex)
        return list(ex.map(self._finish, chips, probs))

    def predict_proba_batch(self, chips: list[dict], feats: list[np.ndarray] | None = None,
                            ex: ThreadPoolExecutor | None = None) -> list[np.ndarray] | None:
        if not self.loaded:
            self.load()
        if not self.models and self.booster is None:
            return None
        own = ex is None
        ex = ex or ThreadPoolExecutor(max(1, min(int(self.cfg["num_threads"]), len(chips))))
        try:
            if feats is None:
                feats = list(ex.map(self._features, chips))
            probs = list(self._unet_probs(feats)) if self.models else [None] * len(chips)
            if self.booster is not None:
                probs = [self._lgbm_blend(f, p) for f, p in zip(feats, probs)]
            if self.stack is not None:
                probs = self._stack_batch(feats, probs, ex)
            return probs
        finally:
            if own:
                ex.shutdown()

    def _finish(self, chip: dict, p: np.ndarray) -> np.ndarray:
        dec = self.cfg["decode"]
        mask = decode_probs(p, dec.get("t_burn"), tuple(dec.get("w", (1, 1, 1))))
        return postprocess_bs(mask, chip, self.cfg.get("postprocess"), probs=p, decode=dec)

    def predict_batch(self, chips: list[dict], feats: list[np.ndarray] | None = None,
                      ex: ThreadPoolExecutor | None = None) -> list[np.ndarray]:
        probs = self.predict_proba_batch(chips, feats, ex)
        if probs is None:
            return [predict_bs_rules(c, tuple(self.cfg["rules_thresholds"])) for c in chips]
        if ex is not None:
            return list(ex.map(self._finish, chips, probs))
        return [self._finish(c, p) for c, p in zip(chips, probs)]

    def predict(self, chip: dict) -> np.ndarray:
        return self.predict_batch([chip])[0]


def predict_bs(chip: dict, _cache: dict = {}) -> np.ndarray:  # noqa: B006 — намеренный кэш предиктора
    """Функция из ТЗ задачи: chip → (512, 512) uint8 0..3, конфиг configs/bs_inference.yaml."""
    if "p" not in _cache:
        from .infer import load_bs_config

        _cache["p"] = BSPredictor(load_bs_config()).load()
    return _cache["p"].predict(chip)
