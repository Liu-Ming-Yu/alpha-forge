"""Gradient-boosted-tree alpha ranker (XGBoost), GPU-capable.

A nonlinear learner for the walk-forward driver — the qlib-inspired upgrade from
the linear IC-weighted ranker. A tree ensemble captures feature *interactions*
the linear ranker cannot (e.g. momentum that only pays in a low-volatility
regime). That is the mechanism by which a richer model can break the consecutive
negative-IC fold streaks that bind the long-only portfolio-candidate eligibility
gate — the constraint the dial and the regime overlay could not move.

**Why XGBoost, not LightGBM.** XGBoost is already a project dependency (the
``ml`` extra) and ships first-class CUDA support on Windows out of the box; GPU
LightGBM needs a custom OpenCL/CUDA build. The architectural point — a
boosted-tree ranker behind the ``AlphaModel`` protocol — is identical either way,
and a LightGBM variant would be a drop-in sibling of this module.

**GPU.** ``device="auto"`` probes for a working CUDA build once per process and
falls back to CPU if unavailable. At universe-300 scale (~280k train rows ×
~36 features × 63 folds) the GPU win is modest — CPU XGBoost is already fast — so
the fallback is production-safe; GPU becomes genuinely *necessary* only for the
heavier sequence models that will share this same protocol.
"""

from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    import xgboost as xgb

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

Device = Literal["auto", "cuda", "cpu"]
Objective = Literal["regression", "rank"]

#: Map the public objective to the XGBoost objective string. ``regression`` is
#: MSE on the forward-return level (Arm I); ``rank`` is pairwise learning-to-rank
#: with per-date query groups (Arm J) — it optimizes the *cross-sectional
#: ordering* the IC gate actually measures, which MSE-on-levels does not.
_OBJECTIVE_XGB: dict[Objective, str] = {
    "regression": "reg:squarederror",
    "rank": "rank:pairwise",
}

#: Conservative defaults for noisy financial cross-sections: shallow trees, row
#: and column subsampling, a non-trivial ``min_child_weight``, and L2
#: regularization to resist overfitting the in-sample IC. The objective is set
#: per-fit from the model's ``objective`` mode (see ``_OBJECTIVE_XGB``), not
#: hardcoded here, so the regression and rank arms share these defaults.
_DEFAULT_PARAMS: dict[str, Any] = {
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
    "verbosity": 0,
}
_DEFAULT_NUM_BOOST_ROUND = 300
_DEFAULT_SEED = 42

#: Per-process CUDA capability cache. ``None`` = not probed yet. Lives at module
#: scope so each ProcessPoolExecutor worker probes at most once; never shared
#: across processes (spawn re-imports the module).
_cuda_probe_cache: bool | None = None


def _require_xgboost() -> ModuleType:
    try:
        import xgboost as xgb  # noqa: PLC0415 — lazy so the package imports without the ml extra
    except ImportError as exc:  # pragma: no cover - exercised only without xgboost
        raise ImportError(
            "GradientBoostedRanker requires xgboost. Install the 'ml' extra: "
            "pip install 'quant-platform[ml]'"
        ) from exc
    return xgb


def _cuda_available() -> bool:
    """Return True iff a tiny GPU training round succeeds (cached per process)."""
    global _cuda_probe_cache
    if _cuda_probe_cache is not None:
        return _cuda_probe_cache
    try:
        xgb = _require_xgboost()
        dmat = xgb.DMatrix(np.zeros((8, 1), dtype="float32"), label=np.zeros(8, dtype="float32"))
        xgb.train(
            {"device": "cuda", "tree_method": "hist", "verbosity": 0},
            dmat,
            num_boost_round=1,
        )
        _cuda_probe_cache = True
    except Exception:  # noqa: BLE001 - any GPU/driver/build failure => fall back to CPU
        _cuda_probe_cache = False
    return _cuda_probe_cache


def _select_feature_names(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str] | None,
) -> list[str]:
    """Mirror ``ranker_metrics._selected_feature_names`` so a GBDT arm sees the
    exact same feature set (and order) the linear ranker would on the same fold."""
    available = {name for row in samples for name in row.features}
    if feature_names is None:
        return sorted(available)
    return sorted({name for name in feature_names if name in available})


def _matrix(samples: Sequence[SupervisedAlphaSample], names: Sequence[str]) -> np.ndarray:
    """Dense float32 design matrix; absent feature -> 0.0 (legacy convention)."""
    return np.array(
        [[float(row.features.get(name, 0.0)) for name in names] for row in samples],
        dtype="float32",
    )


class _FittedGBDT:
    """Immutable per-fold fit: a trained booster + frozen importance weights."""

    __slots__ = ("_booster", "_names", "_weights", "_device")

    def __init__(
        self,
        booster: xgb.Booster,
        names: Sequence[str],
        weights: Mapping[str, float],
        device: str,
    ) -> None:
        self._booster = booster
        self._names = list(names)
        self._weights = dict(weights)
        self._device = device

    def score(self, samples: Sequence[SupervisedAlphaSample]) -> list[float]:
        if not samples:
            return []
        xgb = _require_xgboost()
        dmat = xgb.DMatrix(_matrix(samples, self._names), feature_names=self._names)
        # Pin inference to the same device the booster trained on so XGBoost
        # doesn't warn about data/booster device mismatch.
        self._booster.set_param({"device": self._device})
        return [float(value) for value in self._booster.predict(dmat)]

    def feature_weights(self) -> Mapping[str, float]:
        return dict(self._weights)


class GradientBoostedRanker:
    """XGBoost gradient-boosted ranker behind the :class:`AlphaModel` protocol.

    Refit per fold on the training window; scores test (and train, for the
    volatility scale) rows. ``objective="regression"`` (Arm I) minimizes MSE on
    the forward-return level; ``objective="rank"`` (Arm J) uses pairwise
    learning-to-rank with each date as a query group — optimizing the
    cross-sectional ordering the IC gate measures. The model_version stamped in
    evidence (:attr:`name`) encodes the objective but not the hardware.
    """

    def __init__(
        self,
        *,
        objective: Objective = "regression",
        device: Device = "auto",
        num_boost_round: int = _DEFAULT_NUM_BOOST_ROUND,
        params: Mapping[str, Any] | None = None,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        if objective not in _OBJECTIVE_XGB:
            raise ValueError(f"unsupported objective: {objective!r}")
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"unsupported device: {device!r}")
        if num_boost_round < 1:
            raise ValueError("num_boost_round must be >= 1")
        self._objective: Objective = objective
        self._device: Device = device
        self._num_boost_round = num_boost_round
        self._params: dict[str, Any] = {**_DEFAULT_PARAMS, **(params or {})}
        self._seed = seed
        self.name = "xgboost-gbdt-rank-v1" if objective == "rank" else "xgboost-gbdt-v1"

    def _resolve_device(self) -> str:
        if self._device == "auto":
            return "cuda" if _cuda_available() else "cpu"
        return self._device

    def fit(
        self,
        train: Sequence[SupervisedAlphaSample],
        feature_names: Sequence[str] | None = None,
    ) -> _FittedGBDT:
        xgb = _require_xgboost()
        names = _select_feature_names(train, feature_names)
        if not names:
            raise ValueError("GradientBoostedRanker.fit: no features available in training samples")
        # rank:pairwise needs each cross-section (date) as a query group, with
        # the group's rows contiguous in the matrix. Sort by as_of so the groups
        # are well-formed regardless of caller order; regression ignores groups
        # and keeps the caller's order.
        is_rank = self._objective == "rank"
        rows = sorted(train, key=lambda row: row.as_of) if is_rank else list(train)
        features = _matrix(rows, names)
        labels = np.array([float(row.forward_return) for row in rows], dtype="float32")
        device = self._resolve_device()
        params = {
            **self._params,
            "objective": _OBJECTIVE_XGB[self._objective],
            "device": device,
            "seed": self._seed,
        }
        dtrain = xgb.DMatrix(features, label=labels, feature_names=names)
        if is_rank:
            groups = [len(list(g)) for _, g in groupby(rows, key=lambda r: r.as_of)]
            dtrain.set_group(groups)
        booster = xgb.train(params, dtrain, num_boost_round=self._num_boost_round)
        return _FittedGBDT(booster, names, _importance_weights(booster, names), device)


def _scalar(value: float | list[float]) -> float:
    """Coerce an xgboost importance value to a float.

    ``Booster.get_score`` is typed ``dict[str, float | list[float]]`` (the list
    case is for multi-output boosters); for the scalar regressor used here every
    value is a float, but narrow defensively so mypy and a future multi-output
    swap both stay sound.
    """
    return float(sum(value)) if isinstance(value, list) else float(value)


def _importance_weights(booster: xgb.Booster, names: Sequence[str]) -> dict[str, float]:
    """Normalized gain importances over ``names`` (a reporting proxy).

    XGBoost only returns scores for features that were actually split on, so
    unused features get 0.0. When the booster produced no splits at all (e.g. a
    degenerate fold), fall back to equal weights so ``feature_stability`` and the
    ``selected_weights`` evidence field stay well-formed.
    """
    gain = booster.get_score(importance_type="gain")
    total = sum(_scalar(value) for value in gain.values())
    if total <= 0:
        equal = 1.0 / max(1, len(names))
        return {name: equal for name in names}
    return {name: _scalar(gain.get(name, 0.0)) / total for name in names}
