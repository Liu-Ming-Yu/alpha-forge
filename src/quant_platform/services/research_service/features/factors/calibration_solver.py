"""Solver and orchestration for factor-weight calibration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import structlog

from quant_platform.services.research_service.features.factors.calibration_models import (
    ALPHA_BLOC,
    MOMENTUM_BLOC,
    CalibratedWeights,
    CalibrationSample,
)
from quant_platform.services.research_service.reports.statistics import spearman_ic

if TYPE_CHECKING:
    from collections.abc import Sequence

log = structlog.get_logger(__name__)

_LAMBDA_CANDIDATES: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.1)


def _nnls_ridge(
    x_matrix: np.ndarray,
    y_vector: np.ndarray,
    l2: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Projected-gradient NNLS with L2 penalty."""
    n_samples, n_features = x_matrix.shape
    if n_samples == 0:
        return np.zeros(n_features)

    gram = x_matrix.T @ x_matrix + l2 * np.eye(n_features)
    rhs = x_matrix.T @ y_vector
    lip = float(np.trace(gram)) or 1.0
    step = 1.0 / lip

    weights = np.zeros(n_features)
    for _ in range(max_iter):
        grad = gram @ weights - rhs
        updated = np.maximum(weights - step * grad, 0.0)
        if np.linalg.norm(updated - weights) < tol:
            weights = updated
            break
        weights = updated
    return weights


def _fit_bloc(
    samples: Sequence[CalibrationSample],
    feature_names: Sequence[str],
    l2: float,
) -> tuple[dict[str, float], float, int]:
    """Fit one feature bloc and return weights, r-squared, and sample size."""
    if not samples:
        return {name: 0.0 for name in feature_names}, 0.0, 0

    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    for sample in samples:
        invalid_features = any(
            not np.isfinite(sample.features.get(name, np.nan)) for name in feature_names
        )
        if invalid_features or not np.isfinite(sample.forward_return):
            continue
        x_rows.append([float(sample.features.get(name, 0.0)) for name in feature_names])
        y_vals.append(float(sample.forward_return))

    if not x_rows:
        log.warning(
            "factor_calibration.empty_bloc_after_nan_filter",
            feature_names=list(feature_names),
        )
        return {name: 0.0 for name in feature_names}, 0.0, 0

    x_matrix = np.asarray(x_rows, dtype=float)
    y_vector = np.asarray(y_vals, dtype=float)
    weights = _nnls_ridge(x_matrix, y_vector, l2=l2)

    predictions = x_matrix @ weights
    ss_res = float(np.sum((y_vector - predictions) ** 2))
    ss_tot = float(np.sum((y_vector - y_vector.mean()) ** 2)) or 1.0
    r_squared = 1.0 - ss_res / ss_tot

    total = float(weights.sum())
    if total > 0:
        weights = weights / total
    return dict(zip(feature_names, weights.tolist(), strict=True)), r_squared, len(y_vals)


def _spearman_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank IC between predicted scores and actual returns."""
    return spearman_ic(predicted.tolist(), actual.tolist())


def _select_lambda_by_ic(
    samples: Sequence[CalibrationSample],
    fallback: float,
) -> float:
    """Select the L2 penalty by leave-one-date-out IC."""
    all_features = MOMENTUM_BLOC + ALPHA_BLOC
    dates = sorted({sample.as_of for sample in samples})
    if len(dates) < 2:
        return fallback

    samples_by_date: dict[datetime, list[CalibrationSample]] = {}
    for sample in samples:
        samples_by_date.setdefault(sample.as_of, []).append(sample)

    best_lambda = fallback
    best_ic = float("-inf")
    for lam in _LAMBDA_CANDIDATES:
        ic_scores: list[float] = []
        for held_date in dates:
            train = [sample for sample in samples if sample.as_of != held_date]
            test = samples_by_date[held_date]
            if not train or len(test) < 2:
                continue
            w_dict, _, _ = _fit_bloc(train, all_features, lam)
            w_vec = np.array([w_dict.get(feature, 0.0) for feature in all_features])
            if w_vec.sum() == 0:
                continue
            preds = np.array(
                [
                    sum(
                        sample.features.get(feature, 0.0) * w_vec[idx]
                        for idx, feature in enumerate(all_features)
                    )
                    for sample in test
                ]
            )
            actuals = np.array([sample.forward_return for sample in test])
            ic_scores.append(_spearman_ic(preds, actuals))
        if ic_scores:
            mean_ic = float(np.mean(ic_scores))
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_lambda = lam

    log.info(
        "factor_calibration.lambda_selected",
        best_lambda=best_lambda,
        best_ic=best_ic,
        candidates=list(_LAMBDA_CANDIDATES),
        n_dates=len(dates),
    )
    return best_lambda


def calibrate(
    samples: Sequence[CalibrationSample],
    *,
    horizon_days: int = 21,
    l2_lambda: float = 1e-3,
    momentum_bloc_scale: float = 0.90,
    alpha_bloc_scale: float = 0.10,
    as_of: datetime | None = None,
    cross_validate_lambda: bool = True,
) -> CalibratedWeights:
    """Fit both feature blocs and return the combined calibrated weights."""
    if abs(momentum_bloc_scale + alpha_bloc_scale - 1.0) > 1e-6:
        raise ValueError("momentum_bloc_scale + alpha_bloc_scale must sum to 1.0")

    effective_lambda = l2_lambda
    if cross_validate_lambda and len(samples) > 1:
        effective_lambda = _select_lambda_by_ic(samples, l2_lambda)

    momentum_weights, r2_mom, n_mom = _fit_bloc(samples, MOMENTUM_BLOC, effective_lambda)
    alpha_weights, r2_alpha, n_alpha = _fit_bloc(samples, ALPHA_BLOC, effective_lambda)

    weights: dict[str, float] = {}
    for name, weight in momentum_weights.items():
        weights[name] = float(weight * momentum_bloc_scale)
    for name, weight in alpha_weights.items():
        weights[name] = float(weight * alpha_bloc_scale)

    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {name: weight / total_weight for name, weight in weights.items()}

    when = as_of or max((sample.as_of for sample in samples), default=datetime.now(tz=UTC))
    return CalibratedWeights(
        as_of=when,
        weights=weights,
        sample_size=max(n_mom, n_alpha),
        r_squared_momentum=r2_mom,
        r_squared_alpha=r2_alpha,
        l2_lambda=effective_lambda,
        horizon_days=horizon_days,
        metadata={
            "momentum_bloc_scale": momentum_bloc_scale,
            "alpha_bloc_scale": alpha_bloc_scale,
            "momentum_rows": n_mom,
            "alpha_rows": n_alpha,
            "cross_validated_lambda": cross_validate_lambda,
        },
    )


__all__ = ["calibrate"]
