"""Pure optimizer math helpers."""

from __future__ import annotations

import math
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from quant_platform.core.domain.portfolio import (
    PortfolioRiskModel,
    PortfolioTarget,
    RiskSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime


def covariance_scale(
    weights: dict[uuid.UUID, Decimal],
    risk_model: PortfolioRiskModel,
    halflife_days: int = 0,
    target_as_of: datetime | None = None,
) -> dict[uuid.UUID, Decimal]:
    decay = Decimal("1")
    if halflife_days > 0 and target_as_of is not None and risk_model.as_of is not None:
        age_days = (target_as_of - risk_model.as_of).days
        if age_days > 0:
            decay = Decimal(str(math.exp(-math.log(2) * age_days / halflife_days)))

    variances: dict[uuid.UUID, float] = {}
    for instrument_id in weights:
        raw = risk_model.covariance.get((instrument_id, instrument_id))
        if raw is not None and raw > Decimal("0"):
            variances[instrument_id] = float(raw * decay)
    if len(variances) != len(weights):
        return weights

    vols = np.array([math.sqrt(variances[instrument_id]) for instrument_id in weights])
    if not np.all(np.isfinite(vols)) or np.any(vols <= 0):
        return weights
    avg_vol = float(np.mean(vols))
    scaled: dict[uuid.UUID, Decimal] = {}
    for instrument_id, vol in zip(weights, vols, strict=True):
        multiplier = Decimal(str(avg_vol / float(vol)))
        scaled[instrument_id] = weights[instrument_id] * multiplier
    gross_before = sum(weights.values(), Decimal("0"))
    gross_after = sum(scaled.values(), Decimal("0"))
    if gross_after <= Decimal("0"):
        return weights
    return {
        instrument_id: weight * (gross_before / gross_after)
        for instrument_id, weight in scaled.items()
    }


def factor_exposures(
    weights: Mapping[uuid.UUID, Decimal],
    risk_model: PortfolioRiskModel,
) -> dict[str, Decimal]:
    exposures: dict[str, Decimal] = {}
    for instrument_id, weight in weights.items():
        for name, value in risk_model.factor_exposures.get(instrument_id, {}).items():
            exposures[name] = exposures.get(name, Decimal("0")) + weight * value
    return exposures


def apply_factor_caps(
    weights: dict[uuid.UUID, Decimal],
    risk_model: PortfolioRiskModel,
    cap: Decimal,
) -> dict[uuid.UUID, Decimal]:
    if cap <= Decimal("0") or not weights:
        return weights

    capped = dict(weights)
    for _ in range(4):
        exposures = factor_exposures(capped, risk_model)
        oversized = {
            name: value
            for name, value in exposures.items()
            if value > cap and has_positive_factor_loadings(name, capped, risk_model)
        }
        if not oversized:
            return capped
        for name, exposure in oversized.items():
            scale = cap / exposure
            for instrument_id, weight in list(capped.items()):
                if risk_model.factor_exposures.get(instrument_id, {}).get(name, Decimal("0")) > 0:
                    capped[instrument_id] = weight * scale
    return capped


def has_positive_factor_loadings(
    factor_name: str,
    weights: dict[uuid.UUID, Decimal],
    risk_model: PortfolioRiskModel,
) -> bool:
    return any(
        risk_model.factor_exposures.get(instrument_id, {}).get(factor_name, Decimal("0"))
        > Decimal("0")
        and weight > Decimal("0")
        for instrument_id, weight in weights.items()
    )


def stress_results(
    weights: Mapping[uuid.UUID, Decimal],
    risk_model: PortfolioRiskModel,
) -> dict[str, Decimal]:
    results: dict[str, Decimal] = {}
    for scenario in risk_model.scenarios:
        value = Decimal("0")
        for key, shock in scenario.shocks.items():
            if isinstance(key, uuid.UUID):
                value += weights.get(key, Decimal("0")) * shock
            elif key == "portfolio":
                value += sum(weights.values(), Decimal("0")) * shock
        results[scenario.name] = value
    return results


def stress_cvar(stress: dict[str, Decimal], tail_pct: float = 0.05) -> Decimal | None:
    losses = sorted((-value for value in stress.values() if value < Decimal("0")), reverse=True)
    if not losses:
        return None
    tail = losses[: max(1, math.ceil(len(losses) * tail_pct))]
    return sum(tail, Decimal("0")) / Decimal(len(tail))


def risk_snapshot(
    target: PortfolioTarget,
    weights: Mapping[uuid.UUID, Decimal],
    risk_model: PortfolioRiskModel,
    *,
    passed: bool,
) -> RiskSnapshot:
    stress = stress_results(weights, risk_model)
    return RiskSnapshot(
        snapshot_id=uuid.uuid4(),
        strategy_run_id=target.strategy_run_id,
        as_of=target.as_of,
        gross_exposure=sum(weights.values(), Decimal("0")),
        net_exposure=sum(weights.values(), Decimal("0")),
        cvar=stress_cvar(stress),
        factor_exposures=factor_exposures(weights, risk_model),
        stress_results=stress,
        passed=passed,
    )


__all__ = [
    "apply_factor_caps",
    "covariance_scale",
    "factor_exposures",
    "has_positive_factor_loadings",
    "risk_snapshot",
    "stress_cvar",
    "stress_results",
]
