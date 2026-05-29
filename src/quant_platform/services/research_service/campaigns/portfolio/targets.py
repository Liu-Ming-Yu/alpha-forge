"""Target-weight and turnover-limit helpers for campaign portfolios."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.portfolio.selection import TopNSelection
from quant_platform.services.research_service.campaigns.portfolio.weighting import EqualWeight

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.campaigns.portfolio.selection import (
        SelectionStrategy,
    )
    from quant_platform.services.research_service.campaigns.portfolio.types import (
        CampaignPortfolioConfig,
    )
    from quant_platform.services.research_service.campaigns.portfolio.weighting import (
        WeightingScheme,
    )
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

_EPSILON = 1e-12


def raw_long_only_target(
    rows: Sequence[tuple[SupervisedAlphaSample, float]],
    *,
    config: CampaignPortfolioConfig,
    weighting: WeightingScheme | None = None,
    selection: SelectionStrategy | None = None,
    current_holdings: frozenset[uuid.UUID] | None = None,
) -> dict[uuid.UUID, float]:
    """Selected names, sized by ``weighting`` and scaled to the investable-gross
    budget.

    ``selection=None`` uses :class:`TopNSelection` (the fresh top-N by score) and
    ``weighting=None`` uses :class:`EqualWeight`, together reproducing the prior
    arithmetic bit-for-bit: equal weight's per-name budget was
    ``min(max_single_name_weight, investable / N)``, i.e. a total gross of
    ``min(N · cap, investable)``. That same gross is computed here and split by
    the weighting scheme's proportions, so every scheme holds the same gross and
    only the *distribution* across names differs. Per-name and gross caps are
    enforced downstream by ``enforce_weight_limits``, identically for every
    scheme.

    ``selection`` chooses *which* names to hold from the positive-score
    candidates; ``current_holdings`` (the prior rebalance's book) lets a buffered
    strategy (:class:`BufferedTopKSelection`, Arm M) keep slipping incumbents
    rather than churn them. The default ignores ``current_holdings``.
    """
    selector: SelectionStrategy = selection if selection is not None else TopNSelection()
    scheme: WeightingScheme = weighting if weighting is not None else EqualWeight()
    ranked = sorted(
        ((row, score) for row, score in rows if score > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = selector.select(
        ranked,
        top_n=int(config.top_n),
        current_holdings=current_holdings if current_holdings is not None else frozenset(),
    )
    if not selected:
        return {}
    investable = min(float(config.max_gross_exposure), 1.0 - float(config.min_cash_buffer))
    target_gross = min(investable, len(selected) * float(config.max_single_name_weight))
    proportions = scheme.proportions(selected)
    return {
        row.instrument_id: target_gross * proportion
        for (row, _), proportion in zip(selected, proportions, strict=True)
    }


def enforce_weight_limits(
    weights: Mapping[uuid.UUID, float],
    *,
    config: CampaignPortfolioConfig,
) -> dict[uuid.UUID, float]:
    capped = {
        instrument_id: min(max(0.0, float(weight)), float(config.max_single_name_weight))
        for instrument_id, weight in weights.items()
        if float(weight) > _EPSILON
    }
    gross = sum(capped.values())
    max_invest = min(float(config.max_gross_exposure), 1.0 - float(config.min_cash_buffer))
    if gross <= max_invest + _EPSILON or gross <= _EPSILON:
        return clean_weights(capped)
    scale = max_invest / gross
    return clean_weights(
        {instrument_id: weight * scale for instrument_id, weight in capped.items()}
    )


def apply_no_trade_band(
    *,
    current: Mapping[uuid.UUID, float],
    target: Mapping[uuid.UUID, float],
    band: float,
) -> dict[uuid.UUID, float]:
    """Hold the current weight when the desired change is below ``band``.

    Cost-aware hysteresis applied before the position-change and turnover caps:
    small score wiggles no longer generate trades, which is the dominant driver
    of the turnover that erodes slippage-adjusted Sharpe.  ``band <= 0`` is a
    no-op that returns the target unchanged.
    """
    if band <= 0.0:
        return clean_weights(dict(target))
    adjusted: dict[uuid.UUID, float] = {}
    for instrument_id in set(current) | set(target):
        current_weight = float(current.get(instrument_id, 0.0))
        target_weight = float(target.get(instrument_id, 0.0))
        if abs(target_weight - current_weight) < band:
            adjusted[instrument_id] = current_weight
        else:
            adjusted[instrument_id] = target_weight
    return clean_weights(adjusted)


def apply_position_change_cap(
    *,
    current: Mapping[uuid.UUID, float],
    target: Mapping[uuid.UUID, float],
    max_position_change: float,
) -> dict[uuid.UUID, float]:
    adjusted: dict[uuid.UUID, float] = {}
    for instrument_id in set(current) | set(target):
        current_weight = float(current.get(instrument_id, 0.0))
        target_weight = float(target.get(instrument_id, 0.0))
        delta = target_weight - current_weight
        if abs(delta) > max_position_change:
            delta = math.copysign(max_position_change, delta)
        adjusted[instrument_id] = current_weight + delta
    return clean_weights(adjusted)


def apply_turnover_cap(
    *,
    current: Mapping[uuid.UUID, float],
    target: Mapping[uuid.UUID, float],
    max_daily_turnover: float,
) -> tuple[dict[uuid.UUID, float], float]:
    keys = set(current) | set(target)
    deltas = {
        instrument_id: float(target.get(instrument_id, 0.0))
        - float(current.get(instrument_id, 0.0))
        for instrument_id in keys
    }
    desired_turnover = sum(abs(delta) for delta in deltas.values())
    if desired_turnover <= max_daily_turnover + _EPSILON or desired_turnover <= _EPSILON:
        return clean_weights(dict(target)), desired_turnover
    scale = max_daily_turnover / desired_turnover
    weights = {
        instrument_id: float(current.get(instrument_id, 0.0)) + delta * scale
        for instrument_id, delta in deltas.items()
    }
    return clean_weights(weights), max_daily_turnover


def clean_weights(weights: Mapping[uuid.UUID, float]) -> dict[uuid.UUID, float]:
    return {
        instrument_id: max(0.0, float(weight))
        for instrument_id, weight in weights.items()
        if math.isfinite(float(weight)) and abs(float(weight)) > _EPSILON
    }


__all__ = [
    "apply_no_trade_band",
    "apply_position_change_cap",
    "apply_turnover_cap",
    "clean_weights",
    "enforce_weight_limits",
    "raw_long_only_target",
]
