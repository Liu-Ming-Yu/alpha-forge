"""Formula context helpers for intraday candidate features."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import (
    ensure_utc,
    finite_feature,
)

if TYPE_CHECKING:
    from quant_platform.services.research_service.intraday.candidates.features.types import (
        IntradayFeatureContext,
    )


def context_value(ctx: IntradayFeatureContext, field: str) -> float:
    if ctx.metrics is None:
        return 0.0
    return float(getattr(ctx.metrics, field)) * ctx.decay


def session_direction(ctx: IntradayFeatureContext) -> float:
    if ctx.metrics is None:
        return 0.0
    return 1.0 if ctx.metrics.session_return >= 0.0 else -1.0


def sample_context_value(ctx: IntradayFeatureContext, field: str) -> float:
    if ctx.sample_features is None:
        return 0.0
    return finite_feature(ctx.sample_features, field)


def aggregate_context_value(
    ctx: IntradayFeatureContext,
    field: str,
    lookback_days: int,
    *,
    signed: bool = False,
) -> float:
    if ctx.as_of is None or lookback_days <= 0:
        return context_value(ctx, field)
    total = 0.0
    weight_sum = 0.0
    as_of = ensure_utc(ctx.as_of)
    for metrics in ctx.history:
        age_days = (as_of - metrics.end_at).total_seconds() / 86400.0
        if age_days < 0.0 or age_days > float(lookback_days):
            continue
        decay = max(0.0, 1.0 - (age_days / float(lookback_days)))
        if decay <= 0.0:
            continue
        value = float(getattr(metrics, field))
        if signed:
            value *= 1.0 if metrics.session_return >= 0.0 else -1.0
        total += value * decay
        weight_sum += decay
    return total / weight_sum if weight_sum else 0.0


def aggregate_context_band(
    ctx: IntradayFeatureContext,
    field: str,
    start_days: int,
    end_days: int,
    *,
    signed: bool = False,
) -> float:
    if end_days <= start_days:
        return 0.0
    return aggregate_context_value(
        ctx,
        field,
        end_days,
        signed=signed,
    ) - aggregate_context_value(
        ctx,
        field,
        start_days,
        signed=signed,
    )
