"""Point-in-time text aggregate feature construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.screening.common import (
    ensure_utc,
    finite_feature,
)
from quant_platform.services.research_service.text.candidates.catalog.fields import (
    TEXT_AGGREGATE_LOOKBACK_DAYS,
    TEXT_AGGREGATE_SUM_CLIP,
    TEXT_AGGREGATE_WINDOWS,
)
from quant_platform.services.research_service.text.candidates.catalog.math_utils import (
    _clip,
    _is_finite_number,
    _sign,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from quant_platform.core.domain.research import FeatureVector


def build_text_aggregate_features(
    vectors: Sequence[FeatureVector],
    as_of: datetime,
    *,
    lookback_days: int = TEXT_AGGREGATE_LOOKBACK_DAYS,
) -> tuple[dict[str, float], float | None]:
    """Build latest-compatible and multi-event text features for candidate formulas."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    as_of_utc = ensure_utc(as_of)
    max_window = max(lookback_days, *TEXT_AGGREGATE_WINDOWS)
    visible = _visible_text_vectors(vectors, as_of_utc, lookback_days=max_window)
    active = _window_rows(visible, lookback_days)
    if not active:
        return ({}, None)

    latest, latest_decay, latest_age = max(
        active,
        key=lambda item: (
            ensure_utc(item[0].as_of),
            ensure_utc(item[0].available_at or item[0].as_of),
        ),
    )
    aggregate: dict[str, float] = {
        str(name): float(value)
        for name, value in latest.features.items()
        if isinstance(name, str) and _is_finite_number(value)
    }
    aggregate["text_event_count_21d"] = float(len(active))
    aggregate["days_since_text_event"] = float(latest_age)
    aggregate["latest_text_decay_21d"] = latest_decay

    fields = (
        "text_sentiment",
        "catalyst_sentiment",
        "earnings_quality",
        "forward_outlook",
        "operating_quality",
        "margin_resilience",
        "risk_pressure",
        "disclosure_specificity",
        "macro_sentiment",
        "event_surprise",
        "demand_outlook",
        "guidance_specificity",
        "revision_clarity",
    )
    for window in TEXT_AGGREGATE_WINDOWS:
        _add_window_aggregates(
            aggregate,
            rows=_window_rows(visible, window),
            fields=fields,
            window=window,
        )
    return aggregate, latest_decay


def _visible_text_vectors(
    vectors: Sequence[FeatureVector],
    as_of: datetime,
    *,
    lookback_days: int,
) -> tuple[tuple[FeatureVector, float], ...]:
    visible: list[tuple[FeatureVector, float]] = []
    for vector in sorted(vectors, key=lambda row: ensure_utc(row.as_of)):
        vector_as_of = ensure_utc(vector.as_of)
        available_at = ensure_utc(vector.available_at or vector.as_of)
        if vector_as_of > as_of or available_at > as_of:
            continue
        age_days = (as_of - vector_as_of).total_seconds() / 86400.0
        if 0.0 <= age_days <= float(lookback_days):
            visible.append((vector, age_days))
    return tuple(visible)


def _window_rows(
    visible: Sequence[tuple[FeatureVector, float]],
    window: int,
) -> tuple[tuple[FeatureVector, float, float], ...]:
    rows: list[tuple[FeatureVector, float, float]] = []
    for vector, age_days in visible:
        if age_days > float(window):
            continue
        decay = max(0.0, 1.0 - (age_days / float(window)))
        if decay > 0.0:
            rows.append((vector, decay, age_days))
    return tuple(rows)


def _add_window_aggregates(
    aggregate: dict[str, float],
    *,
    rows: Sequence[tuple[FeatureVector, float, float]],
    fields: Sequence[str],
    window: int,
) -> None:
    suffix = f"{window}d"
    aggregate[f"text_event_count_{suffix}"] = float(len(rows))
    if rows:
        latest_age = min(age_days for _vector, _decay, age_days in rows)
        aggregate[f"days_since_text_event_{suffix}"] = float(latest_age)
        aggregate[f"latest_text_decay_{suffix}"] = max(
            decay for _vector, decay, age_days in rows if age_days == latest_age
        )
    else:
        aggregate[f"days_since_text_event_{suffix}"] = 0.0
        aggregate[f"latest_text_decay_{suffix}"] = 0.0
    for field in fields:
        weighted_sum = 0.0
        decay_sum = 0.0
        signed_max = 0.0
        signed_direction_sum = 0.0
        values: list[tuple[float, float]] = []
        for vector, decay, _age_days in rows:
            value = finite_feature(vector.features, field)
            decayed = value * decay
            weighted_sum += decayed
            decay_sum += decay
            signed_direction_sum += _sign(value) * decay
            values.append((value, decay))
            if abs(decayed) > abs(signed_max):
                signed_max = decayed
        mean = weighted_sum / decay_sum if decay_sum else 0.0
        dispersion = (
            sum(abs(value - mean) * decay for value, decay in values) / decay_sum
            if decay_sum
            else 0.0
        )
        aggregate[f"{field}_decayed_mean_{suffix}"] = mean
        aggregate[f"{field}_decayed_sum_clipped_{suffix}"] = _clip(
            weighted_sum,
            -TEXT_AGGREGATE_SUM_CLIP,
            TEXT_AGGREGATE_SUM_CLIP,
        )
        aggregate[f"{field}_decayed_max_abs_{suffix}"] = signed_max
        aggregate[f"{field}_sign_consistency_{suffix}"] = (
            abs(signed_direction_sum) / decay_sum if decay_sum else 0.0
        )
        aggregate[f"{field}_event_dispersion_{suffix}"] = dispersion
