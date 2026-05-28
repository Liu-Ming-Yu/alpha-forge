"""Simulator vs paper-fill calibration.

Compares the slippage that the participation fill model is *configured* to
charge in research and backtests against the slippage that paper trading
*actually* realised, broken out by EMS tactic, ADV bucket, spread bucket,
stale-price age, order type, and time of day.

The output is a JSON artifact that the research campaign command reads
back so ``calibrated_slippage_bps_per_turnover`` can pick the higher of:

- the static configured floor;
- the recently observed mean slippage in basis points; and
- a tactic-aware adjustment learned from this report.

Producing the artifact is the job of
``quant-platform simulator-calibration report`` which queries
``fill_events JOIN order_intents JOIN instruments`` and writes a
calibration JSON under
``$QP__STORAGE__OBJECT_STORE_ROOT/calibration/``.
"""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.services.governance_service.simulator_calibration.artifacts import (
    calibration_payload,
    write_calibration_report,
)
from quant_platform.services.governance_service.simulator_calibration.models import (
    CalibrationBucket,
    CalibrationReport,
    FillObservation,
)
from quant_platform.services.governance_service.simulator_calibration.postgres import (
    load_paper_fills_from_postgres,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_ADV_BUCKETS = (
    (0.0, 0.005, "<0.5pct_adv"),
    (0.005, 0.02, "0.5-2pct_adv"),
    (0.02, 0.05, "2-5pct_adv"),
    (0.05, 1.0, ">=5pct_adv"),
)

__all__ = [
    "CalibrationBucket",
    "CalibrationReport",
    "FillObservation",
    "calibration_payload",
    "compute_calibration_report",
    "load_paper_fills_from_postgres",
    "write_calibration_report",
]


def _adv_bucket(participation_pct: float) -> str:
    for lower, upper, label in _DEFAULT_ADV_BUCKETS:
        if lower <= participation_pct < upper:
            return label
    return _DEFAULT_ADV_BUCKETS[-1][2]


def _spread_bucket(spread_bps: float) -> str:
    if spread_bps < 5:
        return "tight_spread"
    if spread_bps < 15:
        return "normal_spread"
    return "wide_spread"


def _stale_price_bucket(age_seconds: float) -> str:
    if age_seconds <= 5:
        return "fresh"
    if age_seconds <= 60:
        return "recent"
    return "stale"


def _time_bucket(executed_at: datetime) -> str:
    hour = executed_at.astimezone(UTC).hour
    minute = executed_at.astimezone(UTC).minute
    minutes = hour * 60 + minute
    if 14 * 60 + 30 <= minutes < 15 * 60:
        return "open"
    if 20 * 60 <= minutes <= 21 * 60:
        return "close"
    return "midday"


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    fraction = k - lo
    return sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction


def compute_calibration_report(
    observations: Sequence[FillObservation],
    *,
    as_of: datetime,
    floor_bps: float = 1.0,
    min_sample_count: int = 20,
    p90_safety_margin: float = 0.0,
) -> CalibrationReport:
    """Compute the calibration recommendations for the given fills.

    Args:
        observations: Paper fills to summarise.  Negative ``slippage_bps``
            values are clamped to zero so research is never penalised by
            a fictitious price improvement.
        as_of: Timestamp recorded with the report.
        floor_bps: Lower bound on every recommendation.
        min_sample_count: Minimum overall sample count below which
            ``insufficient_data`` is set; gates can decide to fall back to
            the configured default.
        p90_safety_margin: Multiplicative bump applied on top of the p90
            (e.g. ``0.10`` adds 10% margin to the recommendation).

    Returns:
        :class:`CalibrationReport` with bucketed recommendations.  Pure
        function; never touches the database.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=UTC)

    cleaned: list[FillObservation] = []
    for entry in observations:
        bps = max(0.0, float(entry.slippage_bps))
        cleaned.append(
            FillObservation(
                tactic=entry.tactic.strip() or "unknown",
                side=entry.side,
                quantity=entry.quantity,
                adv_shares_20d=max(0.0, float(entry.adv_shares_20d)),
                spread_bps=float(entry.spread_bps),
                slippage_bps=bps,
                executed_at=entry.executed_at,
                stale_price_age_seconds=max(0.0, float(entry.stale_price_age_seconds)),
                order_type=entry.order_type.strip().lower() or "unknown",
            )
        )

    overall_values = [row.slippage_bps for row in cleaned]
    overall_median = float(statistics.median(overall_values)) if overall_values else 0.0
    overall_p90 = _percentile(overall_values, 0.90)
    overall_recommended = max(
        floor_bps,
        overall_median,
        overall_p90 * (1.0 + max(0.0, p90_safety_margin)),
    )

    grouped: dict[tuple[str, str, str, str, str, str], list[FillObservation]] = {}
    for row in cleaned:
        participation = row.quantity / row.adv_shares_20d if row.adv_shares_20d > 0 else 0.0
        key = (
            row.tactic,
            _adv_bucket(participation),
            _spread_bucket(row.spread_bps),
            _stale_price_bucket(row.stale_price_age_seconds),
            row.order_type,
            _time_bucket(row.executed_at),
        )
        grouped.setdefault(key, []).append(row)

    bucket_reports: list[CalibrationBucket] = []
    for (tactic, bucket, spread_bucket, stale_bucket, order_type, time_bucket), rows in sorted(
        grouped.items()
    ):
        slips = [row.slippage_bps for row in rows]
        median = float(statistics.median(slips))
        p90 = _percentile(slips, 0.90)
        recommended = max(floor_bps, median, p90 * (1.0 + max(0.0, p90_safety_margin)))
        bucket_reports.append(
            CalibrationBucket(
                tactic=tactic,
                adv_bucket=bucket,
                sample_count=len(rows),
                median_slippage_bps=median,
                p90_slippage_bps=p90,
                recommended_bps=recommended,
                spread_bucket=spread_bucket,
                stale_price_bucket=stale_bucket,
                order_type=order_type,
                time_bucket=time_bucket,
            )
        )

    return CalibrationReport(
        generated_at=as_of.astimezone(UTC),
        sample_count=len(cleaned),
        overall_median_bps=overall_median,
        overall_p90_bps=overall_p90,
        overall_recommended_bps=overall_recommended,
        buckets=tuple(bucket_reports),
        insufficient_data=len(cleaned) < min_sample_count,
        floor_bps=floor_bps,
    )
