"""Shared candidate-screening helpers for governed paper alpha research."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from quant_platform.services.research_service.feature_quality.failures.metadata import feature_seed
from quant_platform.services.research_service.feature_quality.failures.metrics import (
    daily_ic_rows,
    ic_summary,
    monthly_ic_rows,
    null_baseline,
    worst_negative_streak_window,
)

if TYPE_CHECKING:
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

MIN_SOURCE_EVENTS_PER_INSTRUMENT = 3
REQUIRED_INSTRUMENTS = 15
RANKED_DIAGNOSTIC_LIMIT = 20


@dataclass(frozen=True)
class CandidateScreenThresholds:
    """Gate thresholds for pre-backfill candidate screening."""

    min_source_density: float = 0.05
    min_null_margin: float = 0.0
    min_ic_mean: float = 0.02
    min_icir: float = 0.10
    max_negative_ic_streak: int = 3
    min_passing_candidates: int = 3


def candidate_screen_threshold_payload(thresholds: CandidateScreenThresholds) -> dict[str, object]:
    return dataclasses.asdict(thresholds)


def build_screen_candidate_row(
    *,
    samples: Sequence[SupervisedAlphaSample],
    feature_name: str,
    expression: str,
    thresholds: CandidateScreenThresholds,
    seed: int,
    permutation_count: int,
) -> dict[str, object]:
    """Build one candidate diagnostics row under the standard unchanged gates."""
    sign = 1.0
    summary = ic_summary(samples, feature_name, sign)
    daily_rows = daily_ic_rows(samples, feature_name, sign)
    null = null_baseline(
        samples,
        feature_name,
        sign,
        seed=feature_seed(seed, feature_name),
        count=permutation_count,
    )
    metrics = {
        "source_density": source_density(samples, feature_name),
        "ic_mean": float_value(summary.get("ic_mean")),
        "icir": float_value(summary.get("icir")),
        "negative_ic_streak": float_value(summary.get("ic_negative_streak")),
        "null_p95": float_value(null.get("null_p95")),
        "null_margin": float_value(null.get("actual_ic_mean")) - float_value(null.get("null_p95")),
        "ic_observations": float_value(summary.get("ic_observations")),
    }
    blockers = candidate_blockers(metrics, thresholds)
    return {
        "feature_name": feature_name,
        "expression": expression,
        "passed": not blockers,
        "blockers": blockers,
        "metrics": metrics,
        "daily_ic": {
            "observations": len(daily_rows),
            "worst_negative_streak": worst_negative_streak_window(daily_rows),
        },
        "monthly_ic": monthly_ic_rows(daily_rows),
        "null_baseline": null,
    }


def candidate_blockers(
    metrics: Mapping[str, float],
    thresholds: CandidateScreenThresholds,
) -> list[str]:
    blockers: list[str] = []
    if metrics["source_density"] < thresholds.min_source_density:
        blockers.append(
            f"source_density {metrics['source_density']:.4f} < {thresholds.min_source_density:.4f}"
        )
    if metrics["null_margin"] <= thresholds.min_null_margin:
        blockers.append(
            f"null_margin {metrics['null_margin']:.4f} <= {thresholds.min_null_margin:.4f}"
        )
    if metrics["ic_mean"] < thresholds.min_ic_mean:
        blockers.append(f"ic_mean {metrics['ic_mean']:.4f} < {thresholds.min_ic_mean:.4f}")
    if metrics["icir"] < thresholds.min_icir:
        blockers.append(f"icir {metrics['icir']:.4f} < {thresholds.min_icir:.4f}")
    if metrics["negative_ic_streak"] > thresholds.max_negative_ic_streak:
        blockers.append(
            f"negative_ic_streak {int(metrics['negative_ic_streak'])} "
            f"> {thresholds.max_negative_ic_streak}"
        )
    return blockers


def ranked_candidate_diagnostics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    summaries = [candidate_summary(row) for row in rows]
    top_by_stability = sorted(
        summaries,
        key=lambda row: (
            float_value(row.get("negative_ic_streak")),
            -float_value(row.get("null_margin")),
            -float_value(row.get("ic_mean")),
            str(row.get("feature_name", "")),
        ),
    )[:RANKED_DIAGNOSTIC_LIMIT]
    top_by_null_margin = sorted(
        summaries,
        key=lambda row: (
            -float_value(row.get("null_margin")),
            float_value(row.get("negative_ic_streak")),
            -float_value(row.get("ic_mean")),
            str(row.get("feature_name", "")),
        ),
    )[:RANKED_DIAGNOSTIC_LIMIT]
    near_misses = [
        row
        for row in sorted(
            summaries,
            key=lambda item: (
                len(cast("Sequence[object]", item.get("blockers", ()))),
                float_value(item.get("negative_ic_streak")),
                -float_value(item.get("null_margin")),
                str(item.get("feature_name", "")),
            ),
        )
        if not bool(row.get("passed"))
        and len(cast("Sequence[object]", row.get("blockers", ()))) <= 2
    ][:RANKED_DIAGNOSTIC_LIMIT]
    return {
        "top_by_stability": top_by_stability,
        "top_by_null_margin": top_by_null_margin,
        "near_misses": near_misses,
    }


def candidate_summary(row: Mapping[str, object]) -> dict[str, object]:
    metrics = cast("Mapping[str, object]", row.get("metrics", {}))
    blockers = row.get("blockers", ())
    blocker_list = (
        [str(blocker) for blocker in blockers]
        if isinstance(blockers, Sequence) and not isinstance(blockers, str)
        else []
    )
    return {
        "feature_name": str(row.get("feature_name", "")),
        "expression": str(row.get("expression", "")),
        "passed": bool(row.get("passed")),
        "blockers": blocker_list,
        "source_density": float_value(metrics.get("source_density")),
        "ic_mean": float_value(metrics.get("ic_mean")),
        "icir": float_value(metrics.get("icir")),
        "negative_ic_streak": float_value(metrics.get("negative_ic_streak")),
        "null_margin": float_value(metrics.get("null_margin")),
        "null_p95": float_value(metrics.get("null_p95")),
    }


def sec_source_manifest_summary(
    source_manifest: Mapping[str, object],
    *,
    count_field: str = "primary_events_by_symbol",
    source_label: str = "SEC primary",
) -> dict[str, object]:
    counts = _counts_by_symbol(source_manifest, count_field=count_field)
    covered = [symbol for symbol, count in counts.items() if count > 0]
    thin = [symbol for symbol, count in counts.items() if count < MIN_SOURCE_EVENTS_PER_INSTRUMENT]
    blockers: list[str] = []
    if len(covered) < REQUIRED_INSTRUMENTS:
        blockers.append(f"{source_label} coverage {len(covered)} < {REQUIRED_INSTRUMENTS}")
    if thin:
        blockers.append(
            f"{source_label} event count below "
            f"{MIN_SOURCE_EVENTS_PER_INSTRUMENT} for: {', '.join(thin)}"
        )
    return {
        "passed": not blockers,
        "blockers": blockers,
        count_field: counts,
        "covered_instruments": len(covered),
        "required_instruments": REQUIRED_INSTRUMENTS,
        "min_source_events_per_instrument": MIN_SOURCE_EVENTS_PER_INSTRUMENT,
    }


def sample_build_summary(
    sample_build: Mapping[str, object],
    *,
    sample_count: int,
) -> dict[str, object]:
    blockers: list[str] = []
    samples = int_value(sample_build.get("samples"))
    if samples is not None and samples != sample_count:
        blockers.append(f"sample count mismatch {samples} != loaded {sample_count}")
    for field in (
        "skipped_missing_features",
        "skipped_stale_features",
        "skipped_missing_bars",
        "skipped_invalid_features",
    ):
        value = int_value(sample_build.get(field)) or 0
        if value:
            blockers.append(f"{field}={value}")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "samples": samples,
        "loaded_samples": sample_count,
    }


def source_density(samples: Sequence[SupervisedAlphaSample], feature_name: str) -> float:
    if not samples:
        return 0.0
    nonzero = sum(1 for sample in samples if abs(finite_feature(sample.features, feature_name)) > 0)
    return nonzero / len(samples)


def finite_feature(features: Mapping[str, object], name: str) -> float:
    return float_value(features.get(name))


def float_value(raw: object, default: float = 0.0) -> float:
    try:
        value = float(cast("Any", raw))
    except (TypeError, ValueError, OverflowError):
        return default
    return value if math.isfinite(value) else default


def int_value(raw: object) -> int | None:
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _counts_by_symbol(
    source_manifest: Mapping[str, object],
    *,
    count_field: str,
) -> dict[str, int]:
    requested: list[str] = []
    download = source_manifest.get("download")
    if isinstance(download, Mapping):
        raw_requested = download.get("requested_symbols")
        if isinstance(raw_requested, list):
            requested = [str(symbol).upper() for symbol in raw_requested]
    counts = {symbol: 0 for symbol in requested}
    raw_counts = source_manifest.get(count_field)
    if isinstance(raw_counts, Mapping):
        for symbol, value in raw_counts.items():
            try:
                counts[str(symbol).upper()] = int(value)
            except (TypeError, ValueError):
                counts[str(symbol).upper()] = 0
    return dict(sorted(counts.items()))
