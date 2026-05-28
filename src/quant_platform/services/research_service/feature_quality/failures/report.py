"""Markdown report rendering for feature failure attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast


def render_feature_failure_operator_report(payload: Mapping[str, object]) -> str:
    """Render a concise operator-facing diagnostic report."""
    features = _feature_rows(payload)
    top = sorted(features, key=_official_ic, reverse=True)[:8]
    families = _mapping_rows(payload.get("families", ()))
    validation = payload.get("data_validation", {})
    feature_set = payload.get("feature_set_version", "unknown")
    lines = [
        f"# {feature_set} Feature Failure Attribution",
        "",
        "## Summary",
        f"- Feature set: `{feature_set}`",
        f"- Diagnostic only: `{payload.get('diagnostic_only', False)}`",
        f"- Official horizon: `{payload.get('official_horizon_days', 'unknown')}d`",
        f"- Promotion artifacts written: `{payload.get('promotion_artifacts_written', True)}`",
        "",
        "## Data Validation",
    ]
    if isinstance(validation, Mapping):
        lines.extend(
            [
                f"- NYSE-session only: `{validation.get('nyse_session_only', False)}`",
                "- Available-at violations: "
                f"`{validation.get('available_at_violations', 'unknown')}`",
                f"- Sample counts by horizon: `{validation.get('sample_counts_by_horizon', {})}`",
                "- Nested data/parquet present: "
                f"`{validation.get('nested_data_parquet_present', False)}`",
            ]
        )
    lines.extend(["", "## Top Official-Horizon Signals"])
    for row in top:
        lines.append(
            "- "
            f"`{row.get('feature_name')}` "
            f"family=`{row.get('family')}` "
            f"ic={_official_ic(row):.4f} "
            f"null_margin={_health_metric(row, 'null_margin'):.4f} "
            f"nonzero={_health_metric(row, 'nonzero_fraction'):.3f} "
            f"label=`{row.get('diagnostic_recommendation')}`"
        )
    lines.extend(["", "## Feature Health"])
    for row in top:
        lines.append(
            "- "
            f"`{row.get('feature_name')}` "
            f"zero={_health_metric(row, 'zero_fraction'):.3f} "
            f"active_dates={_health_int(row, 'active_date_count')} "
            f"cost={_health_metric(row, 'cost_net_mean_return'):.6f} "
            f"incremental={_health_metric(row, 'incremental_delta_ic'):.4f}"
        )
        abstention = _health_mapping(row).get("abstention", {})
        if isinstance(abstention, Mapping) and abstention.get("source_feature"):
            lines.append(
                "  "
                f"source=`{abstention.get('source_feature')}` "
                f"active_vectors=`{abstention.get('active_vectors')}` "
                f"inactive_vectors=`{abstention.get('inactive_vectors')}` "
                f"condition_failures=`{abstention.get('condition_failure_counts', {})}`"
            )
    lines.extend(["", "## Family Representatives"])
    for row in families:
        lines.append(
            "- "
            f"`{row.get('family')}`: "
            f"`{row.get('representative_candidate')}` "
            f"from `{row.get('feature_count')}` features"
        )
    lines.extend(["", "## Dominant Failure Modes"])
    for gate, count in _failure_counts(features):
        lines.append(f"- `{gate}`: `{count}` features")
    lines.extend(["", "## Next Diagnostic Action"])
    lines.append(
        "Keep the diagnosed feature set frozen as failed evidence unless every governed "
        "admission and eligibility gate passes without threshold relaxation."
    )
    return "\n".join(lines) + "\n"


def _feature_rows(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    return _mapping_rows(payload.get("features", ()))


def _mapping_rows(raw: object) -> list[Mapping[str, object]]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return []
    return [cast("Mapping[str, object]", row) for row in raw if isinstance(row, Mapping)]


def _official_ic(row: Mapping[str, object]) -> float:
    horizons = row.get("horizon_comparison", {})
    if not isinstance(horizons, Mapping):
        return 0.0
    official = horizons.get("21", {})
    if not isinstance(official, Mapping):
        return 0.0
    return float(official.get("ic_mean", 0.0))


def _health_mapping(row: Mapping[str, object]) -> Mapping[str, object]:
    raw = row.get("diagnostic_health", {})
    return raw if isinstance(raw, Mapping) else {}


def _health_metric(row: Mapping[str, object], key: str) -> float:
    raw = _health_mapping(row).get(key, 0.0)
    if not isinstance(raw, int | float | str):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _health_int(row: Mapping[str, object], key: str) -> int:
    return int(_health_metric(row, key))


def _failure_counts(features: Sequence[Mapping[str, object]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in features:
        raw = row.get("failed_gates", ())
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            continue
        for gate in raw:
            counts[str(gate)] = counts.get(str(gate), 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


__all__ = ["render_feature_failure_operator_report"]
