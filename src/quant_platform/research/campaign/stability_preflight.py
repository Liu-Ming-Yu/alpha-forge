"""Stability-attribution preflight implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.campaign.blocked import write_blocked_campaign_summary
from quant_platform.research.common import (
    _build_samples_to_path,
    _json_default,
    _samples_result_payload,
)
from quant_platform.services.research_service.feature_quality.diagnostics import (
    null_qualified_features,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@dataclass(frozen=True)
class StabilityAttributionPreflight:
    """Result of the diagnostic-only attribution preflight."""

    passed: bool
    direction_path: Path
    attribution_path: Path
    report_path: Path
    qualified_features: tuple[str, ...]
    quarantined_features: tuple[str, ...]
    blocked_payload: dict[str, object] | None = None


async def run_stability_attribution_preflight(
    *,
    settings: PlatformSettings,
    samples: Sequence[SupervisedAlphaSample],
    sample_build: Mapping[str, object],
    output_root: Path,
    sample_slug: str,
    contracts_file: str,
    start: datetime,
    end: datetime,
    feature_set_version: str,
    official_horizon_days: int,
    horizons: Sequence[int],
    bar_seconds: int,
    max_feature_age_days: int,
    date_policy: str,
    feature_card_dir: Path,
    feature_family_file: Path,
    slippage_bps_per_turnover: float,
    permutation_seed: int,
    permutation_count: int,
    correlation_threshold: float,
    min_null_qualified_features: int,
    candidate_feature_names: Sequence[str] | None = None,
) -> StabilityAttributionPreflight:
    """Run diagnostic attribution and enforce the OHLCV null-baseline preflight."""
    from quant_platform.services.research_service.feature_quality.diagnostics.direction import (
        build_feature_direction_diagnostics,
    )
    from quant_platform.services.research_service.feature_quality.failures.attribution import (
        build_feature_failure_attribution,
    )
    from quant_platform.services.research_service.feature_quality.failures.report import (
        render_feature_failure_operator_report,
    )
    from quant_platform.services.research_service.sampling.factory import load_supervised_samples

    normalized = _normalized_horizons(horizons, official_horizon_days)
    samples_by_horizon: dict[int, tuple[SupervisedAlphaSample, ...]] = {
        official_horizon_days: tuple(samples),
    }
    sample_builds: dict[int, dict[str, object]] = {
        official_horizon_days: dict(sample_build),
    }
    diagnostics_root = output_root / "diagnostics" / f"{sample_slug}_stability_attribution"
    for horizon in normalized:
        if horizon == official_horizon_days:
            continue
        sample_path = output_root / "_diagnostics" / sample_slug / f"samples_{horizon}d.json"
        built_path, result = await _build_samples_to_path(
            settings=settings,
            contracts_file=contracts_file,
            start=start,
            end=end,
            output=sample_path,
            feature_set_version=feature_set_version,
            horizon_days=horizon,
            bar_seconds=bar_seconds,
            max_feature_age_days=max_feature_age_days,
            date_policy=date_policy,
        )
        samples_by_horizon[horizon] = tuple(load_supervised_samples(built_path))
        sample_builds[horizon] = _samples_result_payload(built_path, result)

    direction = build_feature_direction_diagnostics(
        samples=samples,
        feature_set_version=feature_set_version,
        feature_card_dir=feature_card_dir,
        slippage_bps_per_turnover=slippage_bps_per_turnover,
        candidate_feature_names=candidate_feature_names,
    )
    direction.update(
        {
            "date_policy": date_policy,
            "sample_build": dict(sample_build),
            "slippage_bps_per_turnover": slippage_bps_per_turnover,
        }
    )
    family_metadata = _load_json_mapping(feature_family_file)
    attribution = build_feature_failure_attribution(
        samples_by_horizon=samples_by_horizon,
        sample_builds_by_horizon=sample_builds,
        feature_set_version=feature_set_version,
        official_horizon_days=official_horizon_days,
        direction_diagnostics=direction,
        family_metadata=family_metadata,
        date_policy=date_policy,
        nested_object_store_present=(
            Path(settings.storage.object_store_root) / "data" / "parquet"
        ).exists(),
        seed=permutation_seed,
        permutation_count=permutation_count,
        correlation_threshold=correlation_threshold,
        candidate_feature_names=candidate_feature_names,
    )
    qualification = null_qualified_features(
        attribution,
        official_horizon_days=official_horizon_days,
    )
    direction_path, attribution_path, report_path = _write_attribution_artifacts(
        diagnostics_root=diagnostics_root,
        direction=direction,
        attribution=attribution,
        report=render_feature_failure_operator_report(attribution),
    )
    if len(qualification["qualified"]) >= min_null_qualified_features:
        return StabilityAttributionPreflight(
            passed=True,
            direction_path=direction_path,
            attribution_path=attribution_path,
            report_path=report_path,
            qualified_features=qualification["qualified"],
            quarantined_features=qualification["quarantined"],
        )
    reason = (
        f"{feature_set_version} null-baseline attribution blocked paper research campaign: "
        f"{len(qualification['qualified'])} features beat null p95 at "
        f"{official_horizon_days}d, required {min_null_qualified_features}"
    )
    admission = _blocked_admission(
        qualification=qualification,
        min_null_qualified_features=min_null_qualified_features,
        reason=reason,
    )
    summary_path = write_blocked_campaign_summary(
        output_root=output_root,
        sample_slug=sample_slug,
        reason=reason,
        sample_build=sample_build,
        feature_audits=(),
        feature_admission=admission,
        date_policy=date_policy,
        feature_diagnostics_path=direction_path,
        feature_attribution_path=attribution_path,
    )
    return StabilityAttributionPreflight(
        passed=False,
        direction_path=direction_path,
        attribution_path=attribution_path,
        report_path=report_path,
        qualified_features=qualification["qualified"],
        quarantined_features=qualification["quarantined"],
        blocked_payload={
            "passed": False,
            "reason": reason,
            "blocked_campaign_summary": str(summary_path),
            "feature_direction_diagnostics": str(direction_path),
            "feature_failure_attribution": str(attribution_path),
            "operator_report": str(report_path),
            "date_policy": date_policy,
            "sample_build": dict(sample_build),
            "feature_admission": admission,
        },
    )


def _write_attribution_artifacts(
    *,
    diagnostics_root: Path,
    direction: dict[str, object],
    attribution: dict[str, object],
    report: str,
) -> tuple[Path, Path, Path]:
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    direction_path = diagnostics_root / "feature_direction_diagnostics.json"
    attribution_path = diagnostics_root / "feature_failure_attribution.json"
    report_path = diagnostics_root / "operator_report.md"
    direction_path.write_text(
        json.dumps(direction, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    attribution_path.write_text(
        json.dumps(attribution, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")
    return direction_path, attribution_path, report_path


def _blocked_admission(
    *,
    qualification: dict[str, tuple[str, ...]],
    min_null_qualified_features: int,
    reason: str,
) -> dict[str, object]:
    return {
        "mode": "null-baseline-preflight",
        "audit_mode": "paper",
        "passed": False,
        "min_admitted_features": min_null_qualified_features,
        "audited_features": list(qualification["audited"]),
        "admitted_features": list(qualification["qualified"]),
        "quarantined_features": list(qualification["quarantined"]),
        "blockers": [reason],
    }


def _normalized_horizons(raw: Sequence[int], official_horizon: int) -> tuple[int, ...]:
    horizons = {int(value) for value in raw}
    horizons.add(official_horizon)
    if any(horizon <= 0 for horizon in horizons):
        raise OperatorUsageError("--attribution-horizons and --horizon-days must be positive")
    return tuple(sorted(horizons))


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorUsageError(f"failed to load feature family metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OperatorUsageError(f"feature family metadata must be a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


__all__ = ["StabilityAttributionPreflight", "run_stability_attribution_preflight"]
