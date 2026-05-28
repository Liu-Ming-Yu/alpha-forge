"""Research-campaign feature direction diagnostics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.campaign.inputs import _observed_slippage_bps
from quant_platform.research.common import (
    _build_samples_to_path,
    _json_default,
    _latest_calibration_artifact,
    _load_calibration_recommendation_bps,
    _require_durable_research_inputs,
    _samples_result_payload,
    _verify_postgres_schema_if_configured,
    research_json_result,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


async def _research_campaign_diagnose_features(
    settings: PlatformSettings,
    args: Any,
) -> UseCaseResult[dict[str, object]]:
    """Build samples and write feature-direction diagnostics without training models."""
    from quant_platform.services.research_service.feature_quality.diagnostics.direction import (
        build_feature_direction_diagnostics,
    )
    from quant_platform.services.research_service.sampling.factory import (
        calibrated_slippage_bps_per_turnover,
        load_supervised_samples,
        walk_forward_object_root,
    )

    _require_durable_research_inputs(settings)
    await _verify_postgres_schema_if_configured(settings)
    if args.feature_card_dir is None:
        raise OperatorUsageError("diagnose-features requires --feature-card-dir")

    output_root: Path = args.output_root or walk_forward_object_root(
        settings.storage.object_store_root,
    )
    slug = f"{args.feature_set_version}_{args.start:%Y-%m-%d}_{args.end:%Y-%m-%d}"
    sample_path = output_root / "_diagnostics" / slug / "samples.json"
    sample_path, sample_result = await _build_samples_to_path(
        settings=settings,
        contracts_file=args.contracts_file,
        start=args.start,
        end=args.end,
        output=sample_path,
        feature_set_version=args.feature_set_version,
        horizon_days=args.horizon_days,
        bar_seconds=args.bar_seconds,
        max_feature_age_days=args.max_feature_age_days,
        date_policy=str(args.date_policy),
    )
    observed_slippage = await _observed_slippage_bps(settings)
    calibration_path = _latest_calibration_artifact(settings)
    calibration_bps, calibration_meta = _load_calibration_recommendation_bps(
        calibration_path,
        max_age_days=float(getattr(args, "max_calibration_age_days", 14.0) or 14.0),
        as_of=datetime.now(tz=UTC),
    )
    if bool(getattr(args, "require_calibration", False)) and calibration_bps is None:
        raise OperatorUsageError("fresh simulator calibration artifact is required")
    slippage = calibrated_slippage_bps_per_turnover(
        observed_slippage,
        default_bps=float(args.slippage_bps_per_turnover),
        calibration_recommendation_bps=calibration_bps,
    )
    diagnostics = build_feature_direction_diagnostics(
        samples=load_supervised_samples(sample_path),
        feature_set_version=args.feature_set_version,
        feature_card_dir=args.feature_card_dir,
        slippage_bps_per_turnover=slippage,
    )
    diagnostics.update(
        {
            "date_policy": str(args.date_policy),
            "sample_build": _samples_result_payload(sample_path, sample_result),
            "slippage_bps_per_turnover": slippage,
            "calibration_artifact": calibration_meta,
        }
    )
    path = output_root / "diagnostics" / slug / "feature_direction_diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diagnostics, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "passed": not diagnostics["missing_cards"],
        "diagnostics": str(path),
        "feature_count": diagnostics["feature_count"],
        "missing_cards": diagnostics["missing_cards"],
        "sample_build": diagnostics["sample_build"],
    }
    return research_json_result(payload, passed=not diagnostics["missing_cards"])


async def _research_campaign_attribute_feature_failures(
    settings: PlatformSettings,
    args: Any,
) -> UseCaseResult[dict[str, object]]:
    """Write feature failure attribution diagnostics without training models."""
    from pathlib import Path

    from quant_platform.services.research_service.feature_quality.diagnostics.direction import (
        build_feature_direction_diagnostics,
    )
    from quant_platform.services.research_service.feature_quality.failures.attribution import (
        build_feature_failure_attribution,
    )
    from quant_platform.services.research_service.feature_quality.failures.report import (
        render_feature_failure_operator_report,
    )
    from quant_platform.services.research_service.sampling.factory import (
        calibrated_slippage_bps_per_turnover,
        load_supervised_samples,
        walk_forward_object_root,
    )

    _require_durable_research_inputs(settings)
    await _verify_postgres_schema_if_configured(settings)
    output_root: Path = args.output_root or walk_forward_object_root(
        settings.storage.object_store_root,
    )
    horizons = _normalized_horizons(args.horizons, int(args.official_horizon_days))
    slug = (
        f"{args.feature_set_version}_{args.start:%Y-%m-%d}_{args.end:%Y-%m-%d}_failure_attribution"
    )
    samples_by_horizon: dict[int, tuple[SupervisedAlphaSample, ...]] = {}
    sample_builds: dict[int, dict[str, object]] = {}
    for horizon in horizons:
        path = output_root / "_diagnostics" / slug / f"samples_{horizon}d.json"
        sample_path, result = await _build_samples_to_path(
            settings=settings,
            contracts_file=args.contracts_file,
            start=args.start,
            end=args.end,
            output=path,
            feature_set_version=args.feature_set_version,
            horizon_days=horizon,
            bar_seconds=args.bar_seconds,
            max_feature_age_days=args.max_feature_age_days,
            date_policy=str(args.date_policy),
        )
        samples_by_horizon[horizon] = tuple(load_supervised_samples(sample_path))
        sample_builds[horizon] = _samples_result_payload(sample_path, result)

    observed_slippage = await _observed_slippage_bps(settings)
    calibration_bps, calibration_meta = _load_calibration_recommendation_bps(
        _latest_calibration_artifact(settings),
        max_age_days=float(getattr(args, "max_calibration_age_days", 14.0) or 14.0),
        as_of=datetime.now(tz=UTC),
    )
    if bool(getattr(args, "require_calibration", False)) and calibration_bps is None:
        raise OperatorUsageError("fresh simulator calibration artifact is required")
    slippage = calibrated_slippage_bps_per_turnover(
        observed_slippage,
        default_bps=float(args.slippage_bps_per_turnover),
        calibration_recommendation_bps=calibration_bps,
    )
    official_samples = samples_by_horizon[int(args.official_horizon_days)]
    direction = build_feature_direction_diagnostics(
        samples=official_samples,
        feature_set_version=args.feature_set_version,
        feature_card_dir=args.feature_card_dir,
        slippage_bps_per_turnover=slippage,
    )
    direction.update(
        {
            "date_policy": str(args.date_policy),
            "sample_build": sample_builds[int(args.official_horizon_days)],
            "slippage_bps_per_turnover": slippage,
            "calibration_artifact": calibration_meta,
        }
    )
    family_metadata = _load_json_mapping(args.feature_family_file)
    diagnostics_root = output_root / "diagnostics" / slug
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    direction_path = diagnostics_root / "feature_direction_diagnostics.json"
    direction_path.write_text(
        json.dumps(direction, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    attribution = build_feature_failure_attribution(
        samples_by_horizon=samples_by_horizon,
        sample_builds_by_horizon=sample_builds,
        feature_set_version=args.feature_set_version,
        official_horizon_days=int(args.official_horizon_days),
        direction_diagnostics=direction,
        family_metadata=family_metadata,
        date_policy=str(args.date_policy),
        nested_object_store_present=(
            Path(settings.storage.object_store_root) / "data" / "parquet"
        ).exists(),
        seed=int(args.permutation_seed),
        permutation_count=int(args.permutation_count),
        correlation_threshold=float(args.correlation_threshold),
    )
    attribution_path = diagnostics_root / "feature_failure_attribution.json"
    attribution_path.write_text(
        json.dumps(attribution, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = diagnostics_root / "operator_report.md"
    report_path.write_text(render_feature_failure_operator_report(attribution), encoding="utf-8")
    direction_summary = attribution.get("direction_diagnostics", {})
    feature_count = (
        direction_summary.get("feature_count", 0) if isinstance(direction_summary, dict) else 0
    )
    payload = {
        "passed": True,
        "feature_failure_attribution": str(attribution_path),
        "operator_report": str(report_path),
        "feature_direction_diagnostics": str(direction_path),
        "diagnostic_only": True,
        "promotion_artifacts_written": False,
        "feature_count": feature_count,
    }
    return research_json_result(payload)


def _normalized_horizons(raw: object, official_horizon: int) -> tuple[int, ...]:
    horizons = {int(value) for value in raw} if isinstance(raw, list) else {official_horizon}
    horizons.add(official_horizon)
    if any(horizon <= 0 for horizon in horizons):
        raise OperatorUsageError("--horizons and --official-horizon-days must be positive")
    return tuple(sorted(horizons))


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorUsageError(f"failed to load feature family metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OperatorUsageError(f"feature family metadata must be a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


__all__ = [
    "_research_campaign_attribute_feature_failures",
    "_research_campaign_diagnose_features",
]
