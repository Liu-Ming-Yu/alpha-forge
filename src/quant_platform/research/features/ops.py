"""Feature repository and feature-audit operation composition."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.errors import OperatorUsageError
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.research.common import (
    _build_samples_to_path,
    _require_durable_research_inputs,
    _samples_result_payload,
    _verify_postgres_schema_if_configured,
    research_json_result,
)
from quant_platform.research.features.backfill_ops import _features_backfill
from quant_platform.research.intraday.feature_backfill_ops import (
    _features_backfill_intraday_alpha,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.application.research import (
        FeatureAuditRequest,
        FeaturesBuildSamplesRequest,
        FeaturesRequest,
    )
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)

__all__ = [
    "_features",
    "_features_audit",
    "_features_audit_assert",
    "_features_audit_retire",
    "_features_audit_run",
    "_features_audit_status",
    "_features_backfill",
    "_features_build_samples",
    "_features_retention",
]


async def _features_retention(
    settings: PlatformSettings,
    keep_days: int,
    dry_run: bool,
) -> UseCaseResult[dict[str, object]]:
    """Prune feature-vector rows older than ``keep_days``."""
    from quant_platform.infrastructure.support.clock import WallClock

    if keep_days <= 0:
        raise OperatorUsageError("--keep-days must be > 0")

    await _verify_postgres_schema_if_configured(settings)
    session = create_paper_session(settings=settings, initial_cash=Decimal("0"))
    cutoff = WallClock().now() - timedelta(days=keep_days)
    if dry_run:
        log.info("features_retention.dry_run", cutoff=cutoff.isoformat())
        return research_json_result({"dry_run": True, "cutoff": cutoff.isoformat()})
    if not hasattr(session.feature_repo, "prune"):
        raise OperatorUsageError(
            "configured feature repository does not implement prune(); "
            "wire a Postgres DSN or upgrade the repo."
        )
    deleted = await session.feature_repo.prune(cutoff)
    log.info(
        "features_retention.complete",
        cutoff=cutoff.isoformat(),
        deleted=deleted,
    )
    return research_json_result(
        {"dry_run": False, "cutoff": cutoff.isoformat(), "deleted": deleted}
    )


async def _features_build_samples(
    settings: PlatformSettings,
    request: FeaturesBuildSamplesRequest,
) -> UseCaseResult[dict[str, object]]:
    """Build supervised alpha samples from feature and bar stores."""
    _require_durable_research_inputs(settings)
    if settings.storage.postgres_dsn:
        await _verify_postgres_schema_if_configured(settings)
    path, result = await _build_samples_to_path(
        settings=settings,
        contracts_file=request.contracts_file,
        start=request.start,
        end=request.end,
        output=request.output,
        feature_set_version=request.feature_set_version,
        horizon_days=request.horizon_days,
        bar_seconds=request.bar_seconds,
        max_feature_age_days=request.max_feature_age_days,
        date_policy=request.date_policy,
    )
    payload = _samples_result_payload(path, result)
    log.info("features_build_samples.complete", **payload)
    return research_json_result(payload)


async def _features(
    settings: PlatformSettings,
    request: FeaturesRequest,
) -> UseCaseResult[dict[str, object]]:
    """Dispatch feature repository utilities."""
    from quant_platform.application.research import (
        FeatureAuditRequest,
        FeaturesBackfillIntradayAlphaRequest,
        FeaturesBackfillRequest,
        FeaturesBuildSamplesRequest,
    )

    if isinstance(request, FeaturesBuildSamplesRequest):
        return await _features_build_samples(settings, request)
    if isinstance(request, FeaturesBackfillRequest):
        return await _features_backfill(settings, request)
    if isinstance(request, FeaturesBackfillIntradayAlphaRequest):
        return await _features_backfill_intraday_alpha(settings, request)
    if isinstance(request, FeatureAuditRequest):
        return await _features_audit(settings, request)
    raise OperatorUsageError(f"unknown features request: {type(request).__name__}")


async def _features_audit(
    settings: PlatformSettings,
    request: FeatureAuditRequest,
) -> UseCaseResult[dict[str, object]]:
    """Run and inspect feature-level governance audits."""
    if request.command == "run":
        return await _features_audit_run(settings, request)
    if request.command == "status":
        return await _features_audit_status(settings, request)
    if request.command == "assert":
        return await _features_audit_assert(settings, request)
    if request.command == "retire":
        return await _features_audit_retire(settings, request)
    raise OperatorUsageError(f"unknown feature audit subcommand: {request.command}")


async def _features_audit_run(
    settings: PlatformSettings,
    request: FeatureAuditRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.application.features.governance import FeatureAuditRunRequest
    from quant_platform.research.feature_governance import build_feature_audit_use_case

    async def _sample_builder(audit_request: FeatureAuditRunRequest, output: Path) -> Path:
        _require_durable_research_inputs(settings)
        await _verify_postgres_schema_if_configured(settings)
        if audit_request.start is None or audit_request.end is None:
            raise ValueError("feature audit sample build requires start and end datetimes")
        samples_path, _sample_result = await _build_samples_to_path(
            settings=settings,
            contracts_file=str(audit_request.contracts_file or ""),
            start=audit_request.start,
            end=audit_request.end,
            output=output,
            feature_set_version=audit_request.feature_set_version,
            horizon_days=audit_request.horizon_days,
            bar_seconds=audit_request.bar_seconds,
            max_feature_age_days=audit_request.max_feature_age_days,
        )
        return samples_path

    if request.feature_card is None:
        raise OperatorUsageError("feature audit run requires --feature-card")
    audit_request = FeatureAuditRunRequest(
        feature_card=request.feature_card,
        samples=request.samples,
        contracts_file=request.contracts_file,
        start=request.start,
        end=request.end,
        feature_set_version=request.feature_set_version,
        horizon_days=request.horizon_days,
        bar_seconds=request.bar_seconds,
        max_feature_age_days=request.max_feature_age_days,
        output_root=request.output_root,
        baseline_features=request.baseline_features,
        slippage_bps_per_turnover=float(request.slippage_bps_per_turnover),
        min_daily_groups=request.min_daily_groups,
        min_coverage=request.min_coverage,
        min_oos_ic=request.min_oos_ic,
        min_icir=request.min_icir,
        max_negative_ic_streak=request.max_negative_ic_streak,
        max_turnover=request.max_turnover,
        persist=request.persist,
    )
    try:
        result = await build_feature_audit_use_case(
            settings,
            sample_builder=_sample_builder,
        ).run(audit_request)
    except ValueError as exc:
        raise OperatorUsageError(str(exc)) from exc
    return research_json_result(result.payload, passed=result.passed)


async def _features_audit_status(
    settings: PlatformSettings,
    request: FeatureAuditRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.application.features.governance import FeatureAuditStatusRequest
    from quant_platform.research.feature_governance import build_feature_audit_use_case

    result = await build_feature_audit_use_case(settings).status(
        FeatureAuditStatusRequest(
            feature_name=request.feature_name,
            feature_version=request.feature_version,
            limit=request.limit,
            output_root=request.output_root,
        )
    )
    return research_json_result(result.payload)


async def _features_audit_assert(
    settings: PlatformSettings,
    request: FeatureAuditRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.application.features.governance import FeatureAuditAssertRequest
    from quant_platform.research.feature_governance import build_feature_audit_use_case

    try:
        result = await build_feature_audit_use_case(settings).assert_latest(
            FeatureAuditAssertRequest(
                manifest=request.manifest,
                feature_name=request.feature_name,
                feature_version=request.feature_version,
                minimum_state=request.minimum_state,
            )
        )
    except ValueError as exc:
        raise OperatorUsageError(str(exc)) from exc
    return research_json_result(result.payload, passed=result.passed)


async def _features_audit_retire(
    settings: PlatformSettings,
    request: FeatureAuditRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.application.features.governance import FeatureAuditRetireRequest
    from quant_platform.research.feature_governance import build_feature_audit_use_case

    if request.feature_name is None or request.feature_version is None:
        raise OperatorUsageError(
            "feature audit retire requires --feature-name and --feature-version"
        )
    try:
        result = await build_feature_audit_use_case(settings).retire(
            FeatureAuditRetireRequest(
                feature_name=request.feature_name,
                feature_version=request.feature_version,
                feature_set_version=request.feature_set_version,
                reason=request.reason,
            )
        )
    except ValueError as exc:
        raise OperatorUsageError(str(exc)) from exc
    return research_json_result(result.payload)
