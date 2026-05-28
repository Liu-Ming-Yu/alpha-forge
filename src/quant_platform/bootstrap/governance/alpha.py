"""Alpha promotion governance command wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.research.common import research_json_result
from quant_platform.bootstrap.governance.repositories import (
    build_model_registry,
    build_performance_repository,
)
from quant_platform.bootstrap.persistence.migrations import verify_postgres_schema

if TYPE_CHECKING:
    from quant_platform.application.research import AlphaRequest
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


async def alpha_command(
    settings: PlatformSettings,
    request: AlphaRequest,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.governance_service.alpha import (
        alpha_assert,
        alpha_promote,
        alpha_ramp,
        alpha_rollback,
    )

    args = request
    subcommand = request.command
    if subcommand == "assert":
        if args.as_of is None:
            raise OperatorUsageError("alpha assert requires --as-of")
        if settings.storage.postgres_dsn:
            await verify_postgres_schema(settings)
        performance_repo = build_performance_repository(settings.storage.postgres_dsn)
        model_registry = (
            build_model_registry(settings.storage.postgres_dsn)
            if settings.storage.postgres_dsn
            else None
        )
        payload = await alpha_assert(
            settings,
            signal_name=args.signal_name,
            signal_type=args.signal_type,
            as_of=args.as_of,
            artifact_manifest=args.artifact_manifest,
            signal_gate=performance_repo,
            model_registry=model_registry,
        )
        return research_json_result(payload, passed=bool(payload.get("passed")))
    if subcommand == "promote":
        if args.as_of is None:
            raise OperatorUsageError("alpha promote requires --as-of")
        await verify_postgres_schema(settings)
        registry = build_model_registry(settings.storage.postgres_dsn)
        performance_repo = build_performance_repository(settings.storage.postgres_dsn)
        payload = await alpha_promote(
            settings,
            signal_name=args.signal_name,
            signal_type=args.signal_type,
            model_version=args.model_version,
            feature_set_version=args.feature_set_version,
            engine_version=args.engine_version,
            artifact_manifest=args.artifact_manifest,
            rollback_target=args.rollback_target,
            as_of=args.as_of,
            model_registry=registry,
            heartbeat_repository=performance_repo,
        )
        return research_json_result(payload)
    if subcommand == "rollback":
        if args.as_of is None:
            raise OperatorUsageError("alpha rollback requires --as-of")
        await verify_postgres_schema(settings)
        registry = build_model_registry(settings.storage.postgres_dsn)
        performance_repo = build_performance_repository(settings.storage.postgres_dsn)
        payload = await alpha_rollback(
            settings,
            signal_name=args.signal_name,
            target_version=args.target_version,
            as_of=args.as_of,
            model_registry=registry,
            heartbeat_repository=performance_repo,
        )
        return research_json_result(payload)
    if subcommand == "materialize-forecasts":
        from quant_platform.bootstrap.alpha_forecast_ops import (
            alpha_materialize_forecasts_command,
        )

        return await alpha_materialize_forecasts_command(settings, args)
    if subcommand == "ramp":
        return research_json_result(alpha_ramp(settings, clean_live_days=args.clean_live_days))
    raise OperatorUsageError(f"unknown alpha subcommand: {subcommand}")


__all__ = ["alpha_command"]
