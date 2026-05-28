"""CLI adapter for current promoted-source forecast materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quant_platform.application.research.common import research_json_result
from quant_platform.bootstrap.alpha_forecast_ops.materialize import materialize_alpha_forecasts

if TYPE_CHECKING:
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


async def alpha_materialize_forecasts_command(
    settings: PlatformSettings, args: Any
) -> UseCaseResult[dict[str, object]]:
    payload = await materialize_alpha_forecasts(
        settings,
        contracts_file=args.contracts_file,
        as_of=args.as_of,
        sources=tuple(args.source or ()),
        horizon=str(args.horizon),
        xgboost_manifest=args.xgboost_manifest,
        fail_on_missing=bool(args.fail_on_missing),
    )
    return research_json_result(payload, passed=bool(payload.get("passed")))


__all__ = ["alpha_materialize_forecasts_command"]
