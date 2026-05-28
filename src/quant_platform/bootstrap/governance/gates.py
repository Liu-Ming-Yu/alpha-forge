"""Signal and text gate governance command wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.results import ResultPresentation, UseCaseResult, UseCaseStatus
from quant_platform.bootstrap.governance.repositories import build_performance_repository

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings


async def signal_gate_command(
    settings: PlatformSettings,
    *,
    subcommand: str,
    signal_name: str,
    signal_type: str,
    as_of: datetime,
    daily_ic: float | None = None,
    observations: int = 1,
    drawdown: float = 0.0,
    turnover: float = 0.0,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.governance_service.gates.signal_gate import (
        record_signal_observation,
        signal_gate_status,
    )

    gate = build_performance_repository(settings.storage.postgres_dsn)
    if subcommand == "record":
        if daily_ic is None:
            raise OperatorUsageError("signal-gate record requires --daily-ic")
        status = await record_signal_observation(
            settings,
            signal_name=signal_name,
            signal_type=signal_type,
            as_of=as_of,
            daily_ic=daily_ic,
            observations=observations,
            drawdown=drawdown,
            turnover=turnover,
            gate=gate,
        )
    else:
        status = await signal_gate_status(
            settings,
            signal_name=signal_name,
            signal_type=signal_type,
            as_of=as_of,
            gate=gate,
        )
    payload = {**vars(status), "passed": status.passed, "state": status.state.value}
    blocked = subcommand == "assert" and not status.passed
    return UseCaseResult(
        status=UseCaseStatus.BLOCKED if blocked else UseCaseStatus.OK,
        payload=payload,
        exit_code=2 if blocked else 0,
        presentation=ResultPresentation.JSON,
    )


async def text_gate_command(
    settings: PlatformSettings,
    *,
    subcommand: str,
    strategy_name: str,
    as_of: datetime,
    daily_ic: float | None = None,
    observations: int = 1,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.governance_service.gates.text_gate import (
        record_text_ic,
        text_gate_status,
    )

    gate = build_performance_repository(settings.storage.postgres_dsn)
    if subcommand == "record":
        if daily_ic is None:
            raise OperatorUsageError("text-gate record requires --daily-ic")
        status = await record_text_ic(
            settings,
            strategy_name=strategy_name,
            as_of=as_of,
            daily_ic=daily_ic,
            observations=observations,
            gate=gate,
        )
    else:
        status = await text_gate_status(
            settings,
            strategy_name=strategy_name,
            as_of=as_of,
            gate=gate,
        )
    blocked = subcommand == "assert" and not status.passed
    return UseCaseResult(
        status=UseCaseStatus.BLOCKED if blocked else UseCaseStatus.OK,
        payload=vars(status),
        exit_code=2 if blocked else 0,
        presentation=ResultPresentation.JSON,
    )


__all__ = ["signal_gate_command", "text_gate_command"]
