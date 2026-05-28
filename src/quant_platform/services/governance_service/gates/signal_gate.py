"""Generic signal-promotion gate helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import SignalGateRecord, SignalGateStatus

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import SignalPromotionGate


def build_performance_repository(_dsn: str | None) -> SignalPromotionGate:
    """Compatibility injection hook for tests; bootstrap supplies production repositories."""
    raise RuntimeError("signal gate repository must be supplied by bootstrap")


def build_signal_gate(
    settings: PlatformSettings,
    gate: SignalPromotionGate | None = None,
) -> SignalPromotionGate:
    """Build the configured signal-promotion gate repository."""
    if gate is not None:
        return gate
    return build_performance_repository(settings.storage.postgres_dsn)


async def record_signal_observation(
    settings: PlatformSettings,
    *,
    signal_name: str,
    signal_type: str,
    as_of: datetime,
    daily_ic: float,
    observations: int = 1,
    drawdown: float = 0.0,
    turnover: float = 0.0,
    gate: SignalPromotionGate | None = None,
) -> SignalGateStatus:
    """Record one signal observation and return updated gate status."""
    signal_gate = build_signal_gate(settings, gate)
    await signal_gate.record_signal_observation(
        SignalGateRecord(
            signal_name=signal_name,
            signal_type=signal_type,
            as_of=as_of,
            daily_ic=daily_ic,
            observations=observations,
            drawdown=drawdown,
            turnover=turnover,
        )
    )
    return await signal_gate_status(
        settings,
        signal_name=signal_name,
        signal_type=signal_type,
        as_of=as_of,
        gate=signal_gate,
    )


async def signal_gate_status(
    settings: PlatformSettings,
    *,
    signal_name: str,
    signal_type: str,
    as_of: datetime,
    gate: SignalPromotionGate | None = None,
) -> SignalGateStatus:
    """Read generic signal-promotion status."""
    signal_gate = build_signal_gate(settings, gate)
    return await signal_gate.signal_status(
        signal_name,
        signal_type,
        as_of=as_of,
        min_observations=settings.production.text_gate_min_observations,
        min_ic=settings.production.text_gate_min_ic,
        max_negative_streak=settings.production.text_gate_max_negative_streak,
        drawdown_limit=settings.production.signal_gate_max_drawdown,
        turnover_limit=settings.production.signal_gate_max_turnover,
    )
