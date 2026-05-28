"""Text-signal promotion gate helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.core.domain.production import TextSignalGateRecord, TextSignalGateStatus

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import TextSignalPromotionGate


def build_performance_repository(_dsn: str | None) -> TextSignalPromotionGate:
    """Compatibility injection hook for tests; bootstrap supplies production repositories."""
    raise RuntimeError("text signal gate repository must be supplied by bootstrap")


def build_text_signal_gate(
    settings: PlatformSettings,
    gate: TextSignalPromotionGate | None = None,
) -> TextSignalPromotionGate:
    """Build the configured text-signal promotion gate repository."""
    if gate is not None:
        return gate
    return build_performance_repository(settings.storage.postgres_dsn)


async def record_text_ic(
    settings: PlatformSettings,
    *,
    strategy_name: str,
    as_of: datetime,
    daily_ic: float,
    observations: int = 1,
    gate: TextSignalPromotionGate | None = None,
) -> TextSignalGateStatus:
    """Record one daily IC observation and return updated gate status."""
    text_gate = build_text_signal_gate(settings, gate)
    await text_gate.record_ic(
        TextSignalGateRecord(
            strategy_name=strategy_name,
            as_of=as_of,
            daily_ic=daily_ic,
            observations=observations,
        )
    )
    return await text_gate.status(
        strategy_name,
        as_of=as_of,
        min_observations=settings.production.text_gate_min_observations,
        min_ic=settings.production.text_gate_min_ic,
        max_negative_streak=settings.production.text_gate_max_negative_streak,
    )


async def text_gate_status(
    settings: PlatformSettings,
    *,
    strategy_name: str,
    as_of: datetime,
    gate: TextSignalPromotionGate | None = None,
) -> TextSignalGateStatus:
    """Read text-signal promotion status."""
    text_gate = build_text_signal_gate(settings, gate)
    return await text_gate.status(
        strategy_name,
        as_of=as_of,
        min_observations=settings.production.text_gate_min_observations,
        min_ic=settings.production.text_gate_min_ic,
        max_negative_streak=settings.production.text_gate_max_negative_streak,
    )
