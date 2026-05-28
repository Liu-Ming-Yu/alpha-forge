"""Signal-source helpers for production-candidate gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.governance_service.gates.signal_gate import signal_gate_status

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime

    from quant_platform.config import PlatformSettings
    from quant_platform.core.contracts import SignalPromotionGate
    from quant_platform.core.domain.production import SignalGateStatus

CLASSICAL_SOURCE = "classical"


def _resolve_signal_sources(
    explicit: Iterable[str] | None,
    settings: PlatformSettings,
) -> tuple[str, ...]:
    if explicit is not None:
        return tuple(dict.fromkeys(name.strip() for name in explicit if name.strip()))
    weights = settings.alpha.source_weights
    return tuple(name for name, weight in weights.items() if weight > 0)


async def _primary_signal_status(
    settings: PlatformSettings,
    sources: tuple[str, ...],
    as_of: datetime,
    *,
    campaign_payload: Mapping[str, object] | None = None,
    signal_sources_explicit: bool = False,
    signal_name: str | None = None,
    signal_type: str | None = None,
    signal_gate: SignalPromotionGate | None = None,
) -> SignalGateStatus | None:
    """Pick the representative signal-gate status for readiness.

    The readiness report already includes a single ``signal_gate_passed``
    check.  By default it remains the classical source gate when classical is
    active; promoted source and campaign-specific checks are evaluated
    separately.  Operators can override this representative identity when they
    need a source-specific readiness view.
    """
    del campaign_payload, signal_sources_explicit
    identity = _primary_signal_identity(
        sources,
        signal_name=signal_name,
        signal_type=signal_type,
    )
    if identity is None:
        return None
    preferred_name, preferred_type = identity
    return await signal_gate_status(
        settings,
        signal_name=preferred_name,
        signal_type=preferred_type,
        as_of=as_of,
        gate=signal_gate,
    )


def _primary_signal_identity(
    sources: tuple[str, ...],
    *,
    signal_name: str | None = None,
    signal_type: str | None = None,
) -> tuple[str, str] | None:
    explicit_name = str(signal_name or "").strip()
    explicit_type = str(signal_type or "").strip()
    if explicit_name:
        return explicit_name, explicit_type or explicit_name

    if not sources:
        return None
    if CLASSICAL_SOURCE in sources:
        return CLASSICAL_SOURCE, CLASSICAL_SOURCE
    source = sources[0]
    return source, source
