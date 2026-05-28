"""Runtime helpers for engine regime detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Protocol

    from quant_platform.application.runtime.state import Session
    from quant_platform.core.domain.signals import RegimeState

    class RegimeStateDetector(Protocol):
        async def detect(self, as_of: datetime) -> RegimeState: ...


async def detect_session_regime(session: Session, as_of: datetime) -> RegimeState:
    """Refresh market stats for live regime detectors, then detect regime."""
    from quant_platform.engines.session.public_api import _compute_market_stats_from_store
    from quant_platform.services.signal_service.regime_detector import MarketRegimeDetector

    detector = session.regime_detector
    if detector is None:
        raise RuntimeError("engine session is missing a regime detector")
    if isinstance(detector, MarketRegimeDetector):
        stats = await _compute_market_stats_from_store(session, as_of)
        if stats is not None:
            detector.update(stats)
    return await cast("RegimeStateDetector", detector).detect(as_of)
