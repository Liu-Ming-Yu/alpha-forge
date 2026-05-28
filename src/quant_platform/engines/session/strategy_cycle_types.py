"""Shared callable types for strategy-cycle orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from quant_platform.application.runtime.state import Session
from quant_platform.services.signal_service.regime_detector import MarketStats

MarketStatsReader = Callable[[Session, datetime], Awaitable[MarketStats | None]]
