"""Unit tests for engine regime runtime helper."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from quant_platform.engines.runtime.regime import detect_session_regime

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_detect_session_regime_calls_detector_for_simple_detector() -> None:
    regime = object()
    detector = AsyncMock()
    detector.detect = AsyncMock(return_value=regime)
    session = SimpleNamespace(regime_detector=detector)

    result = await detect_session_regime(session, _AS_OF)

    assert result is regime
    detector.detect.assert_awaited_once_with(_AS_OF)
