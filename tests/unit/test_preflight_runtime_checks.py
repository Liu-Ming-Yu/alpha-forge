"""Unit tests for runtime preflight checks: broker connectivity and data freshness."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from quant_platform.core.contracts.common import BrokerHealth, BrokerHealthStatus
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.governance_service.preflight import (
    check_broker_connectivity,
    check_data_freshness,
)

_UTC = UTC
_NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=_UTC)
_INSTRUMENT = uuid.uuid4()


def _healthy_broker() -> object:
    broker = MagicMock()
    broker.health_check = AsyncMock(
        return_value=BrokerHealth(
            status=BrokerHealthStatus.CONNECTED,
            latency_ms=5,
            last_heartbeat_at=_NOW,
        )
    )
    return broker


def _unhealthy_broker(status: BrokerHealthStatus = BrokerHealthStatus.DISCONNECTED) -> object:
    broker = MagicMock()
    broker.health_check = AsyncMock(
        return_value=BrokerHealth(
            status=status,
            latency_ms=0,
            last_heartbeat_at=_NOW,
        )
    )
    return broker


def _fresh_bar(age_minutes: float = 5.0) -> MarketBar:
    from datetime import datetime

    timestamp = datetime.now(tz=UTC) - timedelta(minutes=age_minutes)
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=_INSTRUMENT,
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("102"),
        volume=10000,
        timestamp=timestamp,
        bar_seconds=86400,
    )


class TestCheckBrokerConnectivity:
    @pytest.mark.asyncio
    async def test_connected_broker_passes(self) -> None:
        check = await check_broker_connectivity(_healthy_broker())
        assert check.passed
        assert check.name == "broker_connectivity"
        assert check.severity == "error"

    @pytest.mark.asyncio
    async def test_disconnected_broker_fails(self) -> None:
        check = await check_broker_connectivity(_unhealthy_broker(BrokerHealthStatus.DISCONNECTED))
        assert not check.passed
        assert "disconnected" in check.detail.lower()

    @pytest.mark.asyncio
    async def test_degraded_broker_fails(self) -> None:
        check = await check_broker_connectivity(_unhealthy_broker(BrokerHealthStatus.DEGRADED))
        assert not check.passed

    @pytest.mark.asyncio
    async def test_exception_from_health_check_fails_closed(self) -> None:
        broker = MagicMock()
        broker.health_check = AsyncMock(side_effect=ConnectionError("cannot reach broker"))
        check = await check_broker_connectivity(broker)
        assert not check.passed
        assert "raised" in check.detail.lower()


class TestCheckDataFreshness:
    @pytest.mark.asyncio
    async def test_fresh_bar_passes(self) -> None:
        provider = MagicMock()
        provider.get_last_bar = AsyncMock(return_value=_fresh_bar(age_minutes=30))
        check = await check_data_freshness(provider, _INSTRUMENT, 86400, max_age_minutes=60)
        assert check.passed
        assert check.name == "data_freshness"

    @pytest.mark.asyncio
    async def test_stale_bar_fails(self) -> None:
        provider = MagicMock()
        provider.get_last_bar = AsyncMock(return_value=_fresh_bar(age_minutes=120))
        check = await check_data_freshness(provider, _INSTRUMENT, 86400, max_age_minutes=60)
        assert not check.passed
        assert "120" in check.detail or "age" in check.detail.lower()

    @pytest.mark.asyncio
    async def test_missing_bar_fails(self) -> None:
        provider = MagicMock()
        provider.get_last_bar = AsyncMock(return_value=None)
        check = await check_data_freshness(provider, _INSTRUMENT, 86400, max_age_minutes=60)
        assert not check.passed
        assert "no bar" in check.detail.lower()

    @pytest.mark.asyncio
    async def test_exception_from_provider_fails_closed(self) -> None:
        provider = MagicMock()
        provider.get_last_bar = AsyncMock(side_effect=RuntimeError("connection lost"))
        check = await check_data_freshness(provider, _INSTRUMENT, 86400, max_age_minutes=60)
        assert not check.passed
        assert "raised" in check.detail.lower()

    @pytest.mark.asyncio
    async def test_bar_just_within_max_age_passes(self) -> None:
        provider = MagicMock()
        provider.get_last_bar = AsyncMock(return_value=_fresh_bar(age_minutes=59))
        check = await check_data_freshness(provider, _INSTRUMENT, 86400, max_age_minutes=60)
        assert check.passed
