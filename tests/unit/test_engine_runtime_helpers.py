"""Unit tests for engine runtime helper modules."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from quant_platform.core.exceptions import DataStalenessError
from quant_platform.engines.market_data.provider import build_account_market_data_provider
from quant_platform.engines.runtime.registry import check_model_staleness, maybe_await

_AS_OF = datetime(2026, 2, 3, 14, 30, tzinfo=UTC)


@pytest.mark.asyncio
async def test_maybe_await_accepts_sync_and_async_values() -> None:
    async def _async_value() -> int:
        return 42

    assert await maybe_await(7) == 7
    assert await maybe_await(_async_value()) == 42


@pytest.mark.asyncio
async def test_check_model_staleness_raises_for_old_registered_model() -> None:
    model = SimpleNamespace(
        strategy_name="equity",
        model_version="1.2.3",
        created_at=_AS_OF - timedelta(hours=25),
    )

    with pytest.raises(DataStalenessError, match="registered model is stale"):
        await check_model_staleness(model, as_of=_AS_OF, max_age_hours=24)


@pytest.mark.asyncio
async def test_check_model_staleness_accepts_naive_recent_model_timestamp() -> None:
    model = SimpleNamespace(created_at=(_AS_OF - timedelta(hours=1)).replace(tzinfo=None))

    await check_model_staleness(model, as_of=_AS_OF, max_age_hours=24)


@pytest.mark.asyncio
async def test_account_market_data_provider_returns_none_without_probe() -> None:
    assert build_account_market_data_provider(object(), max_bar_age_minutes=None) is None


@pytest.mark.asyncio
async def test_account_market_data_provider_wraps_sync_probe() -> None:
    bar = object()
    instrument_id = uuid.uuid4()

    class _Broker:
        def get_last_bar(self, requested_id: uuid.UUID, bar_seconds: int) -> object:
            assert requested_id == instrument_id
            assert bar_seconds == 60
            return bar

    provider = build_account_market_data_provider(_Broker(), max_bar_age_minutes=None)

    assert provider is not None
    assert await provider.get_last_bar(instrument_id, 60) is bar


@pytest.mark.asyncio
async def test_account_market_data_provider_wraps_async_probe() -> None:
    bar = object()

    class _Broker:
        async def get_last_bar(self, _instrument_id: uuid.UUID, _bar_seconds: int) -> object:
            return bar

    provider = build_account_market_data_provider(_Broker(), max_bar_age_minutes=None)

    assert provider is not None
    assert await provider.get_last_bar(uuid.uuid4(), 60) is bar


@pytest.mark.asyncio
async def test_account_market_data_provider_treats_signature_mismatch_as_missing() -> None:
    class _Broker:
        def get_last_bar(self) -> object:
            raise AssertionError("signature mismatch should be swallowed as TypeError")

    provider = build_account_market_data_provider(_Broker(), max_bar_age_minutes=None)

    assert provider is not None
    assert await provider.get_last_bar(uuid.uuid4(), 60) is None
