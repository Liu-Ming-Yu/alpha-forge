"""Tests for ``DataMaintenanceSupervisor.backfill_once``."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.instruments import AssetClass, Instrument
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.maintenance.maintenance_scheduler import (
    DataMaintenanceScheduler,
)
from quant_platform.services.data_service.maintenance.maintenance_supervisor import (
    DataMaintenanceSupervisor,
)
from quant_platform.services.data_service.reference.contract_master import ContractMaster
from quant_platform.services.data_service.reference.universe_manager import UniverseManager
from quant_platform.services.data_service.stores.bar_store import InMemoryBarStore


class _FakeRegistry:
    async def due_feature_jobs(self, as_of: datetime) -> list[object]:
        return []

    async def mark_job_completed(
        self, job_id: uuid.UUID, *, run_at: datetime, success: bool
    ) -> None:
        return None


def _make_instrument(symbol: str = "AAPL") -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        exchange="XNAS",
        currency="USD",
    )


def _build_supervisor_with_fetcher(
    fetcher,
    instruments: list[Instrument],
) -> DataMaintenanceSupervisor:
    bar_store = InMemoryBarStore()
    universe = UniverseManager(contract_master=ContractMaster(instruments))
    scheduler = DataMaintenanceScheduler(
        instruments=instruments,
        bar_store=bar_store,
        universe_manager=universe,
        feature_repo=None,  # type: ignore[arg-type]
    )
    return DataMaintenanceSupervisor(
        model_registry=_FakeRegistry(),
        scheduler=scheduler,
        strategy_run_id=uuid.uuid4(),
        instruments=instruments,
        bar_store=bar_store,
        universe_manager=universe,
        bar_fetcher=fetcher,
    )


@pytest.mark.asyncio
async def test_backfill_once_fetches_and_stores_bars() -> None:
    inst = _make_instrument()
    bar = MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=inst.instrument_id,
        timestamp=datetime(2026, 1, 6, tzinfo=UTC),
        bar_seconds=86400,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1000,
        is_complete=True,
    )

    async def fetcher(
        instruments: list[Instrument],
        start: date,
        end: date,
    ) -> list[MarketBar]:
        return [bar]

    supervisor = _build_supervisor_with_fetcher(fetcher, [inst])
    result = await supervisor.backfill_once(date(2026, 1, 5), date(2026, 1, 10))

    assert result.bars_fetched == 1
    assert result.bars_stored == 1


@pytest.mark.asyncio
async def test_backfill_once_requires_fetcher() -> None:
    inst = _make_instrument()
    bar_store = InMemoryBarStore()
    universe = UniverseManager(contract_master=ContractMaster([inst]))
    scheduler = DataMaintenanceScheduler(
        instruments=[inst],
        bar_store=bar_store,
        universe_manager=universe,
        feature_repo=None,  # type: ignore[arg-type]
    )
    supervisor = DataMaintenanceSupervisor(
        model_registry=_FakeRegistry(),
        scheduler=scheduler,
        strategy_run_id=uuid.uuid4(),
    )
    with pytest.raises(ValueError, match="backfill_once requires"):
        await supervisor.backfill_once(date(2026, 1, 5), date(2026, 1, 10))


@pytest.mark.asyncio
async def test_backfill_once_rejects_inverted_range() -> None:
    async def fetcher(
        instruments: list[Instrument],
        start: date,
        end: date,
    ) -> list[MarketBar]:
        return []

    supervisor = _build_supervisor_with_fetcher(fetcher, [_make_instrument()])
    with pytest.raises(ValueError, match="end=.* must be >= start="):
        await supervisor.backfill_once(date(2026, 1, 10), date(2026, 1, 5))
