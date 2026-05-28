"""Shared HistoricalDataStore contract tests for CA-adjusted reads.

Both adapters — the production ``ParquetBarStore`` and the in-memory
``InMemoryBarStore`` used in tests — must return corporate-action-adjusted
bars per the ``HistoricalDataStore`` contract.  Before Phase 1.3 (parity
and data completeness plan) the in-memory store returned raw bars, which
caused tests using it to silently diverge from production behaviour.

Each test here is parameterised over both stores; a regression in either
adapter fails at the same assertion.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from quant_platform.core.domain.instruments import (
    CorporateAction,
    CorporateActionType,
)
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.stores.bar_store import InMemoryBarStore
from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from quant_platform.core.contracts.data import HistoricalDataStore

_UTC = UTC


def _make_bar(
    instrument_id: uuid.UUID,
    day: datetime,
    close: Decimal,
    volume: int = 1_000_000,
) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=day,
        bar_seconds=86400,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        vwap=None,
        is_complete=True,
    )


def _make_split(
    instrument_id: uuid.UUID,
    ex_date: date,
    ratio: Decimal,
) -> CorporateAction:
    return CorporateAction(
        action_id=uuid.uuid4(),
        instrument_id=instrument_id,
        action_type=CorporateActionType.SPLIT,
        ex_date=ex_date,
        record_date=ex_date,
        pay_date=ex_date,
        ratio=ratio,
        cash_amount=Decimal("0"),
        currency="USD",
        supersedes_id=None,
        notes="test",
    )


def _make_dividend(
    instrument_id: uuid.UUID,
    ex_date: date,
    cash_amount: Decimal,
) -> CorporateAction:
    return CorporateAction(
        action_id=uuid.uuid4(),
        instrument_id=instrument_id,
        action_type=CorporateActionType.DIVIDEND,
        ex_date=ex_date,
        record_date=ex_date,
        pay_date=ex_date,
        ratio=Decimal("1"),
        cash_amount=cash_amount,
        currency="USD",
        supersedes_id=None,
        notes="test",
    )


@pytest.fixture
def in_memory_store() -> Callable[[], InMemoryBarStore]:
    def _factory() -> InMemoryBarStore:
        return InMemoryBarStore()

    return _factory


@pytest.fixture
def parquet_store(tmp_path: Path) -> Callable[[], ParquetBarStore]:
    # Each invocation gets its own subdirectory so multiple stores inside one
    # test do not share CA state.
    counter = {"n": 0}

    def _factory() -> ParquetBarStore:
        counter["n"] += 1
        root = tmp_path / f"store_{counter['n']}"
        return ParquetBarStore(root)

    return _factory


_STORE_FIXTURES = ["in_memory_store", "parquet_store"]


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_fixture", _STORE_FIXTURES)
async def test_split_adjusts_historical_bars(
    factory_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """A 2-for-1 split must halve prices and double volume for pre-split bars."""
    factory: Callable[[], HistoricalDataStore] = request.getfixturevalue(factory_fixture)
    store = factory()
    instrument_id = uuid.uuid4()

    split = _make_split(
        instrument_id,
        ex_date=date(2020, 1, 1),
        ratio=Decimal("2"),
    )
    await store.store_corporate_action(split)
    await store.store_bars(
        [
            _make_bar(instrument_id, datetime(2019, 6, 1, tzinfo=_UTC), Decimal("200"), 500_000),
        ]
    )

    bars = await store.get_bars(
        instrument_id,
        86400,
        datetime(2019, 1, 1, tzinfo=_UTC),
        datetime(2019, 12, 31, tzinfo=_UTC),
    )
    assert len(bars) == 1
    assert bars[0].close == Decimal("100")
    assert bars[0].volume == 1_000_000


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_fixture", _STORE_FIXTURES)
async def test_post_split_bars_are_untouched(
    factory_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Bars on or after the split ex_date are assumed post-split and untouched."""
    factory: Callable[[], HistoricalDataStore] = request.getfixturevalue(factory_fixture)
    store = factory()
    instrument_id = uuid.uuid4()

    split = _make_split(
        instrument_id,
        ex_date=date(2020, 1, 1),
        ratio=Decimal("2"),
    )
    await store.store_corporate_action(split)
    await store.store_bars(
        [
            _make_bar(instrument_id, datetime(2020, 6, 1, tzinfo=_UTC), Decimal("100")),
        ]
    )

    bars = await store.get_bars(
        instrument_id,
        86400,
        datetime(2020, 1, 1, tzinfo=_UTC),
        datetime(2020, 12, 31, tzinfo=_UTC),
    )
    assert len(bars) == 1
    assert bars[0].close == Decimal("100")


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_fixture", _STORE_FIXTURES)
async def test_dividend_subtracts_cash_amount_pre_ex_date(
    factory_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """A $1 dividend must subtract $1 from pre-ex_date OHLC."""
    factory: Callable[[], HistoricalDataStore] = request.getfixturevalue(factory_fixture)
    store = factory()
    instrument_id = uuid.uuid4()

    dividend = _make_dividend(
        instrument_id,
        ex_date=date(2021, 3, 15),
        cash_amount=Decimal("1.00"),
    )
    await store.store_corporate_action(dividend)
    await store.store_bars(
        [
            _make_bar(instrument_id, datetime(2021, 2, 1, tzinfo=_UTC), Decimal("50.00")),
        ]
    )

    bars = await store.get_bars(
        instrument_id,
        86400,
        datetime(2021, 1, 1, tzinfo=_UTC),
        datetime(2021, 3, 14, tzinfo=_UTC),
    )
    assert len(bars) == 1
    assert bars[0].close == Decimal("49.00")


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_fixture", _STORE_FIXTURES)
async def test_pre_window_split_is_returned_by_get_corporate_actions(
    factory_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Pre-window CAs are visible to reprocess tooling and adjustment code."""
    factory: Callable[[], HistoricalDataStore] = request.getfixturevalue(factory_fixture)
    store = factory()
    instrument_id = uuid.uuid4()
    split = _make_split(instrument_id, date(2020, 1, 1), Decimal("2"))
    await store.store_corporate_action(split)

    actions = await store.get_corporate_actions(instrument_id, since=date(2023, 1, 1))
    assert len(actions) == 1
    assert actions[0].ex_date == date(2020, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_fixture", _STORE_FIXTURES)
async def test_corporate_action_decimal_round_trip_is_lossless(
    factory_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    """Persisted ratio/cash_amount must round-trip without precision loss.

    Regression: the prior parquet schema serialised these as float64 so
    non-integer ratios (e.g. odd-lot splits) and exact dividend amounts
    were truncated before reaching disk.
    """
    factory: Callable[[], HistoricalDataStore] = request.getfixturevalue(factory_fixture)
    store = factory()
    instrument_id = uuid.uuid4()

    odd_split = CorporateAction(
        action_id=uuid.uuid4(),
        instrument_id=instrument_id,
        action_type=CorporateActionType.SPLIT,
        ex_date=date(2024, 6, 15),
        record_date=date(2024, 6, 15),
        pay_date=date(2024, 6, 15),
        ratio=Decimal("3.0000001"),
        cash_amount=Decimal("0"),
        currency="USD",
        supersedes_id=None,
        notes=None,
    )
    odd_dividend = CorporateAction(
        action_id=uuid.uuid4(),
        instrument_id=instrument_id,
        action_type=CorporateActionType.DIVIDEND,
        ex_date=date(2024, 7, 1),
        record_date=date(2024, 7, 1),
        pay_date=date(2024, 7, 1),
        ratio=Decimal("1"),
        cash_amount=Decimal("0.123456789012345678"),
        currency="USD",
        supersedes_id=None,
        notes=None,
    )
    await store.store_corporate_action(odd_split)
    await store.store_corporate_action(odd_dividend)

    actions = await store.get_corporate_actions(instrument_id, since=date(2020, 1, 1))
    by_id = {a.action_id: a for a in actions}
    assert by_id[odd_split.action_id].ratio == Decimal("3.0000001")
    assert by_id[odd_dividend.action_id].cash_amount == Decimal("0.123456789012345678")
