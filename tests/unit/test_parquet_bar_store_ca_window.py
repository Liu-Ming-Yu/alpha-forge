"""Regression tests for ParquetBarStore.get_corporate_actions window filter.

Pre-window CorporateActions must be returned so in-window bar adjustment
and CA reprocess tooling can account for retroactive corrections.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

import quant_platform.services.data_service.stores.parquet_bar_store as parquet_mod
from quant_platform.core.domain.instruments import (
    CorporateAction,
    CorporateActionType,
)
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore

if TYPE_CHECKING:
    from pathlib import Path

_UTC = UTC


def _make_bar(
    instrument_id: uuid.UUID,
    day: datetime,
    close: Decimal,
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
        volume=1_000_000,
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
        notes="test_split",
    )


@pytest.mark.asyncio
async def test_get_corporate_actions_returns_pre_window_split(
    tmp_path: Path,
) -> None:
    """CAs with ex_date < since must still be returned.

    The previous filter dropped them; this caused bar adjustment and CA
    reprocess tooling to silently miss retroactive corrections.
    """
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    pre_window_split = _make_split(
        instrument_id=instrument_id,
        ex_date=date(2020, 1, 1),
        ratio=Decimal("2"),
    )
    await store.store_corporate_action(pre_window_split)

    actions = await store.get_corporate_actions(
        instrument_id,
        since=date(2023, 1, 1),
    )

    assert len(actions) == 1
    assert actions[0].action_id == pre_window_split.action_id
    assert actions[0].ex_date == date(2020, 1, 1)


@pytest.mark.asyncio
async def test_get_bars_applies_pre_window_split_to_historical_bars(
    tmp_path: Path,
) -> None:
    """Pre-window splits still adjust stored pre-split bars at read time.

    Store the split on 2020-01-01, store bars from 2019 in pre-split units,
    query the 2019 window, and verify the split was applied.  This covers
    the adjustment path that depends on the (previously filtered)
    pre-window CA list.
    """
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    split = _make_split(
        instrument_id=instrument_id,
        ex_date=date(2020, 1, 1),
        ratio=Decimal("2"),
    )
    await store.store_corporate_action(split)

    pre_split_bar = _make_bar(
        instrument_id=instrument_id,
        day=datetime(2019, 6, 1, tzinfo=_UTC),
        close=Decimal("200"),
    )
    await store.store_bars([pre_split_bar])

    bars = await store.get_bars(
        instrument_id=instrument_id,
        bar_seconds=86400,
        start=datetime(2019, 1, 1, tzinfo=_UTC),
        end=datetime(2019, 12, 31, tzinfo=_UTC),
    )
    assert len(bars) == 1
    # The 2019 bar must be split-adjusted (200 -> 100).
    assert bars[0].close == Decimal("100")


@pytest.mark.asyncio
async def test_get_corporate_actions_empty_when_instrument_unknown(
    tmp_path: Path,
) -> None:
    store = ParquetBarStore(tmp_path)
    actions = await store.get_corporate_actions(
        uuid.uuid4(),
        since=date(2020, 1, 1),
    )
    assert actions == []


@pytest.mark.asyncio
async def test_file_lock_serializes_concurrent_partition_writers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bars" / str(uuid.uuid4()) / "2026.parquet"
    active_writers = 0
    max_active_writers = 0
    writer_order: list[int] = []
    guard = threading.Lock()

    def _writer(index: int) -> None:
        nonlocal active_writers, max_active_writers
        with parquet_mod._exclusive_file_lock(path):  # noqa: SLF001
            with guard:
                active_writers += 1
                max_active_writers = max(max_active_writers, active_writers)
            time.sleep(0.01)
            writer_order.append(index)
            with guard:
                active_writers -= 1

    threads = [threading.Thread(target=_writer, args=(idx,)) for idx in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert max_active_writers == 1
    assert sorted(writer_order) == list(range(5))


@pytest.mark.asyncio
async def test_concurrent_corporate_action_writers_are_idempotent(
    tmp_path: Path,
) -> None:
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()
    action = _make_split(
        instrument_id=instrument_id,
        ex_date=date(2026, 1, 5),
        ratio=Decimal("2"),
    )

    await store.store_corporate_action(action)
    await store.store_corporate_action(action)

    actions = await store.get_corporate_actions(instrument_id, since=date.min)
    assert [stored.action_id for stored in actions] == [action.action_id]
