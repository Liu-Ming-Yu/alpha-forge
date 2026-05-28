"""Tests for ``ParquetBarStore.reprocess_corporate_actions``."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from quant_platform.core.domain.instruments import CorporateAction, CorporateActionType
from quant_platform.core.domain.market_data import MarketBar
from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore

if TYPE_CHECKING:
    from pathlib import Path


def _bar(instrument_id: uuid.UUID, day: date, close: Decimal) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid4(),
        instrument_id=instrument_id,
        timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
        bar_seconds=86400,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
        is_complete=True,
    )


@pytest.mark.asyncio
async def test_reprocess_writes_adjusted_partitions(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    instrument_id = uuid.uuid4()

    await store.store_bars(
        [
            _bar(instrument_id, date(2026, 1, 5), Decimal("100")),
            _bar(instrument_id, date(2026, 1, 6), Decimal("102")),
            _bar(instrument_id, date(2026, 3, 1), Decimal("52")),  # post-split price
        ]
    )

    action = CorporateAction(
        action_id=uuid.uuid4(),
        instrument_id=instrument_id,
        action_type=CorporateActionType.SPLIT,
        ex_date=date(2026, 2, 1),
        record_date=date(2026, 2, 1),
        pay_date=date(2026, 2, 1),
        ratio=Decimal("2"),
        cash_amount=Decimal("0"),
        currency="USD",
        notes="2-for-1 split",
    )
    await store.store_corporate_action(action)

    written = await store.reprocess_corporate_actions(instrument_id)
    assert written == 3

    adjusted_file = tmp_path / "bars_adjusted" / str(instrument_id) / "2026.parquet"
    assert adjusted_file.exists()

    table = pq.read_table(adjusted_file)
    rows = sorted(table.to_pylist(), key=lambda r: r["timestamp"])
    # Pre-split bars are halved; post-split bar is untouched.
    assert rows[0]["close"] == pytest.approx(50.0)
    assert rows[1]["close"] == pytest.approx(51.0)
    assert rows[2]["close"] == pytest.approx(52.0)


@pytest.mark.asyncio
async def test_reprocess_unknown_instrument_returns_zero(tmp_path: Path) -> None:
    store = ParquetBarStore(tmp_path)
    written = await store.reprocess_corporate_actions(uuid.uuid4())
    assert written == 0
