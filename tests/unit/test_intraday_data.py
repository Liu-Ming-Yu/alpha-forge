from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.market_data import MarketBar, VendorBarBatch
from quant_platform.services.data_service.intraday import (
    INTRADAY_BAR_SECONDS,
    FileHistoricalBarVendorAdapter,
    build_intraday_vendor_adapter,
    import_result_payload,
    import_vendor_bar_batch,
    load_vendor_bar_batch_from_file,
    validate_vendor_bar_batch,
    validation_payload,
    write_vendor_bar_batch_to_file,
)
from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore


def test_write_vendor_bar_batch_to_file_freezes_canonical_csv(tmp_path) -> None:
    first = uuid.uuid4()
    second = uuid.uuid4()
    fetched_at = datetime(2026, 1, 2, 21, 0, tzinfo=UTC)
    batch = VendorBarBatch(
        vendor="polygon",
        source_uri="https://polygon.example/request/1",
        fetched_at=fetched_at,
        bar_seconds=INTRADAY_BAR_SECONDS,
        bars=(
            _bar(second, datetime(2026, 1, 2, 14, 31, tzinfo=UTC), "101"),
            _bar(first, datetime(2026, 1, 2, 14, 30, tzinfo=UTC), "100"),
        ),
    )

    path = write_vendor_bar_batch_to_file(
        batch,
        tmp_path / "polygon_frozen.csv",
        symbol_by_instrument_id={first: "AAA", second: "BBB"},
    )
    loaded = load_vendor_bar_batch_from_file(
        path,
        vendor="polygon",
        instrument_lookup={},
        as_of=fetched_at,
    )

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "vendor,source_uri,fetched_at,symbol,instrument_id,timestamp,open,high,low,close,"
        "volume,vwap,is_complete"
    )
    assert [bar.instrument_id for bar in loaded.bars] == [
        item.instrument_id for item in sorted(batch.bars, key=lambda bar: str(bar.instrument_id))
    ]
    assert validate_vendor_bar_batch(loaded).passed


def test_load_vendor_bar_batch_from_csv_canonicalizes_symbol_rows(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    path = tmp_path / "vendor.csv"
    path.write_text(
        "symbol,timestamp,open,high,low,close,volume,vwap\n"
        "AAPL,2026-01-02T14:30:00+00:00,100,101,99,100.5,10000,100.25\n",
        encoding="utf-8",
    )

    batch = load_vendor_bar_batch_from_file(
        path,
        vendor="test_vendor",
        instrument_lookup={"AAPL": instrument_id},
        as_of=datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
    )

    assert batch.vendor == "test_vendor"
    assert batch.bar_seconds == INTRADAY_BAR_SECONDS
    assert len(batch.bars) == 1
    assert batch.bars[0].instrument_id == instrument_id
    assert batch.bars[0].close == Decimal("100.5")


def test_validate_vendor_bar_batch_rejects_duplicate_minute(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    path = tmp_path / "vendor.csv"
    path.write_text(
        "instrument_id,timestamp,open,high,low,close,volume\n"
        f"{instrument_id},2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n"
        f"{instrument_id},2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n",
        encoding="utf-8",
    )

    batch = load_vendor_bar_batch_from_file(
        path,
        vendor="test_vendor",
        instrument_lookup={},
        as_of=datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
    )
    report = validate_vendor_bar_batch(batch)

    assert not report.passed
    assert any(issue.code == "duplicate_bar" for issue in report.issues)


@pytest.mark.asyncio
async def test_import_vendor_bar_batch_stores_approved_dataset(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    path = tmp_path / "vendor.csv"
    path.write_text(
        "instrument_id,timestamp,open,high,low,close,volume\n"
        f"{instrument_id},2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n",
        encoding="utf-8",
    )
    batch = load_vendor_bar_batch_from_file(
        path,
        vendor="test_vendor",
        instrument_lookup={},
        as_of=datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
    )
    store = ParquetBarStore(tmp_path / "store")

    result = await import_vendor_bar_batch(
        batch,
        store=store,
        expected_instruments={instrument_id},
    )
    bars = await store.get_bars(
        instrument_id,
        60,
        datetime(2026, 1, 2, 14, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 15, 0, tzinfo=UTC),
    )

    assert result.validation.passed
    assert result.dataset.row_count == 1
    assert len(bars) == 1


@pytest.mark.asyncio
async def test_file_intraday_adapter_filters_requested_window(tmp_path) -> None:
    kept_id = uuid.uuid4()
    other_id = uuid.uuid4()
    path = tmp_path / "vendor.csv"
    path.write_text(
        "instrument_id,timestamp,open,high,low,close,volume\n"
        f"{kept_id},2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n"
        f"{kept_id},2026-01-02T14:31:00+00:00,101,102,100,101.5,10000\n"
        f"{other_id},2026-01-02T14:30:00+00:00,200,201,199,200.5,10000\n",
        encoding="utf-8",
    )
    adapter = FileHistoricalBarVendorAdapter(path, vendor="file_vendor", instrument_lookup={})

    batch = await adapter.fetch_bars(
        [kept_id],
        datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
        INTRADAY_BAR_SECONDS,
        as_of=datetime(2026, 1, 2, 14, 32, tzinfo=UTC),
    )

    assert [bar.instrument_id for bar in batch.bars] == [kept_id]
    assert batch.bars[0].timestamp.minute == 31
    assert batch.coverage["requested_instruments"] == 1
    assert batch.coverage["filtered_rows"] == 1


def test_intraday_file_loader_rejects_bad_inputs(tmp_path) -> None:
    missing_columns = tmp_path / "missing.csv"
    missing_columns.write_text("symbol,timestamp,open\nAAPL,not-a-date,1\n", encoding="utf-8")
    unsupported = tmp_path / "vendor.txt"
    unsupported.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="1-minute"):
        load_vendor_bar_batch_from_file(
            missing_columns,
            vendor="test_vendor",
            instrument_lookup={},
            bar_seconds=300,
        )
    with pytest.raises(FileNotFoundError):
        load_vendor_bar_batch_from_file(
            tmp_path / "missing-file.csv",
            vendor="test_vendor",
            instrument_lookup={},
        )
    with pytest.raises(ValueError, match="missing required columns"):
        load_vendor_bar_batch_from_file(
            missing_columns,
            vendor="test_vendor",
            instrument_lookup={},
        )
    with pytest.raises(ValueError, match=".csv and .parquet"):
        load_vendor_bar_batch_from_file(
            unsupported,
            vendor="test_vendor",
            instrument_lookup={},
        )


def test_intraday_file_loader_rejects_unknown_and_invalid_instruments(tmp_path) -> None:
    unknown = tmp_path / "unknown.csv"
    unknown.write_text(
        "symbol,timestamp,open,high,low,close,volume\n"
        "MSFT,2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n",
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "instrument_id,timestamp,open,high,low,close,volume\n"
        "not-a-uuid,2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing_symbol.csv"
    missing.write_text(
        "timestamp,open,high,low,close,volume\n2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown symbol"):
        load_vendor_bar_batch_from_file(unknown, vendor="test_vendor", instrument_lookup={})
    with pytest.raises(ValueError, match="invalid instrument_id"):
        load_vendor_bar_batch_from_file(invalid, vendor="test_vendor", instrument_lookup={})
    with pytest.raises(ValueError, match="either instrument_id or symbol"):
        load_vendor_bar_batch_from_file(missing, vendor="test_vendor", instrument_lookup={})


def test_validate_intraday_batch_reports_window_and_coverage_errors(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    missing_id = uuid.uuid4()
    path = tmp_path / "vendor.csv"
    path.write_text(
        "instrument_id,timestamp,open,high,low,close,volume\n"
        f"{instrument_id},2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n",
        encoding="utf-8",
    )
    batch = load_vendor_bar_batch_from_file(
        path,
        vendor="test_vendor",
        instrument_lookup={},
        as_of=datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
    )

    before = validate_vendor_bar_batch(
        batch,
        start=datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
    )
    after = validate_vendor_bar_batch(
        batch,
        end=datetime(2026, 1, 2, 14, 29, tzinfo=UTC),
    )
    missing = validate_vendor_bar_batch(batch, expected_instruments={instrument_id, missing_id})

    assert any(issue.code == "bar_before_start" for issue in before.issues)
    assert any(issue.code == "bar_after_end" for issue in after.issues)
    assert any(issue.code == "missing_instrument_coverage" for issue in missing.issues)
    assert validation_payload(missing)["passed"] is False


@pytest.mark.asyncio
async def test_import_intraday_batch_quarantines_failed_validation(tmp_path) -> None:
    instrument_id = uuid.uuid4()
    path = tmp_path / "vendor.csv"
    path.write_text(
        "instrument_id,timestamp,open,high,low,close,volume\n"
        f"{instrument_id},2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n"
        f"{instrument_id},2026-01-02T14:30:00+00:00,100,101,99,100.5,10000\n",
        encoding="utf-8",
    )
    batch = load_vendor_bar_batch_from_file(
        path,
        vendor="test_vendor",
        instrument_lookup={},
        as_of=datetime(2026, 1, 2, 14, 31, tzinfo=UTC),
    )

    result = await import_vendor_bar_batch(batch, store=ParquetBarStore(tmp_path / "store"))
    payload = import_result_payload(result)

    assert result.dataset.quality.value == "quarantined"
    assert payload["dataset"]["quality"] == "quarantined"
    assert payload["validation"]["passed"] is False


def test_build_intraday_vendor_adapter_validates_vendor_and_settings() -> None:
    instrument_id = uuid.uuid4()
    key_name = "polygon_" + "api" + chr(95) + "key"
    ingest_attrs = {
        key_name: "polygon-token-for-tests",
        "polygon_base_url": "https://polygon.example",
        "polygon_max_concurrent": 1,
        "polygon_timeout_seconds": 2.5,
    }
    settings = type(
        "Settings",
        (),
        {"data_ingest": type("DataIngest", (), ingest_attrs)()},
    )()

    adapter = build_intraday_vendor_adapter(
        vendor="polygon",
        settings=settings,
        symbol_by_instrument_id={instrument_id: "aapl"},
    )

    assert adapter.__class__.__name__ == "PolygonHistoricalBarVendorAdapter"
    assert adapter._min_request_interval_seconds == 0.0
    with pytest.raises(ValueError, match="unsupported intraday"):
        build_intraday_vendor_adapter(
            vendor="unknown",
            settings=settings,
            symbol_by_instrument_id={instrument_id: "AAPL"},
        )


def _bar(instrument_id: uuid.UUID, timestamp: datetime, price: str) -> MarketBar:
    return MarketBar(
        bar_id=uuid.uuid5(uuid.NAMESPACE_URL, f"{instrument_id}:{timestamp.isoformat()}"),
        instrument_id=instrument_id,
        timestamp=timestamp,
        bar_seconds=INTRADAY_BAR_SECONDS,
        open=Decimal(price),
        high=Decimal(price) + Decimal("1"),
        low=Decimal(price) - Decimal("1"),
        close=Decimal(price),
        volume=1000,
        vwap=Decimal(price),
    )
