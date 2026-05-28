"""Local-file historical intraday bar adapter and parser."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.market_data import MarketBar, VendorBarBatch
from quant_platform.services.data_service.intraday.intraday_schema import canonical_intraday_bar_id
from quant_platform.services.data_service.intraday.intraday_validation import (
    INTRADAY_BAR_SECONDS,
    ensure_utc,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd


class FileHistoricalBarVendorAdapter:
    """HistoricalBarVendorAdapter backed by an immutable local CSV/Parquet file."""

    def __init__(
        self,
        path: Path,
        *,
        vendor: str,
        instrument_lookup: Mapping[str, uuid.UUID],
    ) -> None:
        self._path = path
        self._vendor = vendor
        self._lookup = instrument_lookup

    async def fetch_bars(
        self,
        instruments: list[uuid.UUID],
        start: datetime,
        end: datetime,
        bar_seconds: int,
        *,
        as_of: datetime,
    ) -> VendorBarBatch:
        batch = load_vendor_bar_batch_from_file(
            self._path,
            vendor=self._vendor,
            instrument_lookup=self._lookup,
            bar_seconds=bar_seconds,
            as_of=as_of,
        )
        allowed = set(instruments)
        bars = tuple(
            bar
            for bar in batch.bars
            if bar.instrument_id in allowed and start <= ensure_utc(bar.timestamp) <= end
        )
        coverage = dict(batch.coverage)
        coverage["requested_instruments"] = len(allowed)
        coverage["filtered_rows"] = len(bars)
        return VendorBarBatch(
            vendor=batch.vendor,
            source_uri=batch.source_uri,
            fetched_at=batch.fetched_at,
            bar_seconds=batch.bar_seconds,
            bars=bars,
            coverage=coverage,
        )


def load_vendor_bar_batch_from_file(
    path: Path,
    *,
    vendor: str,
    instrument_lookup: Mapping[str, uuid.UUID],
    bar_seconds: int = INTRADAY_BAR_SECONDS,
    as_of: datetime | None = None,
    source_uri: str | None = None,
) -> VendorBarBatch:
    """Load a vendor file into canonical ``MarketBar`` rows."""
    if bar_seconds != INTRADAY_BAR_SECONDS:
        raise ValueError("industrial intraday imports require 1-minute bars")
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = _read_frame(path)
    missing = {"timestamp", "open", "high", "low", "close", "volume"} - set(frame.columns)
    if missing:
        raise ValueError(f"intraday file missing required columns: {sorted(missing)}")
    if "instrument_id" not in frame.columns and "symbol" not in frame.columns:
        raise ValueError("intraday file requires either instrument_id or symbol column")

    bars: list[MarketBar] = []
    for idx, row in frame.iterrows():
        instrument_id = _resolve_instrument(row, instrument_lookup, row_index=int(idx))
        timestamp = ensure_utc(_parse_datetime(row["timestamp"]))
        close = Decimal(str(row["close"]))
        vwap_raw = row.get("vwap")
        vwap = None if _is_missing(vwap_raw) else Decimal(str(vwap_raw))
        bars.append(
            MarketBar(
                bar_id=canonical_intraday_bar_id(instrument_id, timestamp, bar_seconds),
                instrument_id=instrument_id,
                timestamp=timestamp,
                bar_seconds=bar_seconds,
                open=Decimal(str(row["open"])),
                high=Decimal(str(row["high"])),
                low=Decimal(str(row["low"])),
                close=close,
                volume=int(row["volume"]),
                vwap=vwap,
                is_complete=bool(row.get("is_complete", True)),
            )
        )

    fetched_at = ensure_utc(as_of or datetime.now(tz=UTC))
    coverage = _coverage_for_bars(tuple(bars))
    return VendorBarBatch(
        vendor=vendor,
        source_uri=source_uri or path.resolve().as_uri(),
        fetched_at=fetched_at,
        bar_seconds=bar_seconds,
        bars=tuple(sorted(bars, key=lambda bar: (str(bar.instrument_id), bar.timestamp))),
        coverage=coverage,
    )


def write_vendor_bar_batch_to_file(
    batch: VendorBarBatch,
    path: Path,
    *,
    symbol_by_instrument_id: Mapping[uuid.UUID, str] | None = None,
) -> Path:
    """Freeze a canonical vendor batch as a deterministic local screen input."""
    frame = _bars_to_frame(batch, symbol_by_instrument_id or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return path
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return path
    raise ValueError("intraday freeze output supports .csv and .parquet files")


def _read_frame(path: Path) -> pd.DataFrame:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("intraday import supports .csv and .parquet files")


def _resolve_instrument(
    row: object, instrument_lookup: Mapping[str, uuid.UUID], *, row_index: int
) -> uuid.UUID:
    raw_id = row.get("instrument_id") if hasattr(row, "get") else None
    if not _is_missing(raw_id):
        try:
            return uuid.UUID(str(raw_id))
        except ValueError as exc:
            raise ValueError(f"row {row_index}: invalid instrument_id {raw_id!r}") from exc
    raw_symbol = row.get("symbol") if hasattr(row, "get") else None
    if _is_missing(raw_symbol):
        raise ValueError(f"row {row_index}: missing instrument_id or symbol")
    key = str(raw_symbol).upper()
    found = instrument_lookup.get(key)
    if found is None:
        raise ValueError(f"row {row_index}: unknown symbol {key!r}")
    return found


def _parse_datetime(raw: object) -> datetime:
    if isinstance(raw, datetime):
        return raw
    try:
        import pandas as pd

        value = pd.Timestamp(raw).to_pydatetime()
    except Exception as exc:
        raise ValueError(f"invalid timestamp {raw!r}") from exc
    if not isinstance(value, datetime):
        raise ValueError(f"invalid timestamp {raw!r}")
    return value


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:
        return False


def _coverage_for_bars(bars: tuple[MarketBar, ...]) -> dict[str, object]:
    by_instrument: dict[str, int] = {}
    for bar in bars:
        key = str(bar.instrument_id)
        by_instrument[key] = by_instrument.get(key, 0) + 1
    return {
        "row_count": len(bars),
        "instrument_count": len(by_instrument),
        "rows_by_instrument": by_instrument,
    }


def _bars_to_frame(
    batch: VendorBarBatch,
    symbol_by_instrument_id: Mapping[uuid.UUID, str],
) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for bar in sorted(batch.bars, key=lambda item: (str(item.instrument_id), item.timestamp)):
        rows.append(
            {
                "vendor": batch.vendor,
                "source_uri": batch.source_uri,
                "fetched_at": batch.fetched_at.isoformat(),
                "symbol": symbol_by_instrument_id.get(bar.instrument_id, ""),
                "instrument_id": str(bar.instrument_id),
                "timestamp": ensure_utc(bar.timestamp).isoformat(),
                "open": str(bar.open),
                "high": str(bar.high),
                "low": str(bar.low),
                "close": str(bar.close),
                "volume": int(bar.volume),
                "vwap": "" if bar.vwap is None else str(bar.vwap),
                "is_complete": bool(bar.is_complete),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "vendor",
            "source_uri",
            "fetched_at",
            "symbol",
            "instrument_id",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "is_complete",
        ],
    )


__all__ = [
    "FileHistoricalBarVendorAdapter",
    "load_vendor_bar_batch_from_file",
    "write_vendor_bar_batch_to_file",
]
