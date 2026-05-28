"""Pure quality checks for daily bar ingest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import uuid
    from datetime import date

    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.core.domain.market_data import MarketBar

log = structlog.get_logger(__name__)

MAX_ALLOWED_GAP_DAYS = 5


def filter_invariant_violations(
    bars: list[MarketBar],
    symbol_by_id: dict[uuid.UUID, str],
) -> tuple[list[MarketBar], dict[str, int]]:
    """Drop bars that violate OHLC sanity and return kept bars plus drop counts."""
    kept: list[MarketBar] = []
    drops: dict[str, int] = {}
    for bar in bars:
        try:
            ok = (
                bar.open > 0
                and bar.high > 0
                and bar.low > 0
                and bar.close > 0
                and bar.high >= max(bar.open, bar.close, bar.low)
                and bar.low <= min(bar.open, bar.close, bar.high)
            )
        except Exception:
            ok = False
        if ok:
            kept.append(bar)
        else:
            key = symbol_by_id.get(bar.instrument_id, str(bar.instrument_id))
            drops[key] = drops.get(key, 0) + 1
    return kept, drops


def check_quality(
    bars: list[MarketBar],
    instruments: list[Instrument],
    trade_date: date,
) -> list[str]:
    """Basic data quality checks: missing instruments and zero-volume flags."""
    warnings: list[str] = []

    bars_by_instrument: dict[uuid.UUID, list[MarketBar]] = {}
    for bar in bars:
        bars_by_instrument.setdefault(bar.instrument_id, []).append(bar)

    instrument_ids_with_bars = set(bars_by_instrument.keys())
    for inst in instruments:
        if inst.instrument_id not in instrument_ids_with_bars:
            warnings.append(f"{inst.symbol}: no bars returned for trade_date={trade_date}")

    for instrument_id, inst_bars in bars_by_instrument.items():
        daily = sorted(
            [b for b in inst_bars if b.bar_seconds == 86400],
            key=lambda b: b.timestamp,
        )
        if not daily:
            continue

        zero_vol = [b for b in daily if b.volume == 0]
        if len(zero_vol) > 3:
            warnings.append(
                f"{instrument_id}: {len(zero_vol)} zero-volume bars in last {len(daily)} days"
            )

    return warnings


def check_continuity(
    bars: list[MarketBar],
    instruments: list[Instrument],
) -> list[str]:
    """Detect abnormal gaps between consecutive daily bars per instrument."""
    warnings: list[str] = []
    symbol_by_id = {inst.instrument_id: inst.symbol for inst in instruments}

    bars_by_instrument: dict[uuid.UUID, list[MarketBar]] = {}
    for bar in bars:
        if bar.bar_seconds == 86400:
            bars_by_instrument.setdefault(bar.instrument_id, []).append(bar)

    for instrument_id, inst_bars in bars_by_instrument.items():
        daily = sorted(inst_bars, key=lambda b: b.timestamp)
        for i in range(1, len(daily)):
            gap_days = (daily[i].timestamp.date() - daily[i - 1].timestamp.date()).days
            if gap_days > MAX_ALLOWED_GAP_DAYS:
                symbol = symbol_by_id.get(instrument_id, str(instrument_id))
                msg = (
                    f"{symbol}: {gap_days}-day gap between "
                    f"{daily[i - 1].timestamp.date()} and {daily[i].timestamp.date()}"
                )
                warnings.append(msg)
                log.warning(
                    "daily_ingest.continuity_gap",
                    instrument_id=str(instrument_id),
                    symbol=symbol,
                    gap_days=gap_days,
                    from_date=str(daily[i - 1].timestamp.date()),
                    to_date=str(daily[i].timestamp.date()),
                )
    return warnings
