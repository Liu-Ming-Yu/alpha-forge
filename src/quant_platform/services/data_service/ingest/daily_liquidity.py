"""Liquidity profile calculations for daily bars."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.services.data_service.reference.universe_manager import LiquidityProfile

if TYPE_CHECKING:
    import uuid
    from datetime import date, datetime

    from quant_platform.core.domain.instruments import Instrument
    from quant_platform.core.domain.market_data import MarketBar


def compute_liquidity_profiles(
    bars: list[MarketBar],
    instruments: list[Instrument],
    trade_date: date,
    now: datetime,
) -> list[LiquidityProfile]:
    """Compute 20-day ADV profiles from daily bar data."""
    del trade_date
    bars_by_instrument: dict[uuid.UUID, list[MarketBar]] = {}
    for bar in bars:
        if bar.bar_seconds == 86400:
            bars_by_instrument.setdefault(bar.instrument_id, []).append(bar)

    profiles: list[LiquidityProfile] = []
    for inst in instruments:
        daily = bars_by_instrument.get(inst.instrument_id, [])
        daily = sorted(daily, key=lambda b: b.timestamp, reverse=True)[:20]
        if not daily:
            continue

        total_volume = sum(b.volume for b in daily)
        avg_volume_d = Decimal(total_volume) / Decimal(len(daily))
        last_close = daily[0].close
        adv_usd_d = avg_volume_d * last_close

        profiles.append(
            LiquidityProfile(
                instrument_id=inst.instrument_id,
                adv_shares_20d=float(avg_volume_d),
                adv_usd_20d=float(adv_usd_d),
                last_close=last_close,
                computed_at=now,
            )
        )

    return profiles
