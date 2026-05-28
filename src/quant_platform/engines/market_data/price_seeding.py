"""Reference-price helpers for engine order planning."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.exceptions import DataStalenessError

if TYPE_CHECKING:
    import uuid

log = structlog.get_logger(__name__)


async def latest_contract_market_prices(
    *,
    exec_session: object,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    existing: dict[uuid.UUID, Decimal],
    as_of: datetime,
) -> dict[uuid.UUID, Decimal]:
    """Seed prices from contract ``last_close`` or latest daily bar close."""
    if not instrument_contracts:
        return {}
    prices: dict[uuid.UUID, Decimal] = {}
    bar_store = getattr(exec_session, "bar_store", None)
    for instrument_id, contract in instrument_contracts.items():
        if instrument_id in existing:
            continue
        raw_last_close = contract.get("last_close")
        if raw_last_close is not None:
            close = Decimal(str(raw_last_close))
            if close > 0:
                prices[instrument_id] = close
                continue
        if bar_store is None:
            continue
        try:
            bars = await bar_store.get_bars(
                instrument_id,
                86400,
                as_of - timedelta(days=30),
                as_of,
            )
        except Exception as exc:
            log.debug(
                "engine.market_price_backfill_failed",
                instrument_id=str(instrument_id),
                error=str(exc),
            )
            continue
        if not bars:
            continue
        latest = max(bars, key=lambda bar: bar.timestamp)
        close = Decimal(str(latest.close))
        if close > 0:
            prices[instrument_id] = close
    return prices


async def build_cycle_market_prices(
    *,
    session: object,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
    existing: dict[uuid.UUID, Decimal] | None,
    as_of: datetime,
) -> dict[uuid.UUID, Decimal]:
    """Merge explicit prices with contract/bar-store reference closes."""
    prices = dict(existing or {})
    prices.update(
        await latest_contract_market_prices(
            exec_session=session,
            instrument_contracts=instrument_contracts,
            existing=prices,
            as_of=as_of,
        )
    )
    return prices


def validate_positive_feature_prices(
    *,
    engine_name: str,
    feature_data: dict[uuid.UUID, dict[str, float]],
    market_prices: dict[uuid.UUID, Decimal],
) -> None:
    """Fail closed when order-capable engines lack prices for scored names."""
    missing = [
        instrument_id
        for instrument_id in feature_data
        if market_prices.get(instrument_id) is None or market_prices[instrument_id] <= Decimal("0")
    ]
    if not missing:
        return
    preview = ", ".join(str(instrument_id) for instrument_id in missing[:5])
    log.error(
        "engine_runner.market_prices_missing",
        engine=engine_name,
        missing_count=len(missing),
        sample=preview,
    )
    raise DataStalenessError(
        f"{engine_name} missing positive reference prices for "
        f"{len(missing)} feature instruments: {preview}"
    )
