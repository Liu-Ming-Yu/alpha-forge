"""Build market-data providers used by engine sessions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

from quant_platform.services.data_service.feeds.live_market_data import PollingMarketDataProvider

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.contracts import MarketDataProvider
    from quant_platform.core.domain.market_data import MarketBar


def build_account_market_data_provider(
    account_broker: object,
    *,
    max_bar_age_minutes: int | None,
    daily_max_bar_age_minutes: int | None = None,
) -> MarketDataProvider | None:
    """Adapt an account broker's optional get_last_bar method into a provider."""
    fetch_last_bar = getattr(account_broker, "get_last_bar", None)
    if not callable(fetch_last_bar):
        return None

    async def _fetch_last_bar(
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> MarketBar | None:
        try:
            maybe_bar = fetch_last_bar(instrument_id, bar_seconds)
        except TypeError:
            return None
        if inspect.isawaitable(maybe_bar):
            maybe_bar = await maybe_bar
        return cast("MarketBar | None", maybe_bar)

    return PollingMarketDataProvider(
        _fetch_last_bar,
        max_bar_age_minutes=max_bar_age_minutes,
        daily_max_bar_age_minutes=daily_max_bar_age_minutes,
    )
