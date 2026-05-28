"""Market-data delegates for the IB gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import uuid
    from datetime import date

    from quant_platform.core.domain.market_data import MarketBar


class IBHistoricalMarketDataPort(Protocol):
    async def reserve_pacing_slot(self) -> None: ...
    async def hydrate_pacing_if_needed(self) -> None: ...
    async def get_last_bar(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> MarketBar | None: ...
    async def fetch_historical_bars(
        self,
        *,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        end_date: date,
        duration: str,
        what_to_show: str,
    ) -> list[MarketBar]: ...


class IBGatewayMarketDataMixin:
    """Historical market-data methods for the IB gateway facade."""

    _historical_market_data: object

    async def _reserve_pacing_slot(self) -> None:
        await self._market_data_runtime.reserve_pacing_slot()

    async def _hydrate_pacing_if_needed(self) -> None:
        await self._market_data_runtime.hydrate_pacing_if_needed()

    async def get_last_bar(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> MarketBar | None:
        return await self._market_data_runtime.get_last_bar(instrument_id, bar_seconds)

    async def fetch_historical_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        end_date: date,
        duration: str = "1 D",
        what_to_show: str = "TRADES",
    ) -> list[MarketBar]:
        return await self._market_data_runtime.fetch_historical_bars(
            instrument_id=instrument_id,
            bar_seconds=bar_seconds,
            end_date=end_date,
            duration=duration,
            what_to_show=what_to_show,
        )

    @property
    def _market_data_runtime(self) -> IBHistoricalMarketDataPort:
        return cast("IBHistoricalMarketDataPort", self._historical_market_data)
