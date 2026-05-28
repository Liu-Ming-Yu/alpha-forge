"""IB historical market-data runtime coordination."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.domain.market_data import SUPPORTED_BAR_SECONDS, MarketBar
from quant_platform.services.execution_service.ib.ib_historical_data_sync import (
    fetch_raw_historical_bars,
)
from quant_platform.services.execution_service.ib.ib_historical_pacing import (
    IBHistoricalPacingLimiter,
)
from quant_platform.services.execution_service.ib.ib_market_data_mapper import (
    bar_size_string,
    market_bar_from_raw,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable
    from datetime import date

    from quant_platform.services.execution_service.stores.pacing_store import HistoricalPacingStore

log = structlog.get_logger(__name__)


class IBHistoricalMarketDataRuntime:
    """Coordinate IB historical-bar pacing, deduplication, and mapping."""

    def __init__(
        self,
        *,
        client: object,
        wrapper: object,
        timeout: float,
        client_id: int,
        enabled: bool,
        pacing_window_seconds: float,
        pacing_max_requests: int,
        pacing_store: HistoricalPacingStore | None,
        instrument_contracts: dict[uuid.UUID, dict[str, object]],
        require_connected: Callable[[], None],
        resolve_contract: Callable[[uuid.UUID], object],
    ) -> None:
        self._client = client
        self._wrapper = wrapper
        self._timeout = timeout
        self._client_id = client_id
        self._enabled = enabled
        self._pacing_limiter = IBHistoricalPacingLimiter(
            client_id=client_id,
            window_seconds=pacing_window_seconds,
            max_requests=pacing_max_requests,
            store=pacing_store,
        )
        self._instrument_contracts = instrument_contracts
        self._require_connected = require_connected
        self._resolve_contract = resolve_contract
        self._in_flight: dict[tuple[uuid.UUID, int], asyncio.Future[MarketBar | None]] = {}
        self._lock = asyncio.Lock()
        self._next_req_id = 8000

    async def reserve_pacing_slot(self) -> None:
        """Enforce the IB historical-data pacing budget."""
        await self._pacing_limiter.reserve()

    async def hydrate_pacing_if_needed(self) -> None:
        """Populate in-memory pacing state from the durable store once."""
        await self._pacing_limiter.hydrate_if_needed()

    async def get_last_bar(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> MarketBar | None:
        """Fetch the most recent completed bar for ``instrument_id``."""
        if not self._enabled:
            return None
        if bar_seconds not in SUPPORTED_BAR_SECONDS:
            raise ValueError(f"bar_seconds {bar_seconds} not supported")

        self._require_connected()

        contract_spec = self._instrument_contracts.get(instrument_id)
        if contract_spec is None:
            log.warning(
                "broker_gateway.get_last_bar.unmapped",
                instrument_id=str(instrument_id),
            )
            return None

        dedup_key = (instrument_id, bar_seconds)
        async with self._lock:
            existing = self._in_flight.get(dedup_key)
            if existing is not None:
                try:
                    return await asyncio.shield(existing)
                except Exception as exc:
                    log.debug(
                        "broker_gateway.get_last_bar.in_flight_failed",
                        instrument_id=str(instrument_id),
                        bar_seconds=bar_seconds,
                        error=str(exc),
                    )
            loop = asyncio.get_running_loop()
            result_future: asyncio.Future[MarketBar | None] = loop.create_future()
            self._in_flight[dedup_key] = result_future

        try:
            await self.reserve_pacing_slot()
            contract = self._resolve_contract(instrument_id)
            bar_size = bar_size_string(bar_seconds)
            duration = "2 D" if bar_seconds >= 86400 else "1 D"
            req_id = self._next_req_id
            self._next_req_id += 1

            try:
                raw_bars = await fetch_raw_historical_bars(
                    client=self._client,
                    wrapper=self._wrapper,
                    timeout=self._timeout,
                    req_id=req_id,
                    contract=contract,
                    end_date_time="",
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show="TRADES",
                    use_rth=1,
                    format_date=1,
                    keep_up_to_date=False,
                )
            except TimeoutError:
                log.warning(
                    "broker_gateway.get_last_bar.timeout",
                    instrument_id=str(instrument_id),
                    bar_seconds=bar_seconds,
                )
                if not result_future.done():
                    result_future.set_result(None)
                return None

            if not raw_bars:
                log.info(
                    "broker_gateway.get_last_bar.empty",
                    instrument_id=str(instrument_id),
                    bar_seconds=bar_seconds,
                )
                if not result_future.done():
                    result_future.set_result(None)
                return None

            bar = market_bar_from_raw(
                instrument_id=instrument_id,
                bar_seconds=bar_seconds,
                raw=raw_bars[-1],
            )
            if not result_future.done():
                result_future.set_result(bar)
            return bar
        except Exception as exc:
            if not result_future.done():
                result_future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._in_flight.pop(dedup_key, None)

    async def fetch_historical_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        end_date: date,
        duration: str = "1 D",
        what_to_show: str = "TRADES",
    ) -> list[MarketBar]:
        """Fetch a window of historical bars ending at ``end_date``."""
        if not self._enabled:
            return []
        if bar_seconds not in SUPPORTED_BAR_SECONDS:
            raise ValueError(f"bar_seconds {bar_seconds} not supported")

        self._require_connected()

        contract_spec = self._instrument_contracts.get(instrument_id)
        if contract_spec is None:
            log.warning(
                "broker_gateway.fetch_historical_bars.unmapped",
                instrument_id=str(instrument_id),
            )
            return []

        await self.reserve_pacing_slot()
        contract = self._resolve_contract(instrument_id)
        bar_size = bar_size_string(bar_seconds)
        end_date_time = f"{end_date.strftime('%Y%m%d')}-23:59:59"

        req_id = self._next_req_id
        self._next_req_id += 1

        try:
            raw_bars = await fetch_raw_historical_bars(
                client=self._client,
                wrapper=self._wrapper,
                timeout=self._timeout,
                req_id=req_id,
                contract=contract,
                end_date_time=end_date_time,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=1,
                format_date=1,
                keep_up_to_date=False,
            )
        except TimeoutError:
            log.warning(
                "broker_gateway.fetch_historical_bars.timeout",
                instrument_id=str(instrument_id),
                end_date=str(end_date),
            )
            return []

        out: list[MarketBar] = []
        for raw in raw_bars:
            try:
                bar = market_bar_from_raw(
                    instrument_id=instrument_id,
                    bar_seconds=bar_seconds,
                    raw=raw,
                )
                if bar.low <= 0 or bar.high <= 0:
                    continue
                out.append(bar)
            except Exception as exc:
                log.warning(
                    "broker_gateway.fetch_historical_bars.bad_row",
                    instrument_id=str(instrument_id),
                    error=str(exc),
                )
                continue

        return out
