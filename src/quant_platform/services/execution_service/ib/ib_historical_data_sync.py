"""IB historical-data request helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from quant_platform.services.execution_service.ib.ib_market_data_mapper import RawIbBar


async def fetch_raw_historical_bars(
    *,
    client: object,
    wrapper: object,
    timeout: float,
    req_id: int,
    contract: object,
    end_date_time: str,
    duration: str,
    bar_size: str,
    what_to_show: str,
    use_rth: int = 1,
    format_date: int = 1,
    keep_up_to_date: bool = False,
) -> list[RawIbBar]:
    """Issue one IB historical-data request and resolve the callback future."""
    client_any = cast("Any", client)
    wrapper_any = cast("Any", wrapper)
    loop = asyncio.get_running_loop()
    raw_future: asyncio.Future[list[RawIbBar]] = loop.create_future()
    with wrapper_any._lifecycle_lock:
        wrapper_any._hist_futures[req_id] = raw_future
        wrapper_any._hist_data[req_id] = []

    client_any.reqHistoricalData(
        req_id,
        contract,
        end_date_time,
        duration,
        bar_size,
        what_to_show,
        use_rth,
        format_date,
        keep_up_to_date,
        [],
    )

    try:
        return await asyncio.wait_for(raw_future, timeout=timeout)
    except TimeoutError:
        with wrapper_any._lifecycle_lock:
            wrapper_any._hist_futures.pop(req_id, None)
            wrapper_any._hist_data.pop(req_id, None)
        raise
