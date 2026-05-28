"""IB Gateway connection lifecycle helpers."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

import structlog
from ibapi.client import EClient

from quant_platform.core.exceptions import BrokerUnavailableError

if TYPE_CHECKING:
    from quant_platform.core.domain.orders import BrokerOrder

log = structlog.get_logger(__name__)


class IBGatewayConnectionLifecycleMixin:
    """Connect/disconnect transport lifecycle for ``IBGatewayBrokerGateway``."""

    _client: EClient
    _client_id: int
    _connected: bool
    _host: str
    _max_local_order_id: int
    _port: int
    _reader_thread: threading.Thread | None
    _timeout: float
    _wrapper: Any

    if TYPE_CHECKING:

        async def _hydrate_pacing_if_needed(self) -> None: ...

        async def fetch_open_orders(self) -> list[BrokerOrder]: ...

    async def connect(self) -> None:
        if self._connected:
            return

        loop = asyncio.get_running_loop()
        self._wrapper.set_loop(loop)

        attempts = 3
        for attempt in range(attempts):
            self._wrapper._connect_event.clear()
            self._wrapper._connect_error_event.clear()
            self._wrapper._connect_error = None

            log.info(
                "broker_gateway.connecting",
                host=self._host,
                port=self._port,
                client_id=self._client_id,
                attempt=attempt + 1,
            )
            # EClient.connect() performs the IB API version handshake with two
            # blocking socket reads before returning.  Running it off the event
            # loop keeps timeouts, signal handling, and other coroutines alive.
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self._client.connect, self._host, self._port, self._client_id
                    ),
                    timeout=self._timeout,
                )
            except TimeoutError as exc:
                await self._close_transport()
                raise BrokerUnavailableError(
                    f"IB Gateway did not complete the API handshake within {self._timeout}s "
                    f"for {self._host}:{self._port} client_id={self._client_id}. "
                    "Verify that the IB Gateway/TWS API is enabled: "
                    "Configure -> API -> Enable ActiveX and Socket Clients."
                ) from exc
            except Exception as exc:
                await self._close_transport()
                raise BrokerUnavailableError(
                    f"IB Gateway socket connection failed for {self._host}:{self._port} "
                    f"client_id={self._client_id}: {exc}. "
                    "If running from WSL, use host.docker.internal or the Windows host IP "
                    "instead of 127.0.0.1 for a Windows-hosted Gateway."
                ) from exc

            self._reader_thread = threading.Thread(
                target=self._run_reader, daemon=True, name=f"ib-reader-{self._client_id}"
            )
            self._reader_thread.start()

            connected = await self._wait_for_connect_result()
            if connected:
                break

            connect_error = self._wrapper._connect_error
            await self._close_transport()
            if connect_error is not None and connect_error[0] == 326:
                if attempt < attempts - 1:
                    delay = 0.5 * (attempt + 1)
                    log.warning(
                        "broker_gateway.client_id_in_use_retrying",
                        client_id=self._client_id,
                        attempt=attempt + 1,
                        delay=delay,
                        message=connect_error[1],
                    )
                    await asyncio.sleep(delay)
                    continue
                raise BrokerUnavailableError(
                    f"IB Gateway client_id={self._client_id} is already in use after "
                    f"{attempts} connect attempts: {connect_error[1]}"
                )

            raise BrokerUnavailableError(
                f"IB Gateway did not send nextValidId within {self._timeout}s "
                f"for {self._host}:{self._port} client_id={self._client_id}. "
                "If running from WSL, do not use 127.0.0.1 for a Windows-hosted "
                "Gateway; use host.docker.internal or the Windows host IP. "
                "Also disable 'Allow connections from localhost only' in IB Gateway/TWS "
                "API settings and add the WSL client IP to Trusted IPs if prompted."
            )
        else:  # pragma: no cover - defensive; loop exits via break/raise
            raise BrokerUnavailableError("IB Gateway connection failed")

        self._connected = True

        # Enforce monotonic order IDs across reconnects.  If TWS restarted and
        # sent a nextValidId lower than what we issued in this process, bump it
        # forward so we never reuse an IB order ID.
        if (
            self._max_local_order_id > 0
            and self._wrapper._next_order_id is not None
            and self._wrapper._next_order_id <= self._max_local_order_id
        ):
            safe_next = self._max_local_order_id + 1
            log.warning(
                "broker_gateway.order_id.monotonic_repair",
                ib_next=self._wrapper._next_order_id,
                local_max=self._max_local_order_id,
                repaired_to=safe_next,
            )
            self._wrapper._next_order_id = safe_next

        await self._hydrate_pacing_if_needed()
        open_orders = await self.fetch_open_orders()
        log.info("broker_gateway.connected", open_orders=len(open_orders))

    async def disconnect(self) -> None:
        await self._close_transport()
        log.info("broker_gateway.disconnected")

    async def _wait_for_connect_result(self) -> bool:
        deadline = asyncio.get_running_loop().time() + self._timeout
        while asyncio.get_running_loop().time() < deadline:
            if self._wrapper._connect_event.is_set():
                return True
            if self._wrapper._connect_error_event.is_set():
                return False
            await asyncio.sleep(0.05)
        return False

    async def _close_transport(self) -> None:
        try:
            self._client.disconnect()
        except Exception:
            log.debug("broker_gateway.disconnect_error", exc_info=True)
        thread = self._reader_thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 2.0)
        self._reader_thread = None
        self._connected = False
        self._wrapper._connect_event.clear()
        self._wrapper._connect_error_event.clear()
        self._wrapper._connect_error = None
        self._client = EClient(self._wrapper)

    def _run_reader(self) -> None:
        """Reader-thread target: run the ibapi message loop."""
        try:
            self._client.run()
        except Exception:
            log.exception("ib_reader_thread.crashed", client_id=self._client_id)
            self._wrapper._connect_error_event.set()
