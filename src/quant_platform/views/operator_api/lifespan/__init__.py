"""FastAPI lifespan helpers for the operator API."""

from __future__ import annotations

import asyncio
import signal
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

log = structlog.get_logger(__name__)


def build_operator_api_lifespan(
    shutting_down: list[bool],
    *,
    on_startup: Callable[[], Awaitable[None]] | None = None,
) -> Callable[[object], AbstractAsyncContextManager[None]]:
    """Return a lifespan context manager that tracks process shutdown signals.

    ``on_startup`` runs once after the app starts (used to hydrate the cash
    ledger from the latest persisted account snapshot). A failure there is
    logged but never blocks startup.
    """

    @asynccontextmanager
    async def _lifespan(_application: object) -> AsyncIterator[None]:
        shutting_down[0] = False
        if on_startup is not None:
            try:
                await on_startup()
            except Exception as exc:  # pragma: no cover - defensive startup path
                log.warning("operator_api.startup_hook_failed", error=str(exc))

        def _handle_shutdown(_signum: int, _frame: object) -> None:
            shutting_down[0] = True
            try:
                loop = asyncio.get_running_loop()
                loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass

        old_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown)
        old_sigint = signal.signal(signal.SIGINT, _handle_shutdown)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.signal(signal.SIGINT, old_sigint)

    return _lifespan
