"""FastAPI lifespan helpers for the operator API."""

from __future__ import annotations

import asyncio
import signal
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


def build_operator_api_lifespan(
    shutting_down: list[bool],
) -> Callable[[object], AbstractAsyncContextManager[None]]:
    """Return a lifespan context manager that tracks process shutdown signals."""

    @asynccontextmanager
    async def _lifespan(_application: object) -> AsyncIterator[None]:
        shutting_down[0] = False

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
