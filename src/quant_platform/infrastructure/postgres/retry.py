"""Shared retry helper for PostgreSQL infrastructure adapters."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar

import structlog
from sqlalchemy.exc import OperationalError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

_T = TypeVar("_T")

log = structlog.get_logger(__name__)


async def with_retry(
    fn: Callable[[], Coroutine[Any, Any, _T]],
    max_attempts: int = 3,
    base_delay: float = 0.5,
) -> _T:
    """Retry an async callable with exponential backoff on OperationalError."""
    for attempt in range(max_attempts):
        try:
            return await fn()
        except OperationalError:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            log.warning(
                "postgres.retry",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay_seconds=delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover
