"""Execution-service circuit breaker helper."""

from __future__ import annotations

import asyncio
import time
from enum import Enum, auto
from typing import TYPE_CHECKING, TypeVar

import structlog

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = structlog.get_logger(__name__)

_T = TypeVar("_T")


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Async circuit breaker for broker network calls."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        open_seconds: float = 60.0,
        name: str = "broker",
    ) -> None:
        self._threshold = failure_threshold
        self._open_seconds = open_seconds
        self._name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._state_lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and (
            self._opened_at is not None and time.monotonic() - self._opened_at >= self._open_seconds
        ):
            self._state = CircuitState.HALF_OPEN
            log.info("circuit_breaker.half_open", name=self._name)
        return self._state

    async def call(self, fn: Callable[[], Awaitable[_T]]) -> _T:
        from quant_platform.core.exceptions import (
            BrokerAckTimeoutError,
            BrokerSubmissionError,
            BrokerUnavailableError,
        )

        async with self._state_lock:
            current_state = self.state
            if current_state == CircuitState.OPEN:
                log.warning("circuit_breaker.rejected", name=self._name)
                raise BrokerUnavailableError(
                    f"circuit breaker OPEN for {self._name!r} - "
                    "refusing call until cooldown expires"
                )

        try:
            result = await fn()
        except BrokerUnavailableError:
            async with self._state_lock:
                self._on_failure()
            raise
        except (BrokerSubmissionError, BrokerAckTimeoutError):
            raise
        else:
            async with self._state_lock:
                self._on_success()
            return result

    def _on_success(self) -> None:
        if self._state != CircuitState.CLOSED:
            log.info(
                "circuit_breaker.closed",
                name=self._name,
                previous_failures=self._failure_count,
            )
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        self._failure_count += 1
        log.warning(
            "circuit_breaker.failure",
            name=self._name,
            failure_count=self._failure_count,
            threshold=self._threshold,
        )
        if self._state == CircuitState.HALF_OPEN or self._failure_count >= self._threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            log.error(
                "circuit_breaker.opened",
                name=self._name,
                failure_count=self._failure_count,
                open_seconds=self._open_seconds,
            )
