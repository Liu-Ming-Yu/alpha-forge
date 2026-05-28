"""Pure data-provider circuit breaker helper."""

from __future__ import annotations

import time

import structlog

log = structlog.get_logger(__name__)


class DataCircuitBreaker:
    """Lightweight synchronous circuit breaker for external HTTP data providers."""

    def __init__(self, name: str, failure_threshold: int, open_seconds: float) -> None:
        self._name = name
        self._threshold = failure_threshold
        self._open_seconds = open_seconds
        self._failures = 0
        self._open_until: float = 0.0

    def is_open(self) -> bool:
        now = time.monotonic()
        if self._open_until and now < self._open_until:
            log.warning("data_circuit_breaker.rejected", name=self._name)
            return True
        if self._open_until:
            self._open_until = 0.0
        return False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold and not self._open_until:
            self._open_until = time.monotonic() + self._open_seconds
            log.error(
                "data_circuit_breaker.opened",
                name=self._name,
                failures=self._failures,
                open_seconds=self._open_seconds,
            )

    def record_success(self) -> None:
        if self._failures:
            log.info(
                "data_circuit_breaker.closed",
                name=self._name,
                previous_failures=self._failures,
            )
        self._failures = 0
        self._open_until = 0.0
