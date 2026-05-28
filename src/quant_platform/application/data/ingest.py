"""Data ingest use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.application.results import UseCaseResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import date
    from pathlib import Path


@dataclass(frozen=True)
class IngestRequest:
    start: date
    end: date
    contracts_file: Path
    bar_seconds: int = 86400
    source: str = "ib"
    """Bar source: ``ib`` (broker historical API) or ``vendor`` (Tiingo/Polygon).

    The vendor path skips the IB connection entirely, which is required for
    large historical backfills where IB pacing limits make per-instrument
    fetches infeasible.
    """


class IngestUseCase:
    def __init__(
        self,
        *,
        reporter: Callable[[IngestRequest], Awaitable[dict[str, object]]],
    ) -> None:
        self._reporter = reporter

    async def run(self, request: IngestRequest) -> UseCaseResult[dict[str, object]]:
        payload = await self._reporter(request)
        return UseCaseResult(payload=payload)


__all__ = ["IngestRequest", "IngestUseCase"]
