"""Data maintenance use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.application.results import UseCaseResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import date
    from pathlib import Path


@dataclass(frozen=True)
class MaintainDataRequest:
    interval_seconds: float
    backfill_start: date | None = None
    backfill_end: date | None = None
    contracts_file: Path | None = None


class MaintainDataUseCase:
    def __init__(
        self,
        *,
        runner: Callable[[MaintainDataRequest], Awaitable[None]],
    ) -> None:
        self._runner = runner

    async def run(self, request: MaintainDataRequest) -> UseCaseResult[dict[str, object]]:
        await self._runner(request)
        return UseCaseResult(payload={"passed": True})


__all__ = ["MaintainDataRequest", "MaintainDataUseCase"]
