"""Data-health application use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.application.results import UseCaseResult, UseCaseStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import date
    from pathlib import Path


@dataclass(frozen=True)
class DataHealthRequest:
    contracts_file: Path
    start: date
    end: date
    bar_seconds: int


class DataHealthUseCase:
    """Build the operator data-health payload through injected data ports."""

    def __init__(
        self,
        *,
        reporter: Callable[[DataHealthRequest], Awaitable[tuple[dict[str, object], bool]]],
    ) -> None:
        self._reporter = reporter

    async def run(self, request: DataHealthRequest) -> UseCaseResult[dict[str, object]]:
        payload, passed = await self._reporter(request)
        return UseCaseResult(
            status=UseCaseStatus.OK if passed else UseCaseStatus.BLOCKED,
            payload=payload,
            exit_code=0 if passed else 2,
        )


__all__ = ["DataHealthRequest", "DataHealthUseCase"]
