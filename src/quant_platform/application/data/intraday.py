"""Intraday data operator use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.application.results import UseCaseResult, UseCaseStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True)
class IntradayDataRequest:
    command: str
    args: object


class IntradayDataUseCase:
    def __init__(
        self,
        *,
        reporter: Callable[[IntradayDataRequest], Awaitable[tuple[dict[str, object], bool]]],
    ) -> None:
        self._reporter = reporter

    async def run(self, request: IntradayDataRequest) -> UseCaseResult[dict[str, object]]:
        payload, passed = await self._reporter(request)
        return UseCaseResult(
            status=UseCaseStatus.OK if passed else UseCaseStatus.BLOCKED,
            payload=payload,
            exit_code=0 if passed else 2,
        )


__all__ = ["IntradayDataRequest", "IntradayDataUseCase"]
