"""Feature computation use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.application.results import UseCaseResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


@dataclass(frozen=True)
class ComputeFeaturesRequest:
    contracts_file: Path | None = None


class ComputeFeaturesUseCase:
    def __init__(
        self,
        *,
        runner: Callable[[ComputeFeaturesRequest], Awaitable[None]],
    ) -> None:
        self._runner = runner

    async def run(self, request: ComputeFeaturesRequest) -> UseCaseResult[dict[str, object]]:
        await self._runner(request)
        return UseCaseResult(payload={"passed": True})


__all__ = ["ComputeFeaturesRequest", "ComputeFeaturesUseCase"]
