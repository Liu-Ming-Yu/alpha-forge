"""Corporate-action reprocessing use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.application.results import UseCaseResult

if TYPE_CHECKING:
    import uuid
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True)
class ReprocessCorporateActionsRequest:
    instrument_id: uuid.UUID


class ReprocessCorporateActionsUseCase:
    def __init__(
        self,
        *,
        runner: Callable[[ReprocessCorporateActionsRequest], Awaitable[None]],
    ) -> None:
        self._runner = runner

    async def run(
        self,
        request: ReprocessCorporateActionsRequest,
    ) -> UseCaseResult[dict[str, object]]:
        await self._runner(request)
        return UseCaseResult(payload={"passed": True})


__all__ = ["ReprocessCorporateActionsRequest", "ReprocessCorporateActionsUseCase"]
