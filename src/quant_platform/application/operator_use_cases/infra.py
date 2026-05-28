"""Infrastructure operator use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

from quant_platform.application.results import ResultPresentation, UseCaseResult
from quant_platform.application.use_cases import CallableUseCase, UseCaseRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from quant_platform.application.operator.requests import NoInputRequest

log = structlog.get_logger(__name__)


class InfraUseCasePorts(Protocol):
    """Infrastructure adapters required by operator use cases."""

    def migrate(self, request: NoInputRequest) -> str: ...

    def migrations_check(self, request: NoInputRequest) -> str: ...

    def verify_schema(self, request: NoInputRequest) -> Awaitable[None]: ...


def register_infra_use_cases(registry: UseCaseRegistry, ports: InfraUseCasePorts) -> None:
    """Register migration/schema use cases."""

    def migrate(request: NoInputRequest) -> UseCaseResult[None]:
        head = ports.migrate(request)
        log.info("migrate.complete", head=head)
        return UseCaseResult()

    def migrations_check(request: NoInputRequest) -> UseCaseResult[str]:
        head = ports.migrations_check(request)
        return UseCaseResult(
            message=f"Packaged Alembic migration chain OK; head={head}",
            presentation=ResultPresentation.TEXT,
        )

    async def verify_schema(request: NoInputRequest) -> UseCaseResult[None]:
        await ports.verify_schema(request)
        return UseCaseResult()

    registry.register("infra.migrate", CallableUseCase(migrate))
    registry.register("infra.migrations_check", CallableUseCase(migrations_check))
    registry.register("infra.verify_schema", CallableUseCase(verify_schema))


__all__ = ["InfraUseCasePorts", "register_infra_use_cases"]
