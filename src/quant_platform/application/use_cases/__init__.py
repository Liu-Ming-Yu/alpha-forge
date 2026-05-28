"""Small application use-case registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from quant_platform.application.results import UseCaseResult

RequestT = TypeVar("RequestT", contravariant=True)
PayloadT = TypeVar("PayloadT", covariant=True)


class UseCase(Protocol[RequestT, PayloadT]):
    """Application use case interface."""

    def run(
        self,
        request: RequestT,
    ) -> UseCaseResult[PayloadT] | Awaitable[UseCaseResult[PayloadT]]:
        """Execute a request and return a typed result."""


@dataclass(frozen=True)
class CallableUseCase(Generic[RequestT, PayloadT]):
    """Adapter for function-backed use cases."""

    handler: Callable[[RequestT], UseCaseResult[PayloadT] | Awaitable[UseCaseResult[PayloadT]]]

    def run(
        self,
        request: RequestT,
    ) -> UseCaseResult[PayloadT] | Awaitable[UseCaseResult[PayloadT]]:
        return self.handler(request)


class UseCaseRegistry:
    """Lookup table for named application use cases."""

    def __init__(self) -> None:
        self._use_cases: dict[str, UseCase[Any, Any]] = {}

    def register(self, name: str, use_case: UseCase[Any, Any]) -> None:
        """Register one named use case."""
        if not name:
            raise ValueError("use case name must not be empty")
        if name in self._use_cases:
            raise ValueError(f"use case {name!r} is already registered")
        self._use_cases[name] = use_case

    def names(self) -> tuple[str, ...]:
        """Return registered use-case names."""
        return tuple(sorted(self._use_cases))

    async def run(self, name: str, request: object) -> UseCaseResult[Any]:
        """Run a named use case."""
        try:
            use_case = self._use_cases[name]
        except KeyError as exc:
            valid = ", ".join(self.names())
            raise RuntimeError(f"unknown use case {name!r}; valid use cases: {valid}") from exc
        result = use_case.run(request)
        if hasattr(result, "__await__"):
            result = await result
        return result


__all__ = ["CallableUseCase", "UseCase", "UseCaseRegistry"]
