"""Application result contracts shared by operator use cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

PayloadT = TypeVar("PayloadT", covariant=True)


class UseCaseStatus(StrEnum):
    """Contracted application-result status for operator use cases."""

    OK = "ok"
    BLOCKED = "blocked"
    FAILED = "failed"


class ResultPresentation(StrEnum):
    """How an interface adapter should render an application result."""

    NONE = "none"
    JSON = "json"
    KEY_VALUE = "key_value"
    TEXT = "text"


@dataclass(frozen=True)
class UseCaseResult(Generic[PayloadT]):
    """Typed application result returned by use cases."""

    status: UseCaseStatus = UseCaseStatus.OK
    payload: PayloadT | None = None
    message: str = ""
    exit_code: int = 0
    presentation: ResultPresentation = ResultPresentation.NONE
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise and reject uncontracted application statuses at construction time."""
        try:
            normalized = UseCaseStatus(self.status)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in UseCaseStatus)
            raise ValueError(
                f"invalid UseCaseResult.status {self.status!r}; expected {allowed}"
            ) from exc
        object.__setattr__(self, "status", normalized)

    @property
    def passed(self) -> bool:
        """Return whether this result should be considered operator-successful."""
        return self.exit_code == 0 and self.status is UseCaseStatus.OK


__all__ = ["ResultPresentation", "UseCaseResult", "UseCaseStatus"]
