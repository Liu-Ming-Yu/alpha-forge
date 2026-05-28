"""Runtime helpers for engine model registry operations."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeVar, cast, overload

from quant_platform.core.exceptions import DataStalenessError

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Awaitable


@overload
async def maybe_await(value: Awaitable[T]) -> T: ...


@overload
async def maybe_await(value: T) -> T: ...


async def maybe_await(value: T | Awaitable[T]) -> T:
    """Accept either a sync value or awaitable and return the result."""
    if inspect.isawaitable(value):
        return await cast("Awaitable[T]", value)
    return value


async def check_model_staleness(
    model: object,
    *,
    as_of: datetime,
    max_age_hours: float,
) -> None:
    """Raise DataStalenessError when a registered model is too old."""
    if max_age_hours <= 0:
        return
    created_at = getattr(model, "created_at", None)
    if created_at is None:
        return
    if getattr(created_at, "tzinfo", None) is None:
        created_at = created_at.replace(tzinfo=UTC)
    age_hours = (as_of - created_at).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        strategy_name = getattr(model, "strategy_name", "unknown")
        model_version = getattr(model, "model_version", "unknown")
        raise DataStalenessError(
            f"registered model is stale: strategy={strategy_name!r} "
            f"version={model_version!r} age={age_hours:.1f}h "
            f"exceeds max_model_age_hours={max_age_hours}"
        )
