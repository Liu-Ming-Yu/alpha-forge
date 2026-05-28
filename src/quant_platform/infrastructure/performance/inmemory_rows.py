"""Collection primitives for in-memory performance adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable


class Sortable(Protocol):
    """Value that can be used as a list sort key."""

    def __lt__(self, other: Self, /) -> bool: ...


T = TypeVar("T")


def append_sorted(
    rows: list[T],
    item: T,
    *,
    sort_key: Callable[[T], Sortable],
) -> None:
    """Append one row and keep the collection sorted."""
    rows.append(item)
    rows.sort(key=sort_key)


def append_if_missing_sorted(
    rows: list[T],
    item: T,
    *,
    identity: Callable[[T], object],
    sort_key: Callable[[T], Sortable],
) -> None:
    """Append one row unless another row with the same identity exists."""
    item_identity = identity(item)
    if any(identity(row) == item_identity for row in rows):
        return
    append_sorted(rows, item, sort_key=sort_key)


def upsert_sorted(
    rows: list[T],
    item: T,
    *,
    identity: Callable[[T], object],
    sort_key: Callable[[T], Sortable],
) -> None:
    """Replace matching rows, append the new row, and keep the collection sorted."""
    item_identity = identity(item)
    rows[:] = [row for row in rows if identity(row) != item_identity]
    append_sorted(rows, item, sort_key=sort_key)


def latest(rows: list[T]) -> T | None:
    """Return the final sorted row, when present."""
    if not rows:
        return None
    return rows[-1]
