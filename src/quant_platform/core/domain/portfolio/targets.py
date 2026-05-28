"""Portfolio target domain model."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping
    from datetime import datetime


@dataclass(frozen=True)
class PortfolioTarget:
    """Desired portfolio state expressed as target weights.

    Produced by PortfolioConstructor after applying regime, risk, and
    settlement constraints.  Consumed by the execution service to determine
    which orders to submit.

    Args:
        target_id: Stable system UUID.
        strategy_run_id: FK to StrategyRun.
        as_of: UTC timestamp of target construction.
        regime_id: FK to RegimeState used in construction.
        weights: Mapping of instrument_id → target weight in [0.0, 1.0].
            Instruments not in the mapping are implicitly targeted at 0.0
            (i.e. should be exited if currently held).
        cash_target_weight: The target fraction to hold as settled cash.
            cash_target_weight + sum(weights.values()) must be <= 1.0 (with
            a small tolerance for Decimal arithmetic).  The sum may be less
            than 1.0 when the constructor intentionally under-invests.
        construction_notes: Human-readable explanation of material constraints
            that were binding during construction (e.g. "AAPL excluded: no
            settled cash").

    Must never contain:
        Share counts, dollar amounts, or broker-specific order parameters.
        Those are derived by the execution service using the current
        AccountSnapshot.
    """

    target_id: uuid.UUID
    strategy_run_id: uuid.UUID
    as_of: datetime
    regime_id: uuid.UUID
    weights: Mapping[uuid.UUID, Decimal]
    cash_target_weight: Decimal
    # ``Iterable[str]`` accepts both ``list[str]`` and ``tuple[str, ...]`` from
    # producers; ``__post_init__`` normalises to an immutable tuple so the
    # frozen dataclass invariant cannot be defeated by an out-of-band caller
    # mutating the originally-passed list.
    construction_notes: tuple[str, ...] | list[str] | Iterable[str] = ()

    def __post_init__(self) -> None:
        # Freeze ``construction_notes`` into a tuple regardless of how it
        # was passed in.
        if not isinstance(self.construction_notes, tuple):
            object.__setattr__(self, "construction_notes", tuple(self.construction_notes))
        for instrument_id, w in self.weights.items():
            if not (Decimal("0") <= w <= Decimal("1")):
                raise ValueError(f"weight for {instrument_id} must be in [0, 1], got {w}")
        if not (Decimal("0") <= self.cash_target_weight <= Decimal("1")):
            raise ValueError(f"cash_target_weight must be in [0, 1], got {self.cash_target_weight}")
        invested_total = sum(self.weights.values(), Decimal("0"))
        grand_total = invested_total + self.cash_target_weight
        # Allow a small tolerance to accommodate Decimal arithmetic rounding.
        tolerance = Decimal("0.001")
        if grand_total > Decimal("1") + tolerance:
            raise ValueError(
                f"cash_target_weight ({self.cash_target_weight}) + sum(weights) "
                f"({invested_total}) = {grand_total} exceeds 1.0"
            )
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
