"""Mutable replay state for the vectorized intraday comparator."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.services.research_service.intraday.backtesting.helpers import (
    account_snapshot,
    advance_settlements,
    apply_fill,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import date, datetime

    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.services.research_service.intraday.backtesting.types import (
        IntradayFillArtifact,
    )


@dataclass
class VectorizedIntradayReplayState:
    """Cash, position, and evidence state for vectorized intraday replay."""

    settled_cash: Decimal
    unsettled_cash: Decimal = Decimal("0")
    positions: dict[uuid.UUID, int] = field(default_factory=dict)
    avg_cost: dict[uuid.UUID, Decimal] = field(default_factory=dict)
    settlements: list[tuple[date, Decimal]] = field(default_factory=list)
    fills: list[IntradayFillArtifact] = field(default_factory=list)
    target_weights: dict[datetime, Mapping[uuid.UUID, Decimal]] = field(default_factory=dict)
    eligible: dict[datetime, tuple[uuid.UUID, ...]] = field(default_factory=dict)
    nav_curve: list[tuple[datetime, Decimal]] = field(default_factory=list)

    def advance_to(self, as_of: datetime) -> None:
        self.settled_cash, self.unsettled_cash, self.settlements = advance_settlements(
            as_of.date(),
            self.settled_cash,
            self.unsettled_cash,
            self.settlements,
        )

    def account(
        self,
        *,
        as_of: datetime,
        minute_bars: Mapping[uuid.UUID, list[MarketBar]],
    ) -> AccountSnapshot:
        return account_snapshot(
            as_of=as_of,
            settled_cash=self.settled_cash,
            unsettled_cash=self.unsettled_cash,
            positions=self.positions,
            avg_cost=self.avg_cost,
            minute_bars=minute_bars,
        )

    def apply_fill(self, fill: IntradayFillArtifact) -> None:
        self.settled_cash, self.unsettled_cash, self.settlements = apply_fill(
            fill,
            self.settled_cash,
            self.unsettled_cash,
            self.settlements,
            self.positions,
            self.avg_cost,
        )
        self.fills.append(fill)

    def record_nav(self, as_of: datetime, nav: Decimal) -> None:
        self.nav_curve.append((as_of, nav))


__all__ = ["VectorizedIntradayReplayState"]
