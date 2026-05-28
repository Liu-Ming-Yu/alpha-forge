"""VectorBT trade accounting result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..artifacts.backtest_artifacts import (
        BacktestFillArtifact,
    )


@dataclass(slots=True)
class CycleTradeResult:
    cash: Decimal
    commission: Decimal = Decimal("0")
    slippage_bps: float = 0.0
    fills_count: int = 0
    participation: list[float] = field(default_factory=list)
    fill_artifacts: list[BacktestFillArtifact] = field(default_factory=list)

    def add_fill(
        self,
        fill: BacktestFillArtifact,
        *,
        commission: Decimal,
        slippage_bps: float,
        participation_pct: float,
    ) -> None:
        self.commission += commission
        self.slippage_bps += slippage_bps
        self.fills_count += 1
        self.participation.append(participation_pct)
        self.fill_artifacts.append(fill)


__all__ = ["CycleTradeResult"]
