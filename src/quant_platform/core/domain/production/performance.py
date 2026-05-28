"""Performance and signal-gate production domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.domain.production.readiness import ReadinessState

if TYPE_CHECKING:
    import uuid
    from datetime import date, datetime


@dataclass(frozen=True)
class NavSnapshot:
    """Persisted strategy NAV point used for live/paper performance governance."""

    snapshot_id: uuid.UUID
    strategy_run_id: uuid.UUID
    as_of: datetime
    net_asset_value: Decimal
    gross_exposure: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    source: str = "runtime"
    realized_pnl: Decimal | None = None
    unrealized_pnl: Decimal | None = None


@dataclass(frozen=True)
class InstrumentPnl:
    """Per-instrument P&L attribution for one strategy cycle.

    Args:
        pnl_id: Stable system UUID.
        strategy_run_id: FK to StrategyRun.
        instrument_id: FK to Instrument.
        as_of: UTC timestamp of the attribution cycle.
        realized_pnl: P&L from closed lots.
        unrealized_pnl: Open position mark-to-market P&L.
        weight: Portfolio weight at cycle close.
        contribution: Contribution to total portfolio P&L (weight × return).
    """

    pnl_id: uuid.UUID
    strategy_run_id: uuid.UUID
    instrument_id: uuid.UUID
    as_of: datetime
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    weight: Decimal
    contribution: Decimal


@dataclass(frozen=True)
class PerformanceReport:
    """Rolling performance state for operator lifecycle decisions."""

    strategy_run_id: uuid.UUID
    as_of: datetime
    observations: int
    rolling_sharpe: float
    max_drawdown: float
    gross_turnover: float
    rolling_ic: float
    slippage_ratio: float


@dataclass(frozen=True)
class MetricRollupSnapshot:
    """Durable metric rollup used to survive Prometheus process restarts."""

    snapshot_id: uuid.UUID
    metric_name: str
    as_of: datetime
    window: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    source: str = "runtime"


@dataclass(frozen=True)
class TextSignalGateRecord:
    """One daily text-signal IC observation."""

    strategy_name: str
    as_of: datetime
    daily_ic: float
    observations: int = 1
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TextSignalGateStatus:
    """Persisted text-signal promotion gate status."""

    strategy_name: str
    as_of: datetime
    observations: int
    rolling_ic: float
    negative_streak: int
    min_observations: int
    min_ic: float
    max_negative_streak: int

    @property
    def passed(self) -> bool:
        return (
            self.observations >= self.min_observations
            and self.rolling_ic >= self.min_ic
            and self.negative_streak < self.max_negative_streak
        )


@dataclass(frozen=True)
class SignalGateRecord:
    """One daily promotion observation for any signal family."""

    signal_name: str
    signal_type: str
    as_of: datetime
    daily_ic: float
    observations: int = 1
    drawdown: float = 0.0
    turnover: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalGateStatus:
    """Generic signal-promotion gate status."""

    signal_name: str
    signal_type: str
    as_of: datetime
    observations: int
    rolling_ic: float
    negative_streak: int
    max_drawdown: float
    max_turnover: float
    min_observations: int
    min_ic: float
    max_negative_streak: int
    drawdown_limit: float
    turnover_limit: float

    @property
    def passed(self) -> bool:
        return (
            self.observations >= self.min_observations
            and self.rolling_ic >= self.min_ic
            and self.negative_streak < self.max_negative_streak
            and self.max_drawdown >= self.drawdown_limit
            and self.max_turnover <= self.turnover_limit
        )

    @property
    def state(self) -> ReadinessState:
        if self.passed:
            return ReadinessState.READY
        if self.observations == 0 or self.negative_streak >= self.max_negative_streak:
            return ReadinessState.HALTED
        return ReadinessState.DEGRADED


@dataclass(frozen=True)
class ShadowPaperParityRecord:
    """One shadow-vs-paper parity observation for promoted live sources."""

    parity_id: uuid.UUID
    signal_name: str
    signal_type: str
    trading_day: date
    as_of: datetime
    instruments_compared: int
    missing_instruments: int
    max_target_weight_diff_bps: float
    order_side_mismatches: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ShadowPaperParityStatus:
    """Aggregated shadow-vs-paper parity status for live promotion."""

    signal_name: str
    signal_type: str
    as_of: datetime
    observations: int
    trading_days: int
    min_trading_days: int
    max_target_weight_diff_bps: float
    max_allowed_target_weight_diff_bps: float
    missing_instruments: int
    order_side_mismatches: int

    @property
    def passed(self) -> bool:
        return (
            self.trading_days >= self.min_trading_days
            and self.missing_instruments == 0
            and self.order_side_mismatches == 0
            and self.max_target_weight_diff_bps <= self.max_allowed_target_weight_diff_bps
        )

    @property
    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.trading_days < self.min_trading_days:
            blockers.append(
                f"trading_days {self.trading_days} below minimum {self.min_trading_days}"
            )
        if self.missing_instruments:
            blockers.append(f"missing_instruments={self.missing_instruments}")
        if self.order_side_mismatches:
            blockers.append(f"order_side_mismatches={self.order_side_mismatches}")
        if self.max_target_weight_diff_bps > self.max_allowed_target_weight_diff_bps:
            blockers.append(
                "max_target_weight_diff_bps "
                f"{self.max_target_weight_diff_bps:.4f} exceeds "
                f"{self.max_allowed_target_weight_diff_bps:.4f}"
            )
        return tuple(blockers)
