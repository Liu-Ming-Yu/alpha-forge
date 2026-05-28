"""DTOs and lightweight protocols for operator read models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterator
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.domain.settlement import CashReservation


class CashLedgerViewPort(Protocol):
    @property
    def settled_cash(self) -> Decimal: ...

    @property
    def unsettled_cash(self) -> Decimal: ...

    @property
    def reserved_cash(self) -> Decimal: ...

    @property
    def available_cash(self) -> Decimal: ...

    def active_reservations(self) -> Iterator[CashReservation]: ...


class ThrottleStateViewPort(Protocol):
    _kill_switch_reason: str

    @property
    def kill_switch_active(self) -> bool: ...

    @property
    def total_submitted(self) -> int: ...

    @property
    def tokens_available(self) -> float: ...


@dataclass(frozen=True)
class CashStatusView:
    """Operator-facing settled/unsettled/reserved cash summary."""

    as_of: datetime
    settled_cash: Decimal
    unsettled_cash: Decimal
    reserved_cash: Decimal
    available_cash: Decimal


@dataclass(frozen=True)
class BrokerHealthView:
    """Operator-facing broker connection summary."""

    connected: bool
    kill_switch_active: bool
    kill_switch_reason: str
    orders_submitted_this_session: int
    throttle_tokens_available: float
    status: str = "unknown"
    detail: str = ""
    latency_ms: float | None = None
    last_heartbeat_at: datetime | None = None


@dataclass(frozen=True)
class BlotterEntry:
    """One row in the order blotter."""

    order_id: uuid.UUID
    instrument_id: uuid.UUID
    side: str
    quantity: int
    order_type: str
    fills_count: int
    total_filled: int
    avg_fill_price: Decimal | None = None
    vwap_at_submission: Decimal | None = None
    commission_paid: Decimal | None = None
    tif_remaining_seconds: int | None = None
    broker_status: str | None = None


@dataclass(frozen=True)
class BlotterView:
    """Operator-facing order blotter."""

    as_of: datetime
    entries: list[BlotterEntry] = field(default_factory=list)


@dataclass(frozen=True)
class PaperGateMetricsView:
    as_of: datetime
    orders_considered: int
    reject_rate: Decimal
    broker_error_rate: Decimal
    reconcile_discrepancies: int
    cash_drift_incidents: int
    stale_reservations: int
    average_fill_slippage_bps: Decimal | None
    fill_quality_summary: str


class StrategyHealth(StrEnum):
    """Lifecycle classification for a running strategy engine."""

    LAUNCHING = "launching"
    SCALING_UP = "scaling_up"
    STABLE = "stable"
    DEGRADED = "degraded"
    RETIRING = "retiring"
    RETIRED = "retired"


@dataclass(frozen=True)
class StrategyLifecycleView:
    """Engine-level lifecycle state and governance criteria."""

    engine_name: str
    engine_version: str
    health: StrategyHealth
    days_active: int
    rolling_sharpe_90d: float
    rolling_ic_60d: float
    max_drawdown_realized: float
    max_drawdown_limit: float
    slippage_ratio: float
    cycles_completed: int
    recommendation: str


@dataclass(frozen=True)
class RegimeStateView:
    """Current market regime classification for operator display."""

    as_of: datetime
    label: str
    gross_exposure_scale: float
    trend_z: float
    annualized_vol: float
    breadth_pct: float


@dataclass(frozen=True)
class SignalDecayView:
    """Signal quality analytics for monitoring alpha persistence."""

    as_of: datetime
    engine_name: str
    signals_generated: int
    mean_score: float
    score_dispersion: float
    top_quintile_count: int
    bottom_quintile_count: int
    turnover_rate: float


@dataclass(frozen=True)
class EngineBudgetView:
    """Operator-facing budget for one engine."""

    engine_name: str
    engine_version: str
    run_mode: str
    capital_weight: Decimal
    max_gross: Decimal
    max_turnover: Decimal
    enabled: bool


@dataclass(frozen=True)
class CombinedExposureView:
    """Account-level exposure summary across engine budgets."""

    as_of: datetime
    enabled_engines: int
    allocated_capital_weight: Decimal
    reserved_cash_weight: Decimal


@dataclass(frozen=True)
class OrderAllocationView:
    """Order attribution back to engine contributions."""

    order_id: uuid.UUID
    engine_name: str
    strategy_run_id: uuid.UUID
    instrument_id: uuid.UUID
    allocated_weight: Decimal
    allocated_notional: Decimal | None


@dataclass(frozen=True)
class SignalContributionView:
    """Operator-facing source contribution for one ensemble score."""

    score_id: uuid.UUID
    strategy_run_id: uuid.UUID
    instrument_id: uuid.UUID
    as_of: datetime
    source: str
    source_model_version: str
    raw_score: float
    normalized_score: float
    blend_weight: float
    confidence: float
    promotion_state: str


@dataclass(frozen=True)
class ForecastEvidenceView:
    """Operator-facing prediction evidence for one alpha source."""

    source: str
    model_version: str
    as_of: datetime
    horizon: str
    observations: int
    mean_confidence: float
    latest_prediction_at: datetime | None
    stale: bool
    passed: bool
    blockers: tuple[str, ...]
    calibration_buckets: tuple[str, ...]
