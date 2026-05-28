"""Production readiness and data-health domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal


class ProductionProfile(StrEnum):
    """Deployment profile for production-readiness checks."""

    PAPER = "paper"
    LLM_LIVE_REHEARSAL = "llm_live_rehearsal"
    LIVE = "live"


class ReadinessState(StrEnum):
    """Operator-facing deployment readiness state."""

    READY = "ready"
    DEGRADED = "degraded"
    HALTED = "halted"


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight assertion result."""

    name: str
    passed: bool
    detail: str
    severity: str = "error"


@dataclass(frozen=True)
class PreflightReport:
    """Aggregated production preflight result."""

    profile: ProductionProfile
    generated_at: datetime
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed or check.severity != "error" for check in self.checks)

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(
            check for check in self.checks if not check.passed and check.severity == "error"
        )


@dataclass(frozen=True)
class DataHealthInstrumentStatus:
    """Data-health status for one active instrument."""

    instrument_id: uuid.UUID
    symbol: str
    bars_found: int
    latest_bar_at: datetime | None
    liquidity_profile_present: bool
    stale: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataHealthReport:
    """Universe-level data health report used as a cycle go/no-go gate."""

    generated_at: datetime
    start: datetime
    end: datetime
    instruments_checked: int
    instruments_with_bars: int
    instruments_with_liquidity: int
    stale_instruments: int
    statuses: tuple[DataHealthInstrumentStatus, ...]

    @property
    def coverage_pct(self) -> float:
        if self.instruments_checked == 0:
            return 0.0
        return self.instruments_with_bars / self.instruments_checked

    @property
    def liquidity_coverage_pct(self) -> float:
        if self.instruments_checked == 0:
            return 0.0
        return self.instruments_with_liquidity / self.instruments_checked

    @property
    def passed(self) -> bool:
        return (
            self.instruments_checked > 0
            and self.coverage_pct == 1.0
            and self.stale_instruments == 0
        )


@dataclass(frozen=True)
class RuntimeHeartbeat:
    """Persisted process heartbeat for deployment readiness evidence."""

    component: str
    as_of: datetime
    status: str
    detail: str = ""


@dataclass(frozen=True)
class BrokerHealthObservation:
    """Persisted broker health sample for operator readiness evidence."""

    observed_at: datetime
    status: str
    latency_ms: float
    last_heartbeat_at: datetime | None = None
    detail: str = ""


@dataclass(frozen=True)
class BrokerSmokeObservation:
    """Persisted read-only IB Gateway smoke result."""

    observed_at: datetime
    status: str
    host: str
    port: int
    client_id: int
    latency_ms: float
    account_status: str
    positions_status: str
    open_orders_status: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.status == "connected"
            and self.account_status == "ok"
            and self.positions_status == "ok"
            and self.open_orders_status == "ok"
        )


@dataclass(frozen=True)
class PaperLifecycleObservation:
    """Persisted safe paper order submit/open/cancel/reconcile result."""

    observed_at: datetime
    status: str
    host: str
    port: int
    client_id: int
    instrument_id: uuid.UUID
    broker_order_id: str
    max_notional_usd: Decimal
    limit_price: Decimal
    quantity: int
    ack_status: str
    cancel_status: str
    stale_open_order_count: int
    detail: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.status == "passed"
            and self.ack_status == "ok"
            and self.cancel_status == "ok"
            and self.stale_open_order_count == 0
        )


@dataclass(frozen=True)
class ProductionReadinessReport:
    """Aggregated promotion/readiness report."""

    profile: ProductionProfile
    generated_at: datetime
    state: ReadinessState
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return self.state == ReadinessState.READY

    @property
    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(
            check for check in self.checks if not check.passed and check.severity == "error"
        )


@dataclass(frozen=True)
class ReadinessSnapshot:
    """Persisted readiness report snapshot."""

    snapshot_id: uuid.UUID
    profile: ProductionProfile
    generated_at: datetime
    state: ReadinessState
    passed: bool
    checks: tuple[PreflightCheck, ...]

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
