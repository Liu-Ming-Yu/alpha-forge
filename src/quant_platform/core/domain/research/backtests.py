"""Backtest request, evidence, and reconciliation domain models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


class BacktestReconciliationStatus(StrEnum):
    """Outcome of reconciling canonical event replay against vectorized replay."""

    PASSED = "passed"
    FAILED = "failed"
    NON_COMPARABLE = "non_comparable"


@dataclass(frozen=True)
class IntradayBacktestSpec:
    """Production request for one canonical intraday backtest replay."""

    strategy_name: str
    strategy_version: str
    start: datetime
    end: datetime
    initial_capital: Decimal
    decision_times: tuple[datetime, ...]
    dataset_ids: tuple[uuid.UUID, ...]
    universe_name: str
    feature_set_version: str
    model_version: str
    execution_profile: str = "ibkr_cash_intraday_v1"
    benchmark_instrument_id: uuid.UUID | None = None

    def __post_init__(self) -> None:
        if not self.strategy_name.strip():
            raise ValueError("strategy_name must not be empty")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        for name, value in (("start", self.start), ("end", self.end)):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if not self.decision_times:
            raise ValueError("decision_times must not be empty")
        if any(ts.tzinfo is None for ts in self.decision_times):
            raise ValueError("decision_times must be timezone-aware")
        if any(ts < self.start or ts > self.end for ts in self.decision_times):
            raise ValueError("decision_times must be inside [start, end]")
        if not self.dataset_ids:
            raise ValueError("dataset_ids must not be empty")
        if len(set(self.dataset_ids)) != len(self.dataset_ids):
            raise ValueError("dataset_ids must be unique")


@dataclass(frozen=True)
class BacktestReconciliationReport:
    """Fail-closed parity report between event-driven and vectorized backtests."""

    report_id: uuid.UUID
    strategy_run_id: uuid.UUID
    generated_at: datetime
    status: BacktestReconciliationStatus
    passed: bool
    comparable: bool
    target_weight_max_diff_bps: Decimal
    nav_diff_bps: Decimal
    max_drawdown_diff_bps: Decimal
    tolerance_target_weight_bps: Decimal = Decimal("1")
    tolerance_nav_bps: Decimal = Decimal("50")
    tolerance_drawdown_bps: Decimal = Decimal("50")
    residual_order_count: int = 0
    missing_artifacts: tuple[str, ...] = ()
    breaches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        expected_passed = self.status == BacktestReconciliationStatus.PASSED
        if self.passed != expected_passed:
            raise ValueError("passed must match status")
        if self.status == BacktestReconciliationStatus.NON_COMPARABLE and self.comparable:
            raise ValueError("NON_COMPARABLE reports must set comparable=False")


@dataclass(frozen=True)
class BacktestEvidenceManifest:
    """Immutable evidence bundle consumed by tearsheets and promotion gates."""

    manifest_id: uuid.UUID
    strategy_run_id: uuid.UUID
    created_at: datetime
    spec: IntradayBacktestSpec
    code_commit: str
    config_hash: str
    dataset_ids: tuple[uuid.UUID, ...]
    universe_snapshot_id: uuid.UUID
    feature_dataset_id: uuid.UUID | None
    model_artifact_id: uuid.UUID | None
    calibration_artifact_uri: str
    execution_quality_uri: str
    reconciliation_report_uri: str
    event_driven_artifact_uri: str
    vectorized_artifact_uri: str
    passed: bool
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not self.code_commit.strip():
            raise ValueError("code_commit must not be empty")
        if not self.config_hash.strip():
            raise ValueError("config_hash must not be empty")
        if len(set(self.dataset_ids)) != len(self.dataset_ids):
            raise ValueError("dataset_ids must be unique")
        if self.passed and self.blockers:
            raise ValueError("passed manifests cannot have blockers")


@dataclass(frozen=True)
class BacktestRun:
    """Metadata and summary statistics for a completed backtest.

    This is a lightweight summary record.  Full trade-by-trade results are
    stored as a Parquet artifact in object storage and referenced via
    artifact_uri.

    Args:
        backtest_id: Stable system UUID.
        strategy_run_id: FK to StrategyRun (run_type must be BACKTEST).
        universe_snapshot_id: UUID of the instrument universe snapshot used.
        start_date: First date in the backtest simulation window.
        end_date: Last date in the backtest simulation window.
        initial_capital: Simulated starting capital in USD.
        final_capital: Simulated ending capital in USD.
        total_return: (final_capital / initial_capital) - 1.
        annualised_sharpe: Annualised Sharpe ratio over the full period.
            ``None`` when fewer than 20 return observations are available —
            a sub-20 sample is statistically meaningless and a hard zero
            silently understates risk-adjusted performance.
        max_drawdown: Maximum peak-to-trough drawdown as a negative fraction.
        artifact_uri: URI to the Parquet file containing per-trade results.
        created_at: UTC timestamp of backtest completion.
    """

    backtest_id: uuid.UUID
    strategy_run_id: uuid.UUID
    universe_snapshot_id: uuid.UUID
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: Decimal
    annualised_sharpe: Decimal | None
    max_drawdown: Decimal
    artifact_uri: str
    created_at: datetime

    def __post_init__(self) -> None:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.max_drawdown > Decimal("0"):
            raise ValueError("max_drawdown must be <= 0")
