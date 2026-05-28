"""Typed governance operator request DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path
    from uuid import UUID


@dataclass(frozen=True)
class PerformanceSnapshotRequest:
    strategy_run_id: UUID
    as_of: datetime
    nav: Decimal
    gross_exposure: Decimal
    cash: Decimal
    source: str


@dataclass(frozen=True)
class PerformanceReportRequest:
    strategy_run_id: UUID
    as_of: datetime
    window: int


@dataclass(frozen=True)
class PerformanceHeartbeatRequest:
    component: str
    as_of: datetime
    status: str
    detail: str


PerformanceRequest: TypeAlias = (
    PerformanceSnapshotRequest | PerformanceReportRequest | PerformanceHeartbeatRequest
)


@dataclass(frozen=True)
class SignalGateRequest:
    command: str
    signal_name: str
    signal_type: str
    as_of: datetime
    daily_ic: float | None = None
    observations: int = 1
    drawdown: float = 0.0
    turnover: float = 0.0


@dataclass(frozen=True)
class TextGateRequest:
    command: str
    strategy_name: str
    as_of: datetime
    daily_ic: float | None = None
    observations: int = 1


@dataclass(frozen=True)
class ReadinessRequest:
    command: str
    profile: str
    as_of: datetime
    contracts_file: str | None
    component: str
    soak_report: Path | None
    backup_manifest: Path | None
    signal_name: str
    signal_type: str
    check_broker: bool


@dataclass(frozen=True)
class ProductionCandidateRequest:
    command: str
    profile: str
    as_of: datetime
    contracts_file: str | None
    component: str
    soak_report: Path | None
    backup_manifest: Path | None
    signal_sources: tuple[str, ...]
    primary_signal_name: str
    primary_signal_type: str
    campaign_max_age_days: int | None
    clean_live_days: int
    check_broker: bool


@dataclass(frozen=True)
class PaperSoakReportRequest:
    profile: str
    strategy_run_id: UUID
    as_of: datetime
    contracts_file: str | None
    signal_name: str
    signal_type: str
    bar_seconds: int
    data_health_window_days: int
    order_latency_window_days: int
    output: Path | None


@dataclass(frozen=True)
class DatasetQuorumRequest:
    command: str
    dataset_kind: str
    as_of: datetime
    vendor_bars: Path | None = None
    required_vendor_count: int = 2
    max_disagreement_bps: Decimal = Decimal("50")


@dataclass(frozen=True)
class SimulatorCalibrationRequest:
    as_of: datetime
    lookback_days: int
    floor_bps: float
    min_sample_count: int
    p90_safety_margin: float
    output: Path | None


__all__ = [
    "DatasetQuorumRequest",
    "PaperSoakReportRequest",
    "PerformanceHeartbeatRequest",
    "PerformanceReportRequest",
    "PerformanceRequest",
    "PerformanceSnapshotRequest",
    "ProductionCandidateRequest",
    "ReadinessRequest",
    "SignalGateRequest",
    "SimulatorCalibrationRequest",
    "TextGateRequest",
]
