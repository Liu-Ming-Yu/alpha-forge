"""Execution-policy and multi-engine target domain models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime


@dataclass(frozen=True)
class ExecutionTacticPolicy:
    """Live/paper execution-tactic policy surface.

    The v1 implementation is configuration-only.  Order routers can consume
    this value to choose passive repricing and close-auction behavior without
    introducing new broker adapters.
    """

    passive_limit_enabled: bool = False
    reprice_interval_seconds: int = 300
    max_reprices_per_order: int = 3
    min_reprice_improvement_bps: float = 5.0
    adverse_drift_escalate_bps: float = 25.0
    close_auction_enabled: bool = False
    max_adv_participation_pct: float = 0.05
    order_timeout_seconds: int = 1800

    def __post_init__(self) -> None:
        if self.reprice_interval_seconds < 1:
            raise ValueError("reprice_interval_seconds must be >= 1")
        if self.max_reprices_per_order < 0:
            raise ValueError("max_reprices_per_order must be >= 0")
        if self.min_reprice_improvement_bps < 0:
            raise ValueError("min_reprice_improvement_bps must be >= 0")
        if self.adverse_drift_escalate_bps < 0:
            raise ValueError("adverse_drift_escalate_bps must be >= 0")
        if not (0 < self.max_adv_participation_pct <= 1):
            raise ValueError("max_adv_participation_pct must be in (0, 1]")
        if self.order_timeout_seconds < 1:
            raise ValueError("order_timeout_seconds must be >= 1")


@dataclass(frozen=True)
class EngineTargetProposal:
    """Proposal-only target emitted by an alpha engine.

    Engines do not submit orders in V2.  They publish these proposals and the
    account-level orchestrator is the only component that may merge, risk-check,
    and route resulting orders.
    """

    proposal_id: uuid.UUID
    engine_name: str
    engine_version: str
    run_mode: str
    strategy_run_id: uuid.UUID
    as_of: datetime
    weights: Mapping[uuid.UUID, Decimal]
    cash_target_weight: Decimal
    promotion_state: str = "paper"
    feature_dataset_id: uuid.UUID | None = None
    model_artifact_id: uuid.UUID | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if not self.engine_name.strip():
            raise ValueError("engine_name must not be empty")
        if not self.engine_version.strip():
            raise ValueError("engine_version must not be empty")
        for instrument_id, weight in self.weights.items():
            if not (Decimal("0") <= weight <= Decimal("1")):
                raise ValueError(f"weight for {instrument_id} must be in [0, 1]")
        if not (Decimal("0") <= self.cash_target_weight <= Decimal("1")):
            raise ValueError("cash_target_weight must be in [0, 1]")
        if sum(self.weights.values(), Decimal("0")) + self.cash_target_weight > Decimal("1.001"):
            raise ValueError("proposal exceeds 100% account allocation")


@dataclass(frozen=True)
class EngineBudget:
    """Capital and risk envelope for one strategy engine."""

    engine_name: str
    engine_version: str
    run_mode: str
    capital_weight: Decimal
    max_gross: Decimal
    max_turnover: Decimal
    enabled: bool = True

    def __post_init__(self) -> None:
        if not (Decimal("0") <= self.capital_weight <= Decimal("1")):
            raise ValueError("capital_weight must be in [0, 1]")
        if not (Decimal("0") <= self.max_gross <= Decimal("1")):
            raise ValueError("max_gross must be in [0, 1]")
        if self.max_turnover < Decimal("0"):
            raise ValueError("max_turnover must be >= 0")


@dataclass(frozen=True)
class EngineTargetContribution:
    """One engine's target before account-level merge."""

    contribution_id: uuid.UUID
    combined_target_id: uuid.UUID
    engine_name: str
    strategy_run_id: uuid.UUID
    as_of: datetime
    weights: Mapping[uuid.UUID, Decimal]
    capital_weight: Decimal

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if not (Decimal("0") <= self.capital_weight <= Decimal("1")):
            raise ValueError("capital_weight must be in [0, 1]")


@dataclass(frozen=True)
class CombinedPortfolioTarget:
    """Account-level target produced by merging engine contributions."""

    target_id: uuid.UUID
    as_of: datetime
    weights: Mapping[uuid.UUID, Decimal]
    cash_target_weight: Decimal
    contributions: tuple[EngineTargetContribution, ...] = ()
    construction_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        for instrument_id, weight in self.weights.items():
            if not (Decimal("0") <= weight <= Decimal("1")):
                raise ValueError(f"weight for {instrument_id} must be in [0, 1]")
        if not (Decimal("0") <= self.cash_target_weight <= Decimal("1")):
            raise ValueError("cash_target_weight must be in [0, 1]")
        if sum(self.weights.values(), Decimal("0")) + self.cash_target_weight > Decimal("1.001"):
            raise ValueError("combined target exceeds 100% account allocation")


@dataclass(frozen=True)
class OrderAllocation:
    """Attribution from a merged order back to an engine contribution."""

    allocation_id: uuid.UUID
    order_id: uuid.UUID
    engine_name: str
    strategy_run_id: uuid.UUID
    instrument_id: uuid.UUID
    allocated_weight: Decimal
    allocated_notional: Decimal | None = None

    def __post_init__(self) -> None:
        if self.allocated_weight < Decimal("0"):
            raise ValueError("allocated_weight must be >= 0")


@dataclass(frozen=True)
class SignalContribution:
    """Attribution row for one source inside an ensemble signal score."""

    contribution_id: uuid.UUID
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
    feature_vector_id: uuid.UUID | None
    promotion_state: str

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if not self.source:
            raise ValueError("source must not be empty")
        if not (-1.0 <= self.raw_score <= 1.0):
            raise ValueError("raw_score must be in [-1, 1]")
        if not (-1.0 <= self.normalized_score <= 1.0):
            raise ValueError("normalized_score must be in [-1, 1]")
        if not (0.0 <= self.blend_weight <= 1.0):
            raise ValueError("blend_weight must be in [0, 1]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0, 1]")
