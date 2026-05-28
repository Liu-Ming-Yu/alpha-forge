"""Portfolio risk-limit and optimizer domain models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime

    from quant_platform.core.domain.portfolio.targets import PortfolioTarget


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk constraints applied during portfolio construction and order approval.

    These limits are loaded from configuration at run start and are immutable
    for the duration of the session.  Any change requires a deliberate operator
    action and creates a new RiskLimits record in the audit trail.

    Args:
        limits_id: Stable system UUID.
        strategy_run_id: FK to StrategyRun these limits apply to.
        effective_from: UTC timestamp from which these limits are active.
        max_single_name_weight: Maximum fraction of NAV in any single name.
        max_sector_weight: Maximum fraction of NAV in any single GICS sector.
        max_gross_exposure: Maximum total long exposure as fraction of NAV
            (must be <= 1.0 for a cash account).
        max_daily_turnover: Maximum fraction of NAV that can be traded in
            a single day.
        min_cash_buffer: Minimum settled cash fraction to maintain at all times.
        max_drawdown_halt: Drawdown fraction at which the strategy must halt
            and await operator action (e.g. -0.15 = -15%).
        vol_target_annualised: Target annualised portfolio volatility.
            None means no vol targeting is applied.
    """

    limits_id: uuid.UUID
    strategy_run_id: uuid.UUID
    effective_from: datetime
    max_single_name_weight: Decimal
    max_sector_weight: Decimal
    max_gross_exposure: Decimal
    max_daily_turnover: Decimal
    min_cash_buffer: Decimal
    max_drawdown_halt: Decimal
    vol_target_annualised: Decimal | None = None
    cvar_tail_pct: float = 0.05  # stress-CVaR tail fraction for optimizer halt gate
    covariance_decay_halflife_days: int = 0  # 0 = no decay; >0 applies exponential decay

    def __post_init__(self) -> None:
        if self.max_gross_exposure > Decimal("1"):
            raise ValueError("max_gross_exposure must be <= 1.0 for a cash account")
        if self.max_drawdown_halt > Decimal("0"):
            raise ValueError("max_drawdown_halt must be <= 0 (it is a loss fraction)")
        if self.max_single_name_weight > self.max_sector_weight:
            raise ValueError("max_single_name_weight must be <= max_sector_weight")


@dataclass(frozen=True)
class CapitalBudget:
    """Account-level capital budget for one engine or sleeve."""

    budget_id: uuid.UUID
    engine_name: str
    as_of: datetime
    capital_weight: Decimal
    max_gross: Decimal
    max_turnover: Decimal
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if not self.engine_name.strip():
            raise ValueError("engine_name must not be empty")
        for name, value in (
            ("capital_weight", self.capital_weight),
            ("max_gross", self.max_gross),
            ("max_turnover", self.max_turnover),
        ):
            if value < Decimal("0"):
                raise ValueError(f"{name} must be >= 0")
        if self.capital_weight > Decimal("1") or self.max_gross > Decimal("1"):
            raise ValueError("capital_weight and max_gross must be <= 1")
        if self.max_gross > self.capital_weight:
            raise ValueError("max_gross must be <= capital_weight")


@dataclass(frozen=True)
class StressScenario:
    """Named portfolio stress scenario."""

    scenario_id: uuid.UUID
    name: str
    shocks: Mapping[uuid.UUID | str, Decimal]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")


@dataclass(frozen=True)
class PortfolioRiskModel:
    """Covariance/factor/stress model used by the production optimizer."""

    model_id: uuid.UUID
    as_of: datetime
    covariance: Mapping[tuple[uuid.UUID, uuid.UUID], Decimal]
    factor_exposures: Mapping[uuid.UUID, Mapping[str, Decimal]]
    scenarios: tuple[StressScenario, ...] = ()
    dataset_id: uuid.UUID | None = None
    schema_hash: str = ""

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        for (left, right), value in self.covariance.items():
            if value < Decimal("0") and left == right:
                raise ValueError("variance terms must be >= 0")
            if (right, left) in self.covariance and self.covariance[(right, left)] != value:
                raise ValueError("covariance matrix must be symmetric")


@dataclass(frozen=True)
class RiskSnapshot:
    """Persisted account-level risk state after optimization/pretrade."""

    snapshot_id: uuid.UUID
    strategy_run_id: uuid.UUID
    as_of: datetime
    gross_exposure: Decimal
    net_exposure: Decimal
    cvar: Decimal | None
    factor_exposures: Mapping[str, Decimal]
    stress_results: Mapping[str, Decimal]
    passed: bool

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if self.gross_exposure < Decimal("0"):
            raise ValueError("gross_exposure must be >= 0")


@dataclass(frozen=True)
class OptimizerResult:
    """Result from the central portfolio optimizer."""

    target: PortfolioTarget
    risk_snapshot: RiskSnapshot
    binding_constraints: tuple[str, ...] = ()
