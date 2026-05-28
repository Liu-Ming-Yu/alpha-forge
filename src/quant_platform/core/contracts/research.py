"""Research contracts: backtest engine and signal scoring model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Sequence
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.algorithms.simulated_execution import ParticipationFillModel
    from quant_platform.core.domain.orders import FillEvent, OrderIntent
    from quant_platform.core.domain.portfolio import PortfolioTarget
    from quant_platform.core.domain.portfolio.positions import AccountSnapshot
    from quant_platform.core.domain.research import (
        AlphaReadinessReport,
        BacktestRun,
        FeatureAuditResult,
        FeatureVector,
        ModelArtifact,
        ModelCard,
        StrategyRun,
    )
    from quant_platform.core.domain.signals import SignalScore


@runtime_checkable
class BacktestEngine(Protocol):
    """Run a simulation of a strategy over historical data.

    Must never:
        Use live order routing, real broker connections, or real cash.
        Share any mutable state with the live execution path.

    Research-to-production parity requirement:
        Must consume the same PortfolioConstructor, RiskPolicy,
        CashConstraintEngine, and ExecutionPolicy implementations as live.
        Only BrokerGateway is replaced by SimulatedBrokerGateway.

    Entry points:
        ``run``            — Minimal protocol surface.  Concrete implementations
                             may refuse this call when they require richer inputs
                             (rebalance timestamps, feature series, price series);
                             in that case they must raise ``NotImplementedError``
                             rather than return a silent no-op result.
        ``run_with_data``  — Where available, the primary path: full strategy
                             loop driven by caller-supplied rebalance data.
    """

    async def run(
        self,
        strategy_run: StrategyRun,
        start: datetime,
        end: datetime,
        initial_capital: Decimal,
    ) -> BacktestRun:
        """Execute the full backtest and return a summary BacktestRun.

        Implementations that require rebalance data must raise
        ``NotImplementedError`` here and redirect callers to their
        ``run_with_data`` method.
        """
        ...


class BacktestExecutionPlan(Protocol):
    """Minimal simulated fill-plan shape consumed by research evidence."""

    @property
    def requested_quantity(self) -> int: ...

    @property
    def filled_quantity(self) -> int: ...

    @property
    def adv_shares_20d(self) -> float: ...

    @property
    def participation_pct(self) -> float: ...

    @property
    def spread_bps(self) -> float: ...

    @property
    def implementation_shortfall_bps(self) -> float: ...

    @property
    def is_complete(self) -> bool: ...


class BacktestReplayBroker(Protocol):
    """Broker surface required by research replay loops."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def sync_account(self) -> AccountSnapshot: ...

    def set_market_price(self, instrument_id: uuid.UUID, price: Decimal) -> None: ...

    def configure_execution_model(self, model: ParticipationFillModel) -> None: ...

    def configure_execution_cost_model(
        self,
        *,
        fill_price_adjuster: Callable[[OrderIntent, Decimal], Decimal],
        commission_calculator: Callable[[OrderIntent, Decimal], Decimal],
    ) -> None: ...

    def execution_plan_for(self, order_id: uuid.UUID) -> BacktestExecutionPlan | None: ...


class BacktestOrderIntentRepository(Protocol):
    async def get_intent(self, order_id: uuid.UUID) -> OrderIntent | None: ...


class BacktestSession(Protocol):
    @property
    def order_repo(self) -> BacktestOrderIntentRepository: ...


class BacktestCycleResult(Protocol):
    @property
    def submitted_ids(self) -> Sequence[uuid.UUID]: ...

    @property
    def approved(self) -> Sequence[OrderIntent]: ...

    @property
    def fills(self) -> Sequence[FillEvent]: ...

    @property
    def signals(self) -> Sequence[SignalScore]: ...

    @property
    def target(self) -> PortfolioTarget | None: ...


class PaperSessionFactory(Protocol):
    """Callable that creates the live-like paper session used by research replay."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class StrategyCycleRunner(Protocol):
    """Callable that runs one live-like strategy cycle during research replay."""

    async def __call__(self, *args: object, **kwargs: object) -> BacktestCycleResult: ...


@runtime_checkable
class SignalModel(Protocol):
    """Compute cross-sectional signal scores from feature vectors.

    Must never:
        Perform capital allocation, position sizing, or order construction.
        Access the broker gateway or account state.
    """

    def score(
        self,
        vectors: list[FeatureVector],
        strategy_run: StrategyRun,
    ) -> list[SignalScore]:
        """Return one SignalScore per input FeatureVector."""
        ...

    @property
    def model_version(self) -> str:
        """Semantic version of this signal model implementation."""
        ...


@runtime_checkable
class AlphaSource(SignalModel, Protocol):
    """Governed alpha source used in champion/challenger ensembles."""

    @property
    def source_name(self) -> str:
        """Stable alpha source name, e.g. classical, xgboost, text."""
        ...

    @property
    def required_feature_schema_hash(self) -> str:
        """Feature schema hash required by this source."""
        ...


@runtime_checkable
class ModelArtifactRepository(Protocol):
    """Durable model artifact and model-card registry."""

    async def register_artifact(self, artifact: ModelArtifact) -> None:
        """Persist an immutable model artifact manifest."""
        ...

    async def get_artifact(self, artifact_id: uuid.UUID) -> ModelArtifact | None:
        """Return a model artifact by UUID string."""
        ...

    async def save_model_card(self, card: ModelCard) -> None:
        """Persist the human-reviewable model card for an artifact."""
        ...


@runtime_checkable
class FeatureAuditRepository(Protocol):
    """Durable state for feature-level production admission evidence."""

    async def save_feature_audit(self, result: FeatureAuditResult) -> None:
        """Persist one immutable feature audit result."""
        ...

    async def latest_feature_audit(
        self,
        feature_name: str,
        feature_version: str | None = None,
    ) -> FeatureAuditResult | None:
        """Return the latest audit for a feature, optionally version-scoped."""
        ...

    async def list_feature_audits(
        self,
        *,
        feature_name: str | None = None,
        limit: int = 100,
    ) -> list[FeatureAuditResult]:
        """Return recent feature audit rows, newest first."""
        ...


@runtime_checkable
class PromotionGate(Protocol):
    """Evaluate whether an alpha source may be promoted or retained."""

    async def evaluate_alpha(
        self,
        source_name: str,
        *,
        as_of: datetime,
    ) -> AlphaReadinessReport:
        """Return readiness evidence for an alpha source."""
        ...
