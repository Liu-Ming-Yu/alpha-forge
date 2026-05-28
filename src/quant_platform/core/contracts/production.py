"""Production-readiness contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.production import (
        AlertEvent,
        BrokerHealthObservation,
        BrokerSmokeObservation,
        CombinedPortfolioTarget,
        EngineBudget,
        EngineTargetContribution,
        ForecastEvidence,
        MetricRollupSnapshot,
        NavSnapshot,
        OperatorAction,
        OperatorApiKey,
        OrderAllocation,
        PaperLifecycleObservation,
        PerformanceReport,
        PredictionResult,
        ReadinessSnapshot,
        RunbookEvidence,
        RuntimeHeartbeat,
        ShadowPaperParityRecord,
        ShadowPaperParityStatus,
        SignalContribution,
        SignalGateRecord,
        SignalGateStatus,
        TextSignalGateRecord,
        TextSignalGateStatus,
    )


@runtime_checkable
class NavSnapshotRepository(Protocol):
    """Persistence contract for live/paper NAV snapshots."""

    async def save_nav_snapshot(self, snapshot: NavSnapshot) -> None: ...

    async def list_nav_snapshots(
        self,
        strategy_run_id: uuid.UUID,
        *,
        limit: int = 252,
    ) -> list[NavSnapshot]: ...


@runtime_checkable
class PerformanceRepository(NavSnapshotRepository, Protocol):
    """Repository for strategy performance governance state."""

    async def performance_report(
        self,
        strategy_run_id: uuid.UUID,
        *,
        as_of: datetime,
        window: int = 90,
    ) -> PerformanceReport: ...

    async def save_metric_rollup(self, snapshot: MetricRollupSnapshot) -> None: ...

    async def list_metric_rollups(
        self,
        metric_name: str | None = None,
        *,
        limit: int = 500,
    ) -> list[MetricRollupSnapshot]: ...


@runtime_checkable
class TextSignalPromotionGate(Protocol):
    """Persistence-backed text-signal promotion gate."""

    async def record_ic(self, record: TextSignalGateRecord) -> None: ...

    async def status(
        self,
        strategy_name: str,
        *,
        as_of: datetime,
        min_observations: int = 20,
        min_ic: float = 0.05,
        max_negative_streak: int = 3,
    ) -> TextSignalGateStatus: ...


@runtime_checkable
class SignalPromotionGate(Protocol):
    """Persistence-backed promotion gate for all signal families."""

    async def record_signal_observation(self, record: SignalGateRecord) -> None: ...

    async def signal_status(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        min_observations: int = 20,
        min_ic: float = 0.05,
        max_negative_streak: int = 3,
        drawdown_limit: float = -0.10,
        turnover_limit: float = 1.0,
    ) -> SignalGateStatus: ...


@runtime_checkable
class PredictionEvidenceRepository(Protocol):
    """Persistence for point-in-time alpha forecasts and promotion evidence."""

    async def save_prediction_result(self, result: PredictionResult) -> None: ...

    async def list_prediction_results(
        self,
        *,
        source: str | None = None,
        model_version: str | None = None,
        strategy_run_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        as_of: datetime | None = None,
        limit: int = 500,
    ) -> list[PredictionResult]: ...

    async def forecast_evidence(
        self,
        source: str,
        *,
        model_version: str | None = None,
        as_of: datetime,
        stale_after_hours: int = 24,
        min_confidence: float = 0.0,
        limit: int = 500,
    ) -> ForecastEvidence: ...


@runtime_checkable
class ShadowPaperParityRepository(Protocol):
    """Persistence for shadow-vs-paper parity promotion evidence."""

    async def save_shadow_paper_parity(self, record: ShadowPaperParityRecord) -> None: ...

    async def list_shadow_paper_parity(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        limit: int = 252,
    ) -> list[ShadowPaperParityRecord]: ...

    async def shadow_paper_parity_status(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        min_trading_days: int = 20,
        max_target_weight_diff_bps: float = 1.0,
        limit: int = 252,
    ) -> ShadowPaperParityStatus: ...


@runtime_checkable
class OperationalReadinessRepository(Protocol):
    """Persistence for runtime/broker evidence used by readiness gates."""

    async def save_runtime_heartbeat(self, heartbeat: RuntimeHeartbeat) -> None: ...

    async def latest_runtime_heartbeat(self, component: str) -> RuntimeHeartbeat | None: ...

    async def save_broker_health(self, observation: BrokerHealthObservation) -> None: ...

    async def latest_broker_health(self) -> BrokerHealthObservation | None: ...

    async def save_broker_smoke(self, observation: BrokerSmokeObservation) -> None: ...

    async def latest_broker_smoke(self) -> BrokerSmokeObservation | None: ...

    async def save_paper_lifecycle(self, observation: PaperLifecycleObservation) -> None: ...

    async def latest_paper_lifecycle(self) -> PaperLifecycleObservation | None: ...


@runtime_checkable
class GovernanceEvidenceRepository(
    PerformanceRepository,
    OperationalReadinessRepository,
    PredictionEvidenceRepository,
    ShadowPaperParityRepository,
    SignalPromotionGate,
    TextSignalPromotionGate,
    Protocol,
):
    """Combined governance evidence repository used at bootstrap boundaries."""


@runtime_checkable
class MultiEngineGovernanceRepository(Protocol):
    """Persistence for multi-engine budgets, merged targets, and attribution."""

    async def save_engine_budget(self, budget: EngineBudget) -> None: ...

    async def list_engine_budgets(self) -> list[EngineBudget]: ...

    async def save_combined_target(self, target: CombinedPortfolioTarget) -> None: ...

    async def list_target_contributions(
        self,
        combined_target_id: uuid.UUID,
    ) -> list[EngineTargetContribution]: ...

    async def save_order_allocations(self, allocations: list[OrderAllocation]) -> None: ...

    async def list_order_allocations(self, order_id: uuid.UUID) -> list[OrderAllocation]: ...


@runtime_checkable
class SignalContributionRepository(Protocol):
    """Persistence for ensemble/source signal attribution."""

    async def save_signal_contributions(
        self,
        contributions: list[SignalContribution],
    ) -> None: ...

    async def list_signal_contributions(
        self,
        *,
        strategy_run_id: uuid.UUID | None = None,
        score_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        limit: int = 500,
    ) -> list[SignalContribution]: ...


@runtime_checkable
class OperatorActionRepository(Protocol):
    """Durable human command/audit evidence."""

    async def record_operator_action(self, action: OperatorAction) -> None: ...

    async def list_operator_actions(
        self,
        *,
        action_type: str | None = None,
        limit: int = 200,
    ) -> list[OperatorAction]: ...


@runtime_checkable
class ProductionEvidenceRepository(Protocol):
    """Runbook, alert, and readiness evidence persistence."""

    async def save_runbook_evidence(self, evidence: RunbookEvidence) -> None: ...

    async def save_alert_event(self, event: AlertEvent) -> None: ...

    async def save_readiness_snapshot(self, snapshot: ReadinessSnapshot) -> None: ...

    async def latest_readiness_snapshot(self, profile: str) -> ReadinessSnapshot | None: ...


@runtime_checkable
class OperatorAuthRepository(Protocol):
    """Durable operator API key and RBAC repository."""

    async def save_api_key(self, key: OperatorApiKey) -> None: ...

    async def get_api_key_by_hash(self, key_hash: str) -> OperatorApiKey | None: ...

    async def revoke_api_key(
        self,
        key_id: uuid.UUID,
        *,
        revoked_at: datetime,
    ) -> None: ...
