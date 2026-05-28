"""Multi-engine allocation and signal-contribution read models."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.operator_api.read_model_types import (
    CombinedExposureView,
    EngineBudgetView,
    ForecastEvidenceView,
    OrderAllocationView,
    SignalContributionView,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.core.contracts import (
        Clock,
        MultiEngineGovernanceRepository,
        PredictionEvidenceRepository,
        SignalContributionRepository,
    )


class MultiEngineReadModelMixin:
    _clock: Clock
    _multi_engine: MultiEngineGovernanceRepository | None
    _signal_contributions: SignalContributionRepository | None
    _prediction_evidence: PredictionEvidenceRepository | None

    async def engine_budgets(self) -> list[EngineBudgetView]:
        """Return configured multi-engine budgets."""
        if self._multi_engine is None:
            return []
        budgets = await self._multi_engine.list_engine_budgets()
        return [
            EngineBudgetView(
                engine_name=budget.engine_name,
                engine_version=budget.engine_version,
                run_mode=budget.run_mode,
                capital_weight=budget.capital_weight,
                max_gross=budget.max_gross,
                max_turnover=budget.max_turnover,
                enabled=budget.enabled,
            )
            for budget in budgets
        ]

    async def combined_exposure(self) -> CombinedExposureView:
        """Summarise enabled engine budget allocation."""
        budgets = await self.engine_budgets()
        allocated = sum(
            (budget.capital_weight for budget in budgets if budget.enabled),
            Decimal("0"),
        )
        return CombinedExposureView(
            as_of=self._clock.now(),
            enabled_engines=sum(1 for budget in budgets if budget.enabled),
            allocated_capital_weight=allocated,
            reserved_cash_weight=max(Decimal("0"), Decimal("1") - allocated),
        )

    async def order_allocations(self, order_id: uuid.UUID) -> list[OrderAllocationView]:
        """Return attribution rows for one merged order."""
        if self._multi_engine is None:
            return []
        rows = await self._multi_engine.list_order_allocations(order_id)
        return [
            OrderAllocationView(
                order_id=row.order_id,
                engine_name=row.engine_name,
                strategy_run_id=row.strategy_run_id,
                instrument_id=row.instrument_id,
                allocated_weight=row.allocated_weight,
                allocated_notional=row.allocated_notional,
            )
            for row in rows
        ]

    async def signal_contributions(
        self,
        *,
        strategy_run_id: uuid.UUID | None = None,
        instrument_id: uuid.UUID | None = None,
        limit: int = 500,
    ) -> list[SignalContributionView]:
        """Return latest ensemble/source signal contribution rows."""
        if self._signal_contributions is None:
            return []
        rows = await self._signal_contributions.list_signal_contributions(
            strategy_run_id=strategy_run_id,
            instrument_id=instrument_id,
            limit=limit,
        )
        return [
            SignalContributionView(
                score_id=row.score_id,
                strategy_run_id=row.strategy_run_id,
                instrument_id=row.instrument_id,
                as_of=row.as_of,
                source=row.source,
                source_model_version=row.source_model_version,
                raw_score=row.raw_score,
                normalized_score=row.normalized_score,
                blend_weight=row.blend_weight,
                confidence=row.confidence,
                promotion_state=row.promotion_state,
            )
            for row in rows
        ]

    async def forecast_evidence(
        self,
        sources: list[str],
        *,
        stale_after_hours: int = 24,
        min_confidence: float = 0.0,
    ) -> list[ForecastEvidenceView]:
        """Return prediction evidence summaries for promoted alpha sources."""
        if self._prediction_evidence is None:
            return []
        rows = []
        as_of = self._clock.now()
        for source in sources:
            evidence = await self._prediction_evidence.forecast_evidence(
                source,
                as_of=as_of,
                stale_after_hours=stale_after_hours,
                min_confidence=min_confidence,
            )
            rows.append(
                ForecastEvidenceView(
                    source=evidence.source,
                    model_version=evidence.model_version,
                    as_of=evidence.as_of,
                    horizon=evidence.horizon,
                    observations=evidence.observations,
                    mean_confidence=evidence.mean_confidence,
                    latest_prediction_at=evidence.latest_prediction_at,
                    stale=evidence.stale,
                    passed=evidence.passed,
                    blockers=evidence.blockers,
                    calibration_buckets=evidence.calibration_buckets,
                )
            )
        return rows
