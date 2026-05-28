"""Strategy lifecycle, regime, and signal-decay read models."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.operator_api.read_model_types import (
    RegimeStateView,
    SignalDecayView,
    StrategyHealth,
    StrategyLifecycleView,
)
from quant_platform.core.events import RegimeStateDetected, SignalScorePublished

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.contracts import Clock, EventBus, PerformanceRepository
    from quant_platform.core.events import DomainEvent


class LifecycleReadModelMixin:
    _clock: Clock
    _events: EventBus | None
    _performance: PerformanceRepository | None

    async def _event_history(self, limit: int = 5000) -> list[DomainEvent]:
        raise NotImplementedError

    def strategy_lifecycle(
        self,
        engine_name: str,
        engine_version: str,
        days_active: int,
        rolling_sharpe_90d: float,
        rolling_ic_60d: float,
        max_drawdown_realized: float,
        max_drawdown_limit: float,
        slippage_ratio: float,
        cycles_completed: int,
    ) -> StrategyLifecycleView:
        """Assess strategy health and generate lifecycle recommendation."""
        if days_active < 20:
            health = StrategyHealth.LAUNCHING
            recommendation = (
                f"Shadow/paper mode \u2014 {20 - days_active} days until launch eligibility"
            )
        elif rolling_ic_60d < 0.01 or max_drawdown_realized < max_drawdown_limit:
            health = StrategyHealth.RETIRING
            recommendation = "Retire: IC below 0.01 or max drawdown breached"
        elif rolling_ic_60d < 0.03 or slippage_ratio > 3.0:
            health = StrategyHealth.DEGRADED
            recommendation = "Reduce exposure: IC degrading or slippage excessive"
        elif rolling_sharpe_90d > 0.5 and rolling_ic_60d > 0.03 and slippage_ratio < 1.5:
            health = StrategyHealth.SCALING_UP
            recommendation = "Eligible for capital scale-up"
        else:
            health = StrategyHealth.STABLE
            recommendation = "Operating within normal parameters"

        return StrategyLifecycleView(
            engine_name=engine_name,
            engine_version=engine_version,
            health=health,
            days_active=days_active,
            rolling_sharpe_90d=rolling_sharpe_90d,
            rolling_ic_60d=rolling_ic_60d,
            max_drawdown_realized=max_drawdown_realized,
            max_drawdown_limit=max_drawdown_limit,
            slippage_ratio=slippage_ratio,
            cycles_completed=cycles_completed,
            recommendation=recommendation,
        )

    def regime_state(
        self,
        label: str,
        gross_exposure_scale: float,
        trend_z: float = 0.0,
        annualized_vol: float = 0.0,
        breadth_pct: float = 0.0,
    ) -> RegimeStateView:
        """Build a regime-state operator view."""
        return RegimeStateView(
            as_of=self._clock.now(),
            label=label,
            gross_exposure_scale=gross_exposure_scale,
            trend_z=trend_z,
            annualized_vol=annualized_vol,
            breadth_pct=breadth_pct,
        )

    async def current_regime_state(self) -> RegimeStateView:
        """Return the last ``RegimeStateDetected`` event as an operator view."""
        latest: RegimeStateDetected | None = None
        for event in await self._event_history():
            if isinstance(event, RegimeStateDetected):
                latest = event
        if latest is None:
            return RegimeStateView(
                as_of=self._clock.now(),
                label="unknown",
                gross_exposure_scale=1.0,
                trend_z=0.0,
                annualized_vol=0.0,
                breadth_pct=0.0,
            )
        features = latest.supporting_features
        return RegimeStateView(
            as_of=latest.occurred_at,
            label=latest.regime_label,
            gross_exposure_scale=latest.gross_exposure_scale,
            trend_z=_feature_float(features.get("trend_z", 0.0)),
            annualized_vol=_feature_float(features.get("annualized_vol", 0.0)),
            breadth_pct=_feature_float(features.get("breadth_pct", 0.0)),
        )

    async def current_strategy_lifecycle(
        self,
        strategy_run_id: uuid.UUID,
        *,
        engine_name: str,
        engine_version: str,
        max_drawdown_limit: float = -0.15,
    ) -> StrategyLifecycleView:
        """Compute strategy lifecycle state from persisted data."""
        history = await self._event_history()
        regime_events = [event for event in history if isinstance(event, RegimeStateDetected)]
        cycles_completed = len(regime_events)

        earliest = min((event.occurred_at for event in regime_events), default=None)
        latest = max((event.occurred_at for event in regime_events), default=None)
        days_active = 0
        if earliest is not None and latest is not None:
            delta = latest - earliest
            days_active = max(0, delta.days)

        rolling_sharpe = 0.0
        max_drawdown_realized = 0.0
        gross_turnover = 0.0
        slippage_ratio = 1.0
        if self._performance is not None:
            report = await self._performance.performance_report(
                strategy_run_id,
                as_of=self._clock.now(),
                window=90,
            )
            rolling_sharpe = report.rolling_sharpe
            max_drawdown_realized = report.max_drawdown
            gross_turnover = report.gross_turnover
            slippage_ratio = report.slippage_ratio

        view = self.strategy_lifecycle(
            engine_name=engine_name,
            engine_version=engine_version,
            days_active=days_active,
            rolling_sharpe_90d=rolling_sharpe,
            rolling_ic_60d=0.0,
            max_drawdown_realized=max_drawdown_realized,
            max_drawdown_limit=max_drawdown_limit,
            slippage_ratio=slippage_ratio,
            cycles_completed=cycles_completed,
        )
        return StrategyLifecycleView(
            engine_name=view.engine_name,
            engine_version=view.engine_version,
            health=view.health,
            days_active=view.days_active,
            rolling_sharpe_90d=view.rolling_sharpe_90d,
            rolling_ic_60d=view.rolling_ic_60d,
            max_drawdown_realized=view.max_drawdown_realized,
            max_drawdown_limit=view.max_drawdown_limit,
            slippage_ratio=view.slippage_ratio,
            cycles_completed=view.cycles_completed,
            recommendation=f"{view.recommendation}; turnover={gross_turnover:.4f}",
        )

    async def current_signal_decay(
        self,
        engine_name: str,
        *,
        window: int = 500,
    ) -> SignalDecayView:
        """Compute signal-decay analytics from the last ``window`` scores."""
        history = await self._event_history(limit=window * 4)
        score_events = [event for event in history if isinstance(event, SignalScorePublished)]
        if not score_events:
            return self.signal_decay(engine_name=engine_name, scores=[])

        recent = score_events[-window:]
        scores = [float(getattr(event, "score", 0.0)) for event in recent]

        by_time: dict[datetime, set[uuid.UUID]] = {}
        for event in recent:
            instrument_id = getattr(event, "instrument_id", None)
            if instrument_id is None:
                continue
            by_time.setdefault(event.occurred_at, set()).add(instrument_id)
        cycle_buckets = [by_time[timestamp] for timestamp in sorted(by_time.keys())]
        previous_instruments: set[uuid.UUID] | None = None
        current_instruments: set[uuid.UUID] | None = None
        if len(cycle_buckets) >= 2:
            previous_instruments = cycle_buckets[-2]
            current_instruments = cycle_buckets[-1]

        return self.signal_decay(
            engine_name=engine_name,
            scores=scores,
            previous_instruments=previous_instruments,
            current_instruments=current_instruments,
        )

    def signal_decay(
        self,
        engine_name: str,
        scores: list[float],
        previous_instruments: set[uuid.UUID] | None = None,
        current_instruments: set[uuid.UUID] | None = None,
    ) -> SignalDecayView:
        """Compute signal quality metrics for alpha-decay monitoring."""
        n = len(scores)
        if n == 0:
            return SignalDecayView(
                as_of=self._clock.now(),
                engine_name=engine_name,
                signals_generated=0,
                mean_score=0.0,
                score_dispersion=0.0,
                top_quintile_count=0,
                bottom_quintile_count=0,
                turnover_rate=0.0,
            )

        mean = sum(scores) / n
        variance = sum((score - mean) ** 2 for score in scores) / n
        dispersion = variance**0.5
        sorted_scores = sorted(scores)
        quintile_size = max(1, n // 5)
        top_quintile = sum(1 for score in scores if score >= sorted_scores[-quintile_size])
        bottom_quintile = sum(1 for score in scores if score <= sorted_scores[quintile_size - 1])

        turnover = 0.0
        if previous_instruments and current_instruments:
            union = previous_instruments | current_instruments
            if union:
                changed = len(previous_instruments.symmetric_difference(current_instruments))
                turnover = changed / len(union)

        return SignalDecayView(
            as_of=self._clock.now(),
            engine_name=engine_name,
            signals_generated=n,
            mean_score=mean,
            score_dispersion=dispersion,
            top_quintile_count=top_quintile,
            bottom_quintile_count=bottom_quintile,
            turnover_rate=turnover,
        )


def _feature_float(value: object) -> float:
    if isinstance(value, int | float | str | Decimal):
        return float(value)
    return 0.0
