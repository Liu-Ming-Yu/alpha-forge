"""Cross-sectional portfolio simulation for VectorBT research backtests."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.core.contracts import RegimeScaleProvider

from ..artifacts.backtest_artifacts import (
    BacktestCycleMetrics,
    BacktestFillArtifact,
)
from .vectorbt_trade_accounting import (
    exit_non_target_positions,
    rebalance_target_positions,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    import pandas as pd

    from quant_platform.core.contracts import PortfolioConstructor
    from quant_platform.core.domain.signals import RegimeState

    from ..simple.backtest_execution_model import (
        BacktestExecutionModel,
    )
    from ..slippage import (
        IBKRCommissionSchedule,
        SlippageModel,
    )
    from .vectorbt_trade_types import (
        CycleTradeResult,
    )


@dataclass(slots=True)
class VectorBTPortfolioSimulator:
    """Equal-weight target simulation over VectorBT signal frames."""

    execution_model: BacktestExecutionModel
    slippage_model: SlippageModel
    commission_schedule: IBKRCommissionSchedule
    portfolio_constructor: PortfolioConstructor
    parity_mode: bool

    def simulate(
        self,
        *,
        signal_frames: dict[uuid.UUID, pd.DataFrame],
        rebalance_timestamps: list[datetime],
        initial_capital: Decimal,
        regime_series: dict[datetime, RegimeState],
    ) -> tuple[list[Decimal], list[BacktestFillArtifact], list[BacktestCycleMetrics]]:
        """Run equal-weight cross-sectional portfolio accounting."""
        positions: dict[uuid.UUID, Decimal] = {}
        cash = initial_capital
        nav_curve: list[Decimal] = []
        fill_artifacts: list[BacktestFillArtifact] = []
        cycle_metrics: list[BacktestCycleMetrics] = []

        for ts in rebalance_timestamps:
            prices = self._prices_at_timestamp(signal_frames, ts)
            position_value = sum(
                positions.get(iid, Decimal("0")) * prices.get(iid, Decimal("0"))
                for iid in positions
            )
            current_nav = cash + position_value
            if current_nav <= 0:
                current_nav = initial_capital

            target_ids = self._target_ids(signal_frames, prices, ts)
            regime_scale = self._regime_scale(ts, regime_series)

            cycle_commission = Decimal("0")
            cycle_slippage_bps = 0.0
            cycle_fills = 0
            cycle_participation: list[float] = []

            exit_result = self._exit_non_targets(
                ts=ts,
                positions=positions,
                target_ids=target_ids,
                prices=prices,
                cash=cash,
            )
            cash = exit_result.cash
            cycle_commission += exit_result.commission
            cycle_slippage_bps += exit_result.slippage_bps
            cycle_fills += exit_result.fills_count
            cycle_participation.extend(exit_result.participation)
            fill_artifacts.extend(exit_result.fill_artifacts)

            if target_ids:
                rebalance_result = self._rebalance_targets(
                    ts=ts,
                    positions=positions,
                    target_ids=target_ids,
                    prices=prices,
                    cash=cash,
                    regime_scale=regime_scale,
                )
                cash = rebalance_result.cash
                cycle_commission += rebalance_result.commission
                cycle_slippage_bps += rebalance_result.slippage_bps
                cycle_fills += rebalance_result.fills_count
                cycle_participation.extend(rebalance_result.participation)
                fill_artifacts.extend(rebalance_result.fill_artifacts)

            nav = cash + sum(
                positions.get(iid, Decimal("0")) * prices.get(iid, Decimal("0"))
                for iid in positions
            )
            nav_curve.append(nav)
            cycle_metrics.append(
                BacktestCycleMetrics(
                    timestamp=ts,
                    nav=nav,
                    total_commission=cycle_commission,
                    total_slippage_bps=cycle_slippage_bps,
                    signals_count=len(target_ids),
                    fills_count=cycle_fills,
                    orders_count=cycle_fills,
                    fill_rate=1.0 if cycle_fills > 0 else 0.0,
                    average_participation_pct=(
                        sum(cycle_participation) / len(cycle_participation)
                        if cycle_participation
                        else 0.0
                    ),
                    implementation_shortfall_bps=(
                        cycle_slippage_bps / cycle_fills if cycle_fills > 0 else 0.0
                    ),
                )
            )

        return nav_curve, fill_artifacts, cycle_metrics

    def _exit_non_targets(
        self,
        *,
        ts: datetime,
        positions: dict[uuid.UUID, Decimal],
        target_ids: list[uuid.UUID],
        prices: dict[uuid.UUID, Decimal],
        cash: Decimal,
    ) -> CycleTradeResult:
        return exit_non_target_positions(
            ts=ts,
            positions=positions,
            target_ids=target_ids,
            prices=prices,
            cash=cash,
            execution_model=self.execution_model,
            slippage_model=self.slippage_model,
            commission_schedule=self.commission_schedule,
        )

    def _rebalance_targets(
        self,
        *,
        ts: datetime,
        positions: dict[uuid.UUID, Decimal],
        target_ids: list[uuid.UUID],
        prices: dict[uuid.UUID, Decimal],
        cash: Decimal,
        regime_scale: Decimal,
    ) -> CycleTradeResult:
        return rebalance_target_positions(
            ts=ts,
            positions=positions,
            target_ids=target_ids,
            prices=prices,
            cash=cash,
            regime_scale=regime_scale,
            execution_model=self.execution_model,
            slippage_model=self.slippage_model,
            commission_schedule=self.commission_schedule,
        )

    def _regime_scale(
        self,
        ts: datetime,
        regime_series: dict[datetime, RegimeState],
    ) -> Decimal:
        if self.parity_mode:
            return Decimal("1.0")
        regime_state = regime_series.get(ts)
        if regime_state is None:
            return Decimal("1.0")
        scale_provider = self.portfolio_constructor
        if isinstance(scale_provider, RegimeScaleProvider):
            return scale_provider.scale_for_regime(regime_state.regime_label)
        return Decimal("1.0")

    @staticmethod
    def _prices_at_timestamp(
        signal_frames: dict[uuid.UUID, pd.DataFrame],
        ts: datetime,
    ) -> dict[uuid.UUID, Decimal]:
        prices: dict[uuid.UUID, Decimal] = {}
        for iid, frame in signal_frames.items():
            try:
                value = float(frame.loc[ts, "close"])
            except (KeyError, TypeError):
                continue
            if value > 0:
                prices[iid] = Decimal(str(round(value, 6)))
        return prices

    @staticmethod
    def _target_ids(
        signal_frames: dict[uuid.UUID, pd.DataFrame],
        prices: dict[uuid.UUID, Decimal],
        ts: datetime,
    ) -> list[uuid.UUID]:
        return sorted(
            (
                iid
                for iid, frame in signal_frames.items()
                if iid in prices and ts in frame.index and float(frame.loc[ts, "signal"]) > 0.5
            ),
            key=str,
        )
