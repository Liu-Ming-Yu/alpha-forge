"""Industrial intraday backtest replay and fail-closed evidence checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.config import PlatformSettings
from quant_platform.services.research_service.intraday.backtesting.types import (
    IntradayBacktestResult,
    IntradayFillArtifact,
)
from quant_platform.services.research_service.intraday.evidence.evidence import (
    assert_backtest_evidence,
    reconcile_intraday_backtests,
    write_backtest_evidence_manifest,
    write_reconciliation_report,
)
from quant_platform.services.research_service.intraday.replay.event_replay import (
    run_event_driven_intraday_backtest,
)
from quant_platform.services.research_service.intraday.replay.replay import (
    IntradayTacticReplayModel,
)
from quant_platform.services.research_service.intraday.vectorized.engine import (
    VectorizedIntradayBacktestEngine,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path

    from quant_platform.core.contracts import PaperSessionFactory, PortfolioConstructor
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.core.domain.research import IntradayBacktestSpec, StrategyRun

__all__ = [
    "IntradayBacktestEngine",
    "IntradayBacktestResult",
    "IntradayFillArtifact",
    "IntradayTacticReplayModel",
    "VectorizedIntradayBacktestEngine",
    "assert_backtest_evidence",
    "reconcile_intraday_backtests",
    "write_backtest_evidence_manifest",
    "write_reconciliation_report",
]


class IntradayBacktestEngine:
    """Canonical event-driven intraday backtest engine."""

    def __init__(
        self,
        *,
        settings: PlatformSettings | None = None,
        replay_model: IntradayTacticReplayModel | None = None,
        portfolio_constructor: PortfolioConstructor | None = None,
        signal_model: object | None = None,
        paper_session_factory: PaperSessionFactory | None = None,
    ) -> None:
        self._settings = settings or PlatformSettings()
        self._replay = replay_model or IntradayTacticReplayModel()
        self._portfolio_constructor = portfolio_constructor
        self._signal_model = signal_model
        self._paper_session_factory = paper_session_factory

    async def run(
        self,
        *,
        spec: IntradayBacktestSpec,
        feature_series: Mapping[datetime, Mapping[uuid.UUID, Mapping[str, float]]],
        feature_available_at: Mapping[datetime, datetime],
        minute_bars: Mapping[uuid.UUID, list[MarketBar]],
        instrument_contracts: Mapping[uuid.UUID, dict[str, object]],
        output_root: Path,
        strategy_run: StrategyRun | None = None,
    ) -> IntradayBacktestResult:
        """Run canonical intraday replay and write evidence artifacts."""
        return await run_event_driven_intraday_backtest(
            settings=self._settings,
            replay_model=self._replay,
            portfolio_constructor=self._portfolio_constructor,
            signal_model=self._signal_model,
            paper_session_factory=self._paper_session_factory,
            spec=spec,
            feature_series=feature_series,
            feature_available_at=feature_available_at,
            minute_bars=minute_bars,
            instrument_contracts=instrument_contracts,
            output_root=output_root,
            strategy_run=strategy_run,
        )
