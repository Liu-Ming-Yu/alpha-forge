"""Backtest operation composition helpers."""

from __future__ import annotations

import json
import uuid
from datetime import UTC
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from quant_platform.application.errors import OperatorUsageError
from quant_platform.bootstrap.session.public_api import create_paper_session
from quant_platform.research.backtesting.data import (
    prepare_vectorbt_backtest_data,
)
from quant_platform.research.common import (
    _load_instrument_contracts,
    research_json_result,
)
from quant_platform.research.intraday.backtest_ops import (
    backtest_evidence_command,
    backtest_intraday_command,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.application.research import (
        BacktestEvidenceAssertRequest,
        BacktestIntradayRequest,
        BacktestRequest,
        BacktestRunRequest,
    )
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings

log = structlog.get_logger(__name__)


async def _backtest_run(
    settings: PlatformSettings,
    request: BacktestRequest,
) -> UseCaseResult[dict[str, object]]:
    """Dispatch the ``backtest`` subcommands."""
    from quant_platform.application.research import (
        BacktestEvidenceAssertRequest,
        BacktestIntradayRequest,
        BacktestRunRequest,
    )

    if isinstance(request, BacktestRunRequest):
        return await _backtest_run_impl(settings, request)
    if isinstance(request, BacktestIntradayRequest):
        return await backtest_intraday_command(settings, request)
    if isinstance(request, BacktestEvidenceAssertRequest):
        return await backtest_evidence_command(settings, request)
    raise OperatorUsageError(f"unknown backtest request: {type(request).__name__}")


async def _backtest_evidence(
    settings: PlatformSettings,
    request: BacktestEvidenceAssertRequest,
) -> UseCaseResult[dict[str, object]]:
    """Compatibility wrapper for the split backtest evidence operation."""
    return await backtest_evidence_command(settings, request)


async def _backtest_intraday_impl(
    settings: PlatformSettings,
    args: BacktestIntradayRequest,
) -> UseCaseResult[dict[str, object]]:
    """Compatibility wrapper for the split intraday backtest operation."""
    return await backtest_intraday_command(settings, args)


async def _backtest_run_impl(
    settings: PlatformSettings,
    args: BacktestRunRequest,
) -> UseCaseResult[dict[str, object]]:
    """Run a VectorBT vectorized backtest over IB bar data with inline features.

    Data flow:
    1. Backfill required: run ``backfill`` first if bar data is absent.
    2. Full bar history loaded once per instrument (start - warmup to end).
    3. For each rebalance timestamp:
       a. Try the feature repository (populated when Postgres is configured).
       b. Fall back to inline ``build_feature_bundle`` from accumulated closes.
    4. VectorBTBacktestEngine executes the vectorised simulation.
    """
    try:
        import vectorbt  # noqa: F401, PLC0415
    except ModuleNotFoundError as exc:
        raise OperatorUsageError(
            "vectorbt is not installed. Install with: pip install 'quant-platform[backtest]'"
        ) from exc

    from quant_platform.bootstrap.signal_models import build_default_primary_signal_model
    from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun
    from quant_platform.infrastructure.support.clock import WallClock
    from quant_platform.research.backtesting import (
        create_vectorbt_backtest_engine,
    )
    from quant_platform.services.research_service.sampling.samples import daily_as_of_dates

    clock = WallClock()
    now = clock.now()

    start = args.start if args.start.tzinfo else args.start.replace(tzinfo=UTC)
    end = args.end if args.end.tzinfo else args.end.replace(tzinfo=UTC)
    initial_capital = Decimal(str(args.initial_capital))

    contracts = _load_instrument_contracts(args.contracts_file)
    if not contracts:
        raise OperatorUsageError("contracts-file contains no instruments.")
    instrument_ids = list(contracts.keys())

    session = create_paper_session(
        settings=settings,
        initial_cash=Decimal("0"),
        instrument_contracts=contracts,
    )

    # Rebalance schedule
    rebalance_every: int = args.rebalance_every
    all_dates = daily_as_of_dates(start, end)
    rebalance_timestamps = [d for i, d in enumerate(all_dates) if i % rebalance_every == 0]
    if not rebalance_timestamps:
        raise OperatorUsageError("Date range produces no rebalance timestamps.")

    prepared = await prepare_vectorbt_backtest_data(
        session=session,
        instrument_ids=instrument_ids,
        contracts_file=args.contracts_file,
        start=start,
        end=end,
        now=now,
        bar_seconds=args.bar_seconds,
        rebalance_timestamps=rebalance_timestamps,
        feature_set_version=args.feature_set_version,
    )

    # ------------------------------------------------------------------ #
    # Build StrategyRun and engine                                         #
    # ------------------------------------------------------------------ #
    run_id = uuid.uuid4()
    strategy_run = StrategyRun(
        run_id=run_id,
        strategy_name=args.strategy_name,
        strategy_version=args.strategy_version,
        run_type=RunType.BACKTEST,
        status=RunStatus.RUNNING,
        config_snapshot={
            "initial_capital": str(initial_capital),
            "feature_set_version": args.feature_set_version,
            "bar_seconds": args.bar_seconds,
            "rebalance_every": rebalance_every,
            "top_n": args.top_n,
            "inline_features": prepared.inline_feature_count > 0,
        },
        created_at=now,
        started_at=now,
    )

    signal_model = build_default_primary_signal_model(settings)
    engine = create_vectorbt_backtest_engine(
        clock=clock,
        signal_model=signal_model,
        settings=settings,
        universe_manager=session.universe_manager,
        top_n=args.top_n,
    )

    log.info("backtest.simulation_start", run_id=str(run_id))
    result = await engine.run_with_data(
        strategy_run=strategy_run,
        start=start,
        end=end,
        initial_capital=initial_capital,
        rebalance_timestamps=rebalance_timestamps,
        feature_series=prepared.feature_series,
        price_series=prepared.price_series,
        regime_index_series=prepared.regime_index_series,
    )

    # ------------------------------------------------------------------ #
    # Write manifest                                                        #
    # ------------------------------------------------------------------ #
    output_root: Path = args.output_root
    run_dir = output_root / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": str(result.backtest_id),
        "strategy_run_id": str(result.strategy_run_id),
        "strategy_name": args.strategy_name,
        "strategy_version": args.strategy_version,
        "start": result.start_date.isoformat(),
        "end": result.end_date.isoformat(),
        "initial_capital": str(result.initial_capital),
        "final_capital": str(result.final_capital),
        "total_return_pct": f"{float(result.total_return) * 100:.2f}",
        "annualised_sharpe": str(result.annualised_sharpe) if result.annualised_sharpe else None,
        "max_drawdown_pct": f"{float(result.max_drawdown) * 100:.2f}",
        "artifact_uri": result.artifact_uri,
        "created_at": result.created_at.isoformat(),
        "inline_features": prepared.inline_feature_count > 0,
        "instruments": len(instrument_ids),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log.info("backtest.complete", **{k: v for k, v in manifest.items()})
    return research_json_result(
        {
            **manifest,
            "covered_instruments": prepared.covered_instruments,
            "rebalances": len(rebalance_timestamps),
            "manifest_path": str(manifest_path),
        }
    )
