"""Intraday backtest CLI operation composition."""

from __future__ import annotations

import uuid
from datetime import UTC
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.common import (
    _instrument_lookup_from_contracts,
    _load_instrument_contracts,
    _load_intraday_feature_series,
    _parse_intraday_decision_times,
    research_json_result,
)

if TYPE_CHECKING:
    from quant_platform.application.research import (
        BacktestEvidenceAssertRequest,
        BacktestIntradayRequest,
    )
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.market_data.bars import MarketBar


async def backtest_evidence_command(
    settings: PlatformSettings,
    request: BacktestEvidenceAssertRequest,
) -> UseCaseResult[dict[str, object]]:
    del settings
    if request.command != "assert":
        raise OperatorUsageError(f"unknown backtest evidence command: {request.command}")
    from quant_platform.services.research_service.intraday.backtesting.backtest import (
        assert_backtest_evidence,
    )

    try:
        result = assert_backtest_evidence(request.manifest)
    except ValueError as exc:
        raise OperatorUsageError(str(exc)) from exc
    return research_json_result(result)


async def backtest_intraday_command(
    settings: PlatformSettings,
    args: BacktestIntradayRequest,
) -> UseCaseResult[dict[str, object]]:
    """Run canonical intraday event replay plus fail-closed reconciliation."""
    from quant_platform.core.domain.research import IntradayBacktestSpec
    from quant_platform.infrastructure.support.clock import WallClock
    from quant_platform.research.backtesting import (
        create_intraday_backtest_engine,
        create_vectorized_intraday_backtest_engine,
    )
    from quant_platform.services.data_service.intraday import (
        import_result_payload,
        import_vendor_bar_batch,
        load_vendor_bar_batch_from_file,
    )
    from quant_platform.services.data_service.stores.parquet_bar_store import ParquetBarStore
    from quant_platform.services.research_service.intraday.backtesting.backtest import (
        reconcile_intraday_backtests,
        write_backtest_evidence_manifest,
        write_reconciliation_report,
    )

    contracts = _load_instrument_contracts(args.contracts_file)
    lookup = _instrument_lookup_from_contracts(contracts)
    as_of = args.end if args.end.tzinfo else args.end.replace(tzinfo=UTC)
    store = ParquetBarStore(settings.storage.object_store_root)
    dataset_result = None
    minute_bars: dict[uuid.UUID, list[MarketBar]] = {}
    if args.data_file is not None:
        batch = load_vendor_bar_batch_from_file(
            args.data_file,
            vendor=args.vendor,
            instrument_lookup=lookup,
            as_of=as_of,
        )
        dataset_result = await import_vendor_bar_batch(
            batch,
            store=store,
            expected_instruments=set(contracts.keys()),
        )
        if not dataset_result.validation.passed:
            return research_json_result(import_result_payload(dataset_result), passed=False)
        for bar in batch.bars:
            minute_bars.setdefault(bar.instrument_id, []).append(bar)
    else:
        if not args.dataset_id:
            raise OperatorUsageError(
                "backtest intraday requires --data-file or at least one --dataset-id"
            )
        start_utc = args.start if args.start.tzinfo else args.start.replace(tzinfo=UTC)
        end_utc = args.end if args.end.tzinfo else args.end.replace(tzinfo=UTC)
        for instrument_id in contracts:
            bars = await store.get_bars(instrument_id, 60, start_utc, end_utc)
            if bars:
                minute_bars[instrument_id] = bars
        if not minute_bars:
            raise OperatorUsageError(
                "no 1-minute bars found in canonical store for requested dataset window"
            )

    decision_times = _parse_intraday_decision_times(list(args.decision_time), args.start, args.end)
    dataset_ids = tuple(uuid.UUID(str(item)) for item in args.dataset_id) or (
        (dataset_result.dataset.dataset_id,) if dataset_result is not None else ()
    )
    spec = IntradayBacktestSpec(
        strategy_name=args.strategy_name,
        strategy_version=args.strategy_version,
        start=args.start if args.start.tzinfo else args.start.replace(tzinfo=UTC),
        end=args.end if args.end.tzinfo else args.end.replace(tzinfo=UTC),
        initial_capital=Decimal(str(args.initial_capital)),
        decision_times=decision_times,
        dataset_ids=dataset_ids,
        universe_name=args.universe_name,
        feature_set_version=args.feature_set_version,
        model_version=args.model_version,
    )
    feature_series, feature_available_at = await _load_intraday_feature_series(
        settings,
        contracts,
        spec.feature_set_version,
        spec.decision_times,
    )
    event_result = await create_intraday_backtest_engine(settings=settings).run(
        spec=spec,
        feature_series=feature_series,
        feature_available_at=feature_available_at,
        minute_bars=minute_bars,
        instrument_contracts=contracts,
        output_root=args.output_root,
    )
    vector_result = await create_vectorized_intraday_backtest_engine(settings=settings).run(
        spec=spec,
        feature_series=feature_series,
        feature_available_at=feature_available_at,
        minute_bars=minute_bars,
        instrument_contracts=contracts,
        output_root=args.output_root,
    )
    reconciliation = reconcile_intraday_backtests(
        event_result=event_result,
        vectorized_result=vector_result,
        generated_at=WallClock().now(),
    )
    rec_path = event_result.artifact_root / "backtest_reconciliation.json"
    write_reconciliation_report(reconciliation, rec_path)
    manifest_path = write_backtest_evidence_manifest(
        spec=spec,
        event_result=event_result,
        vectorized_result=vector_result,
        reconciliation_report=reconciliation,
        reconciliation_report_path=rec_path,
        output_path=event_result.artifact_root / "backtest_evidence_manifest.json",
        config_payload={
            "settings": settings.model_dump(mode="json"),
            "contracts": {str(k): v for k, v in contracts.items()},
        },
    )
    payload = {
        "passed": reconciliation.passed,
        "strategy_run_id": str(event_result.strategy_run_id),
        "final_capital": str(event_result.final_capital),
        "total_return": str(event_result.total_return),
        "max_drawdown": str(event_result.max_drawdown),
        "residual_order_count": event_result.residual_order_count,
        "artifact_root": str(event_result.artifact_root),
        "reconciliation_report": str(rec_path),
        "evidence_manifest": str(manifest_path),
        "dataset": import_result_payload(dataset_result)
        if dataset_result is not None
        else {"dataset_ids": [str(item) for item in dataset_ids], "source": "canonical_store"},
    }
    return research_json_result(payload, passed=reconciliation.passed)
