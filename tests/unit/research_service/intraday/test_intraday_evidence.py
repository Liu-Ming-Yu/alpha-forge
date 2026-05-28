from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_platform.core.domain.research import IntradayBacktestSpec
from quant_platform.services.research_service.intraday.backtesting.backtest import (
    IntradayBacktestResult,
    assert_backtest_evidence,
    reconcile_intraday_backtests,
    write_backtest_evidence_manifest,
    write_reconciliation_report,
)


def test_backtest_evidence_manifest_asserts_passing_reconciliation(tmp_path) -> None:
    now = datetime(2026, 1, 2, 15, 0, tzinfo=UTC)
    dataset_id = uuid.uuid4()
    spec = IntradayBacktestSpec(
        strategy_name="industrial_test",
        strategy_version="0.1.0",
        start=now,
        end=now + timedelta(minutes=1),
        initial_capital=Decimal("100000"),
        decision_times=(now,),
        dataset_ids=(dataset_id,),
        universe_name="test_universe",
        feature_set_version="test_features",
        model_version="test_model",
    )
    run_id = uuid.uuid4()
    common = {
        "strategy_run_id": run_id,
        "final_capital": Decimal("100000"),
        "total_return": Decimal("0"),
        "max_drawdown": Decimal("0"),
        "nav_curve": ((now, Decimal("100000")),),
        "target_weights": {now: {}},
        "eligible_universe": {now: ()},
        "fills": (),
        "residual_order_count": 0,
        "artifact_root": tmp_path,
        "run_summary_uri": (tmp_path / "run_summary.json").as_uri(),
        "execution_quality_uri": (tmp_path / "execution_quality.json").as_uri(),
        "fills_uri": (tmp_path / "fills.json").as_uri(),
        "target_weights_uri": (tmp_path / "target_weights.json").as_uri(),
    }
    event = IntradayBacktestResult(**common)
    vectorized = IntradayBacktestResult(**common)
    reconciliation = reconcile_intraday_backtests(
        event_result=event,
        vectorized_result=vectorized,
        generated_at=now,
    )
    rec_path = write_reconciliation_report(reconciliation, tmp_path / "reconciliation.json")
    manifest_path = write_backtest_evidence_manifest(
        spec=spec,
        event_result=event,
        vectorized_result=vectorized,
        reconciliation_report=reconciliation,
        reconciliation_report_path=rec_path,
        output_path=tmp_path / "backtest_evidence_manifest.json",
        config_payload={"test": True},
    )

    result = assert_backtest_evidence(manifest_path)

    assert result["passed"] is True
