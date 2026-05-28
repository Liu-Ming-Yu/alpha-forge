from __future__ import annotations

import json
import sys
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform.config import BacktestSettings, PlatformSettings, StorageSettings
from quant_platform.core.domain.research import IntradayBacktestSpec
from quant_platform.core.domain.signals import SignalScore
from quant_platform.services.data_service.intraday import (
    compute_intraday_quorum_evidence,
    load_vendor_bar_batch_from_file,
)
from quant_platform.services.research_service.intraday.backtesting.backtest import (
    IntradayBacktestEngine,
    VectorizedIntradayBacktestEngine,
    assert_backtest_evidence,
    reconcile_intraday_backtests,
    write_backtest_evidence_manifest,
    write_reconciliation_report,
)
from quant_platform.session import create_paper_session

_TEST_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "tests")
_FIXTURE_ROOT = _TEST_ROOT / "fixtures" / "intraday_golden"
_UTC = UTC


class _AlphaSignalModel:
    def score(self, vectors, strategy_run):  # type: ignore[no-untyped-def]
        return [
            SignalScore(
                score_id=uuid.uuid5(uuid.NAMESPACE_URL, f"score:{vec.instrument_id}:{vec.as_of}"),
                instrument_id=vec.instrument_id,
                strategy_run_id=strategy_run.run_id,
                as_of=vec.as_of,
                score=float(vec.features.get("alpha", 0.0)),
                confidence=1.0,
                model_version="fixture",
                feature_vector_id=vec.vector_id,
            )
            for vec in vectors
        ]


@pytest.mark.industrial_backtest
async def test_intraday_golden_fixture_pipeline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vectorbt", SimpleNamespace(__version__="test-vectorbt"))
    expected = json.loads((_FIXTURE_ROOT / "expected_summary.json").read_text(encoding="utf-8"))
    contracts_payload = json.loads((_FIXTURE_ROOT / "contracts.json").read_text(encoding="utf-8"))
    contracts = {uuid.UUID(key): dict(value) for key, value in contracts_payload.items()}
    lookup = {value["symbol"]: key for key, value in contracts.items()}
    start = datetime(2026, 1, 2, 14, 30, tzinfo=_UTC)
    end = datetime(2026, 1, 5, 20, 59, tzinfo=_UTC)
    as_of = datetime(2026, 1, 6, tzinfo=_UTC)
    primary = load_vendor_bar_batch_from_file(
        _FIXTURE_ROOT / "vendor_primary.csv",
        vendor="fixture_primary",
        instrument_lookup=lookup,
        as_of=as_of,
    )
    secondary = load_vendor_bar_batch_from_file(
        _FIXTURE_ROOT / "vendor_secondary.csv",
        vendor="fixture_secondary",
        instrument_lookup=lookup,
        as_of=as_of,
    )
    quorum = compute_intraday_quorum_evidence(
        {"fixture_primary": primary, "fixture_secondary": secondary},
        as_of=as_of,
    )

    assert len(primary.bars) == expected["primary_rows"]
    assert len(secondary.bars) == expected["secondary_rows"]
    assert quorum.passed is expected["quorum_passed"]

    decision_times = (
        datetime(2026, 1, 2, 14, 30, tzinfo=_UTC),
        datetime(2026, 1, 5, 14, 30, tzinfo=_UTC),
    )
    ids = sorted(contracts)
    feature_series = {
        decision_times[0]: {
            ids[0]: {"alpha": 1.0},
            ids[1]: {"alpha": 0.5},
            ids[2]: {"alpha": -1.0},
        },
        decision_times[1]: {
            ids[0]: {"alpha": -1.0},
            ids[1]: {"alpha": 1.0},
            ids[2]: {"alpha": -0.5},
        },
    }
    availability = {ts: ts for ts in decision_times}
    minute_bars = {}
    for bar in primary.bars:
        minute_bars.setdefault(bar.instrument_id, []).append(bar)
    spec = IntradayBacktestSpec(
        strategy_name="golden_intraday",
        strategy_version="0.1.0",
        start=start,
        end=end,
        initial_capital=Decimal("100000"),
        decision_times=decision_times,
        dataset_ids=(uuid.uuid5(uuid.NAMESPACE_URL, "golden-primary"),),
        universe_name="golden",
        feature_set_version="golden",
        model_version="fixture",
    )
    settings = PlatformSettings(
        _env_file=None,
        storage=StorageSettings(object_store_root=str(tmp_path)),
        backtest=BacktestSettings(require_market_regime=False),
    )
    signal_model = _AlphaSignalModel()
    event = await IntradayBacktestEngine(
        settings=settings,
        signal_model=signal_model,
        paper_session_factory=create_paper_session,
    ).run(
        spec=spec,
        feature_series=feature_series,
        feature_available_at=availability,
        minute_bars=minute_bars,
        instrument_contracts=contracts,
        output_root=tmp_path / "event",
    )
    vectorized = await VectorizedIntradayBacktestEngine(
        settings=settings,
        signal_model=signal_model,
        paper_session_factory=create_paper_session,
    ).run(
        spec=spec,
        feature_series=feature_series,
        feature_available_at=availability,
        minute_bars=minute_bars,
        instrument_contracts=contracts,
        output_root=tmp_path / "vector",
    )
    reconciliation = reconcile_intraday_backtests(
        event_result=event,
        vectorized_result=vectorized,
        generated_at=as_of,
    )
    rec_path = write_reconciliation_report(
        reconciliation, event.artifact_root / "backtest_reconciliation.json"
    )
    manifest = write_backtest_evidence_manifest(
        spec=spec,
        event_result=event,
        vectorized_result=vectorized,
        reconciliation_report=reconciliation,
        reconciliation_report_path=rec_path,
        output_path=event.artifact_root / "backtest_evidence_manifest.json",
        config_payload={"fixture": "golden"},
    )

    assert_backtest_evidence(manifest)
    assert sorted(path.name for path in event.artifact_root.iterdir()) == expected["artifact_files"]
