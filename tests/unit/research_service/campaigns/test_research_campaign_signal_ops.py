from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from quant_platform.config import AlphaSettings, LLMSettings, PlatformSettings
from quant_platform.core.domain.production import (
    PredictionResult,
    ProductionProfile,
    SignalGateRecord,
    SignalGateStatus,
)
from quant_platform.infrastructure.performance import (
    InMemoryPerformanceRepository,
    build_signal_gate_status,
)
from quant_platform.research.campaign import signal_ops
from quant_platform.services.governance_service.production_candidate.signal import (
    _prediction_evidence_check,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@pytest.mark.asyncio
async def test_record_campaign_signal_gates_records_text_model_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def _fake_record_signal_observation(_settings: object, **kwargs: object):
        calls.append(dict(kwargs))
        return SignalGateStatus(
            signal_name=str(kwargs["signal_name"]),
            signal_type=str(kwargs["signal_type"]),
            as_of=kwargs["as_of"],
            observations=int(kwargs["observations"]),
            rolling_ic=float(kwargs["daily_ic"]),
            negative_streak=0,
            max_drawdown=float(kwargs["drawdown"]),
            max_turnover=float(kwargs["turnover"]),
            min_observations=1,
            min_ic=0.0,
            max_negative_streak=3,
            drawdown_limit=-1.0,
            turnover_limit=2.0,
        )

    monkeypatch.setattr(
        signal_ops,
        "record_signal_observation",
        _fake_record_signal_observation,
    )

    await signal_ops.record_campaign_signal_gates(
        object(),
        model_version="text_model_v1",
        train_xgboost=False,
        signal_type="text",
        as_of=datetime(2026, 4, 17, tzinfo=UTC),
        daily_ic=0.04,
        observations=252,
        drawdown=-0.05,
        turnover=1.0,
    )

    assert [(call["signal_name"], call["signal_type"]) for call in calls] == [
        ("text_model_v1", "text"),
        ("text", "text"),
    ]


@pytest.mark.asyncio
async def test_campaign_daily_ics_are_not_double_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _SignalRepo()
    monkeypatch.setattr(signal_ops, "build_performance_repository", lambda _dsn: repo)
    as_of = datetime(2026, 4, 18, tzinfo=UTC)
    daily_ics = tuple(((as_of - timedelta(days=20 - idx)).isoformat(), 0.06) for idx in range(20))

    source_status, model_status = await signal_ops.record_campaign_signal_gates(
        PlatformSettings(_env_file=None),
        model_version="text",
        train_xgboost=False,
        signal_type="text",
        as_of=as_of,
        daily_ic=0.99,
        observations=252,
        drawdown=0.0,
        turnover=0.2,
        daily_ics=daily_ics,
    )

    assert source_status is model_status
    assert len(repo.rows) == 20
    assert source_status.observations == 20
    assert all(row.observations == 1 for row in repo.rows)
    assert {row.as_of for row in repo.rows} == {
        datetime.fromisoformat(raw).astimezone(UTC) for raw, _ in daily_ics
    }


@pytest.mark.asyncio
async def test_19_daily_ics_do_not_pass_min_observations_even_with_aggregate_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _SignalRepo()
    monkeypatch.setattr(signal_ops, "build_performance_repository", lambda _dsn: repo)
    as_of = datetime(2026, 4, 18, tzinfo=UTC)
    daily_ics = tuple(((as_of - timedelta(days=19 - idx)).isoformat(), 0.99) for idx in range(19))

    source_status, _model_status = await signal_ops.record_campaign_signal_gates(
        PlatformSettings(_env_file=None),
        model_version="text",
        train_xgboost=False,
        signal_type="text",
        as_of=as_of,
        daily_ic=0.99,
        observations=20,
        drawdown=0.0,
        turnover=0.2,
        daily_ics=daily_ics,
    )

    assert len(repo.rows) == 19
    assert source_status.observations == 19
    assert source_status.rolling_ic == pytest.approx(0.99)
    assert source_status.passed is False


@pytest.mark.asyncio
async def test_campaign_prediction_evidence_records_each_positive_paper_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _Repo()
    monkeypatch.setattr(signal_ops, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        llm=LLMSettings(text_feature_weights={"text_alpha": 1.0}),
        alpha=AlphaSettings(
            source_weights={
                "classical": 0.70,
                "xgboost": 0.15,
                "text": 0.05,
                "event": 0.05,
                "intraday": 0.05,
            },
            event_feature_weights={"event_alpha": 1.0},
            intraday_feature_weights={"intraday_alpha": 1.0},
        ),
    )
    sample = SupervisedAlphaSample(
        as_of=datetime(2026, 4, 17, tzinfo=UTC),
        instrument_id=uuid.uuid4(),
        features={
            "text_alpha": 0.1,
            "event_alpha": 0.2,
            "intraday_alpha": 0.3,
            "xgb_alpha": 0.4,
        },
        forward_return=0.05,
    )

    counts = await signal_ops.record_campaign_prediction_evidence(
        settings,
        samples=(sample,),
        source_weights=settings.alpha.source_weights,
        model_version="paper-alpha-composite-v1",
        feature_set_version="paper-alpha-composite-v1",
        as_of=datetime(2026, 4, 18, tzinfo=UTC),
        selected_weights={"xgb_alpha": 1.0},
    )

    assert counts == {"event": 1, "intraday": 1, "text": 1, "xgboost": 1}
    assert {row.source for row in repo.rows} == {
        "campaign:event",
        "campaign:intraday",
        "campaign:text",
        "campaign:xgboost",
    }
    assert all(row.horizon == "21d" for row in repo.rows)
    assert all(row.confidence == 1.0 for row in repo.rows)
    assert all(
        row.metadata["feature_set_version"] == "paper-alpha-composite-v1" for row in repo.rows
    )
    assert {
        row.metadata["production_source"]
        for row in repo.rows
        if row.metadata["evidence_scope"] == "offline_campaign"
    } == {"event", "intraday", "text", "xgboost"}


@pytest.mark.asyncio
async def test_offline_campaign_prediction_rows_do_not_satisfy_fresh_runtime_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = InMemoryPerformanceRepository()
    monkeypatch.setattr(signal_ops, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        llm=LLMSettings(text_feature_weights={"text_alpha": 1.0}),
        alpha=AlphaSettings(source_weights={"classical": 0.95, "text": 0.05}),
    )
    campaign_end = datetime(2026, 4, 18, tzinfo=UTC)
    sample_as_of = campaign_end - timedelta(days=10)
    sample = SupervisedAlphaSample(
        as_of=sample_as_of,
        instrument_id=uuid.uuid4(),
        features={"text_alpha": 0.1},
        forward_return=0.05,
    )

    counts = await signal_ops.record_campaign_prediction_evidence(
        settings,
        samples=(sample,),
        source_weights=settings.alpha.source_weights,
        signal_type="text",
        model_version="paper-alpha-catalyst-v10",
        feature_set_version="paper-alpha-catalyst-v10",
        as_of=campaign_end,
        selected_weights={},
    )
    evidence = await repo.forecast_evidence(
        "text",
        as_of=campaign_end,
        stale_after_hours=24,
    )
    check = _prediction_evidence_check("text", evidence, ProductionProfile.PAPER)
    rows = await repo.list_prediction_results(source="campaign:text")

    assert counts == {"text": 1}
    assert rows[0].as_of == sample_as_of
    assert rows[0].blockers == ("offline_campaign_prediction_only",)
    assert rows[0].source == "campaign:text"
    assert rows[0].metadata["production_source"] == "text"
    assert rows[0].metadata["evidence_scope"] == "offline_campaign"
    assert rows[0].metadata["campaign_as_of"] == campaign_end.isoformat()
    assert evidence.stale is True
    assert evidence.blockers == ("no prediction evidence recorded",)
    assert check.name == "prediction_evidence_text_fresh"
    assert check.passed is False


@pytest.mark.asyncio
async def test_offline_campaign_rows_do_not_poison_fresh_runtime_prediction_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = InMemoryPerformanceRepository()
    monkeypatch.setattr(signal_ops, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        llm=LLMSettings(text_feature_weights={"text_alpha": 1.0}),
        alpha=AlphaSettings(source_weights={"classical": 0.95, "text": 0.05}),
    )
    campaign_end = datetime(2026, 4, 18, tzinfo=UTC)
    runtime_as_of = campaign_end + timedelta(hours=1)
    sample = SupervisedAlphaSample(
        as_of=campaign_end - timedelta(days=10),
        instrument_id=uuid.uuid4(),
        features={"text_alpha": 0.1},
        forward_return=0.05,
    )

    await signal_ops.record_campaign_prediction_evidence(
        settings,
        samples=(sample,),
        source_weights=settings.alpha.source_weights,
        signal_type="text",
        model_version="paper-alpha-catalyst-v10",
        feature_set_version="paper-alpha-catalyst-v10",
        as_of=campaign_end,
        selected_weights={},
    )
    await repo.save_prediction_result(
        PredictionResult(
            prediction_id=uuid.uuid4(),
            strategy_run_id=uuid.uuid4(),
            instrument_id=sample.instrument_id,
            source="text",
            model_version="paper-runtime-text-v1",
            as_of=runtime_as_of,
            horizon="rank_1d",
            expected_return=0.02,
            rank_score=0.5,
            confidence=0.8,
            feature_schema_hash="text_runtime_schema",
            calibration_bucket="rank_score_uncalibrated",
        )
    )

    evidence = await repo.forecast_evidence(
        "text",
        as_of=runtime_as_of,
        stale_after_hours=24,
        min_confidence=0.5,
    )

    assert evidence.passed is True
    assert "offline_campaign_prediction_only" not in evidence.blockers
    assert evidence.latest_prediction_at == runtime_as_of
    assert evidence.feature_schema_hashes == ("text_runtime_schema",)


@pytest.mark.asyncio
async def test_campaign_prediction_evidence_records_only_requested_signal_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _Repo()
    monkeypatch.setattr(signal_ops, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        llm=LLMSettings(text_feature_weights={"text_alpha": 1.0}),
        alpha=AlphaSettings(
            source_weights={
                "classical": 0.70,
                "xgboost": 0.15,
                "text": 0.05,
                "event": 0.05,
                "intraday": 0.05,
            },
            event_feature_weights={"event_alpha": 1.0},
            intraday_feature_weights={"intraday_alpha": 1.0},
        ),
    )
    sample = SupervisedAlphaSample(
        as_of=datetime(2026, 4, 17, tzinfo=UTC),
        instrument_id=uuid.uuid4(),
        features={
            "text_alpha": 0.1,
            "event_alpha": 0.2,
            "intraday_alpha": 0.3,
            "xgb_alpha": 0.4,
        },
        forward_return=0.05,
    )

    counts = await signal_ops.record_campaign_prediction_evidence(
        settings,
        samples=(sample,),
        source_weights=settings.alpha.source_weights,
        signal_type="event",
        model_version="paper-alpha-event-v2",
        feature_set_version="paper-alpha-event-reaction-v2",
        as_of=datetime(2026, 4, 18, tzinfo=UTC),
        selected_weights={"xgb_alpha": 1.0},
    )

    assert counts == {"event": 1}
    assert [row.source for row in repo.rows] == ["campaign:event"]
    assert repo.rows[0].model_version == "paper-alpha-event-v2:event"
    assert repo.rows[0].metadata["production_source"] == "event"


@pytest.mark.asyncio
async def test_campaign_prediction_evidence_skips_zero_coverage_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _Repo()
    monkeypatch.setattr(signal_ops, "build_performance_repository", lambda _dsn: repo)
    settings = PlatformSettings(
        _env_file=None,
        alpha=AlphaSettings(
            source_weights={"classical": 0.95, "event": 0.05},
            event_feature_weights={"event_alpha": 1.0},
        ),
    )
    sample = SupervisedAlphaSample(
        as_of=datetime(2026, 4, 17, tzinfo=UTC),
        instrument_id=uuid.uuid4(),
        features={"momentum_1m": 0.1},
        forward_return=0.05,
    )

    counts = await signal_ops.record_campaign_prediction_evidence(
        settings,
        samples=(sample,),
        source_weights=settings.alpha.source_weights,
        signal_type="event",
        model_version="paper-alpha-event-v2",
        feature_set_version="paper-alpha-event-reaction-v2",
        as_of=datetime(2026, 4, 18, tzinfo=UTC),
        selected_weights={},
    )

    assert counts == {"event": 0}
    assert repo.rows == []


class _Repo:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    async def save_prediction_result(self, result: Any) -> None:
        self.rows.append(result)


class _SignalRepo:
    def __init__(self) -> None:
        self.rows: list[SignalGateRecord] = []

    async def record_signal_observation(self, record: SignalGateRecord) -> None:
        self.rows.append(record)
        self.rows.sort(key=lambda row: row.as_of)

    async def signal_status(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        min_observations: int,
        min_ic: float,
        max_negative_streak: int,
        drawdown_limit: float,
        turnover_limit: float,
    ) -> SignalGateStatus:
        return build_signal_gate_status(
            signal_name,
            signal_type,
            as_of=as_of,
            records=list(self.rows),
            min_observations=min_observations,
            min_ic=min_ic,
            max_negative_streak=max_negative_streak,
            drawdown_limit=drawdown_limit,
            turnover_limit=turnover_limit,
        )
