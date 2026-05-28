from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.config import (
    ApiSettings,
    ExecutionSettings,
    LiquiditySettings,
    PlatformSettings,
    ProductionSettings,
    RegimeSettings,
    RiskSettings,
    StorageSettings,
    V2Settings,
)
from quant_platform.core.domain.production import (
    MetricRollupSnapshot,
    PredictionResult,
    ProductionProfile,
)
from quant_platform.infrastructure.performance import InMemoryPerformanceRepository
from quant_platform.services.governance_service.production_candidate.signal import (
    _prediction_evidence_check,
)


def _prediction(
    *,
    as_of: datetime,
    confidence: float = 0.7,
    source: str = "xgboost",
) -> PredictionResult:
    return PredictionResult(
        prediction_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        instrument_id=uuid.uuid4(),
        source=source,
        model_version=f"{source}-v1",
        as_of=as_of,
        horizon="5d",
        expected_return=0.012,
        rank_score=0.91,
        confidence=confidence,
        feature_schema_hash="abc123",
        calibration_bucket="passive|0.5-2pct_adv|tight_spread|fresh|limit|open",
    )


@pytest.mark.asyncio
async def test_prediction_evidence_passes_when_fresh_and_confident() -> None:
    repo = InMemoryPerformanceRepository()
    as_of = datetime(2026, 5, 9, 15, tzinfo=UTC)
    await repo.save_prediction_result(_prediction(as_of=as_of - timedelta(hours=2)))

    evidence = await repo.forecast_evidence(
        "xgboost",
        as_of=as_of,
        stale_after_hours=24,
        min_confidence=0.5,
    )

    assert evidence.passed
    assert evidence.observations == 1
    assert not evidence.stale
    assert evidence.calibration_buckets


@pytest.mark.asyncio
async def test_fresh_text_prediction_evidence_satisfies_candidate_check() -> None:
    repo = InMemoryPerformanceRepository()
    as_of = datetime(2026, 5, 9, 15, tzinfo=UTC)
    await repo.save_prediction_result(_prediction(as_of=as_of, source="text"))

    evidence = await repo.forecast_evidence(
        "text",
        as_of=as_of,
        stale_after_hours=24,
        min_confidence=0.5,
    )
    check = _prediction_evidence_check("text", evidence, ProductionProfile.PAPER)

    assert evidence.passed
    assert check.name == "prediction_evidence_text_fresh"
    assert check.passed


@pytest.mark.asyncio
async def test_prediction_evidence_fails_stale_or_low_confidence() -> None:
    repo = InMemoryPerformanceRepository()
    as_of = datetime(2026, 5, 9, 15, tzinfo=UTC)
    await repo.save_prediction_result(_prediction(as_of=as_of - timedelta(days=2), confidence=0.1))

    evidence = await repo.forecast_evidence(
        "xgboost",
        as_of=as_of,
        stale_after_hours=24,
        min_confidence=0.5,
    )

    assert not evidence.passed
    assert evidence.stale
    assert "below minimum" in "; ".join(evidence.blockers)


@pytest.mark.asyncio
async def test_metric_rollups_are_durable_evidence_snapshots() -> None:
    repo = InMemoryPerformanceRepository()
    as_of = datetime(2026, 5, 9, 15, tzinfo=UTC)
    snapshot = MetricRollupSnapshot(
        snapshot_id=uuid.uuid4(),
        metric_name="quant_order_submit_latency_seconds_p95",
        as_of=as_of,
        window="7d",
        value=0.42,
        labels={"engine": "xsec"},
    )

    await repo.save_metric_rollup(snapshot)

    rows = await repo.list_metric_rollups("quant_order_submit_latency_seconds_p95")
    assert rows == [snapshot]


def test_prediction_result_requires_point_in_time_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _prediction(as_of=datetime(2026, 5, 9, 15))


def test_industrial_profile_fails_closed_on_permissive_defaults(tmp_path) -> None:
    object_root = tmp_path / "objects"
    object_root.mkdir()

    with pytest.raises(ValueError, match="profile_preset='industrial'"):
        PlatformSettings(
            _env_file=None,
            production=ProductionSettings(profile_preset="industrial"),
            storage=StorageSettings(object_store_root=str(object_root)),
        )


def test_industrial_profile_accepts_live_grade_settings(tmp_path) -> None:
    object_root = tmp_path / "objects"
    object_root.mkdir()

    settings = PlatformSettings(
        _env_file=None,
        production=ProductionSettings(profile_preset="industrial"),
        storage=StorageSettings(
            postgres_dsn="postgresql+psycopg://u:p@localhost/db",
            redis_url="redis://localhost:6379/0",
            event_bus_backend="redis_streams",
            object_store_root=str(object_root),
        ),
        api=ApiSettings(operator_api_key="secret"),
        liquidity=LiquiditySettings(allow_missing_profile=False),
        risk=RiskSettings(
            require_sector_mapping=True,
            require_registered_model_match=True,
        ),
        execution=ExecutionSettings(trading_hours_enforced=True),
        regime=RegimeSettings(
            market_proxy_instrument_id=str(uuid.uuid4()),
            require_seed_on_cycle=True,
        ),
        v2=V2Settings(
            enabled=True,
            account_orchestrator_enabled=True,
            require_event_sourced_oms=True,
            require_dataset_quorum=True,
            third_eod_vendor="polygon",
            readiness_snapshot_required=True,
        ),
    )

    assert settings.production.profile_preset == "industrial"
