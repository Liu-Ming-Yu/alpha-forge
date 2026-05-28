from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from quant_platform.core.domain.instruments import (
    AssetClass,
    Instrument,
    SecurityMasterQuality,
    SecurityMasterRecord,
    SymbolHistory,
    UniverseSnapshot,
)
from quant_platform.core.domain.orders import (
    ExecutionQualityReport,
    ExecutionTactic,
    OrderStateEvent,
    OrderStateEventType,
    OrderStatus,
)
from quant_platform.core.domain.portfolio import PortfolioRiskModel, RiskSnapshot
from quant_platform.core.domain.production import OperatorApiKey
from quant_platform.core.domain.research import (
    AlphaReadinessReport,
    FeatureDataset,
    ModelArtifact,
    PromotionState,
)
from quant_platform.infrastructure.v2.state import (
    InMemoryDatasetCatalog,
    InMemoryExecutionQualityRepository,
    InMemoryInstrumentRepository,
    InMemoryModelArtifactRepository,
    InMemoryOrderStateStore,
    InMemoryPortfolioRiskModelRepository,
    InMemoryProductionEvidenceRepository,
)


def _instrument(symbol: str = "AAPL") -> Instrument:
    return Instrument(
        instrument_id=uuid.uuid4(),
        symbol=symbol,
        exchange="XNAS",
        asset_class=AssetClass.EQUITY,
        currency="USD",
        sector="Information Technology",
    )


@pytest.mark.asyncio
async def test_security_master_is_point_in_time_and_fail_closed() -> None:
    repo = InMemoryInstrumentRepository()
    instrument = _instrument()
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    record = SecurityMasterRecord(
        record_id=uuid.uuid4(),
        instrument=instrument,
        as_of=as_of,
        available_at=as_of,
        identifiers={"ib_con_id": "265598"},
        quality=SecurityMasterQuality.APPROVED,
    )

    await repo.upsert_security_master_record(record)
    await repo.add_symbol_history(
        SymbolHistory(
            history_id=uuid.uuid4(),
            instrument_id=instrument.instrument_id,
            symbol="AAPL",
            valid_from=date(1980, 12, 12),
        )
    )
    await repo.save_universe_snapshot(
        UniverseSnapshot(
            snapshot_id=uuid.uuid4(),
            universe_name="us_equities",
            as_of=as_of,
            available_at=as_of,
            instrument_ids=(instrument.instrument_id,),
        )
    )

    assert await repo.require_record(instrument.instrument_id, as_of=as_of) == record
    assert await repo.resolve_symbol("AAPL", as_of=as_of) == instrument.instrument_id
    snapshot = await repo.latest_universe_snapshot("us_equities", as_of=as_of)
    assert snapshot is not None
    assert snapshot.instrument_ids == (instrument.instrument_id,)

    with pytest.raises(LookupError):
        await repo.require_record(uuid.uuid4(), as_of=as_of)


@pytest.mark.asyncio
async def test_feature_dataset_catalog_returns_latest_approved_manifest() -> None:
    catalog = InMemoryDatasetCatalog()
    older = FeatureDataset(
        dataset_id=uuid.uuid4(),
        feature_set_version="v2",
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        available_at=datetime(2026, 1, 1, tzinfo=UTC),
        schema_hash="schema-a",
        source_dataset_ids=(uuid.uuid4(),),
        artifact_uri="s3://features/older",
        quality_status="approved",
    )
    newer_pending = FeatureDataset(
        dataset_id=uuid.uuid4(),
        feature_set_version="v2",
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        available_at=datetime(2026, 1, 2, tzinfo=UTC),
        schema_hash="schema-b",
        source_dataset_ids=(uuid.uuid4(),),
        artifact_uri="s3://features/newer",
        quality_status="pending",
    )

    await catalog.register_feature_dataset(older)
    await catalog.register_feature_dataset(newer_pending)

    latest = await catalog.latest_feature_dataset(
        "v2",
        as_of=datetime(2026, 1, 3, tzinfo=UTC),
    )
    assert latest == older


@pytest.mark.asyncio
async def test_model_risk_execution_and_operator_evidence_repositories() -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    artifact_repo = InMemoryModelArtifactRepository()
    artifact = ModelArtifact(
        artifact_id=uuid.uuid4(),
        model_name="xsec",
        model_version="2.0.0",
        artifact_uri="s3://models/xsec",
        artifact_hash="abc",
        feature_schema_hash="schema",
        training_start=datetime(2025, 1, 1, tzinfo=UTC),
        training_end=datetime(2025, 12, 31, tzinfo=UTC),
        created_at=as_of,
        promotion_state=PromotionState.PAPER,
    )
    await artifact_repo.register_artifact(artifact)
    await artifact_repo.save_alpha_readiness(
        AlphaReadinessReport(
            report_id=uuid.uuid4(),
            alpha_source="xsec",
            as_of=as_of,
            promotion_state=PromotionState.PAPER,
            passed=True,
            metrics={"ic": 0.08},
            drift={"shadow_live": 0.01},
            rollback_target="1.9.0",
        )
    )

    assert await artifact_repo.get_artifact(artifact.artifact_id) == artifact
    assert (await artifact_repo.evaluate_alpha("xsec", as_of=as_of)).passed

    risk_repo = InMemoryPortfolioRiskModelRepository()
    risk_model = PortfolioRiskModel(
        model_id=uuid.uuid4(),
        as_of=as_of,
        covariance={},
        factor_exposures={},
    )
    await risk_repo.save_risk_model(risk_model)
    snapshot = RiskSnapshot(
        snapshot_id=uuid.uuid4(),
        strategy_run_id=uuid.uuid4(),
        as_of=as_of,
        gross_exposure=Decimal("0.5"),
        net_exposure=Decimal("0.5"),
        cvar=None,
        factor_exposures={},
        stress_results={},
        passed=True,
    )
    await risk_repo.save_risk_snapshot(snapshot)
    assert await risk_repo.latest_risk_model(as_of=as_of) == risk_model

    quality_repo = InMemoryExecutionQualityRepository()
    quality = ExecutionQualityReport(
        report_id=uuid.uuid4(),
        order_id=uuid.uuid4(),
        as_of=as_of,
        venue="IBKR_SMART",
        tactic=ExecutionTactic.PASSIVE_LIMIT,
        slippage_bps=Decimal("1.5"),
    )
    await quality_repo.save_execution_quality(quality)
    assert await quality_repo.list_execution_quality(quality.order_id) == [quality]

    evidence_repo = InMemoryProductionEvidenceRepository()
    key = OperatorApiKey(
        key_id=uuid.uuid4(),
        key_hash="hash",
        role="operator",
        created_at=as_of,
        created_by="tester",
    )
    await evidence_repo.save_api_key(key)
    assert await evidence_repo.get_api_key_by_hash("hash") == key


@pytest.mark.asyncio
async def test_order_state_store_is_idempotent_and_append_only() -> None:
    store = InMemoryOrderStateStore()
    order_id = uuid.uuid4()
    occurred_at = datetime(2026, 1, 2, tzinfo=UTC)
    event = OrderStateEvent(
        event_id=uuid.uuid4(),
        order_id=order_id,
        event_type=OrderStateEventType.CREATED,
        occurred_at=occurred_at,
        status=OrderStatus.PENDING_APPROVAL,
        idempotency_key="broker-key-1",
        payload={"quantity": 10, "notional": str(Decimal("1000"))},
    )

    await store.append(event)
    await store.append(event)

    assert await store.list_events(order_id) == [event]
    assert await store.latest(order_id) == event
