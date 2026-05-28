from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_platform.config import (
    ApiSettings,
    BrokerSettings,
    ExecutionSettings,
    LiquiditySettings,
    PlatformSettings,
    RegimeSettings,
    RiskSettings,
    StorageSettings,
    V2Settings,
)
from quant_platform.core.domain.production import (
    BrokerHealthObservation,
    BrokerSmokeObservation,
    PaperLifecycleObservation,
    ProductionProfile,
    RuntimeHeartbeat,
    SignalGateStatus,
)
from quant_platform.services.governance_service import readiness


def _live_settings(tmp_path) -> PlatformSettings:  # type: ignore[no-untyped-def]
    object_root = tmp_path / "parquet"
    object_root.mkdir()
    return PlatformSettings(
        _env_file=None,
        broker=BrokerSettings(paper_trading=False),
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
            require_security_master=True,
            require_feature_datasets=True,
            require_event_sourced_oms=True,
            require_dataset_quorum=True,
            third_eod_vendor="third-party-eod",
            readiness_snapshot_required=True,
        ),
    )


def _contracts() -> dict[uuid.UUID, dict[str, object]]:
    return {
        uuid.uuid4(): {
            "symbol": "AAPL",
            "exchange": "SMART",
            "con_id": 265598,
            "sector": "Information Technology",
            "adv_shares_20d": 50_000_000,
            "last_close": 190,
        }
    }


class _Repo:
    def __init__(
        self,
        *,
        heartbeat: RuntimeHeartbeat | None = None,
        broker: BrokerHealthObservation | None = None,
        smoke: BrokerSmokeObservation | None = None,
        lifecycle: PaperLifecycleObservation | None = None,
    ) -> None:
        self._heartbeat = heartbeat
        self._broker = broker
        self._smoke = smoke
        self._lifecycle = lifecycle

    async def latest_runtime_heartbeat(self, _component: str) -> RuntimeHeartbeat | None:
        return self._heartbeat

    async def latest_broker_health(self) -> BrokerHealthObservation | None:
        return self._broker

    async def latest_broker_smoke(self) -> BrokerSmokeObservation | None:
        return self._smoke

    async def latest_paper_lifecycle(self) -> PaperLifecycleObservation | None:
        return self._lifecycle


def _soak_payload(as_of: datetime) -> str:
    import json

    return json.dumps(
        {
            "generated_at": as_of.isoformat(),
            "broker_health": {"passed": True},
            "lifecycle_result": {"passed": True},
            "nav_snapshot": {"net_asset_value": "100000"},
            "data_health": {"passed": True},
            "signal_gate": {"passed": True},
            "prediction_quality": [],
            "reconciliation": {"drift_detected": False},
            "order_latency": {"p95_ms": 25.0},
        }
    )


@pytest.mark.asyncio
async def test_live_readiness_requires_broker_signal_soak_and_backup(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        readiness,
        "build_performance_repository",
        lambda _dsn: _Repo(),
    )

    report = await readiness.build_readiness_report(
        _live_settings(tmp_path),
        profile=ProductionProfile.LIVE,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        instrument_contracts=_contracts(),
        broker_checked=False,
    )

    failed = {check.name for check in report.failures}
    assert "broker_check_requested" in failed
    assert "signal_gate_passed" in failed
    assert "paper_smoke_persisted" not in failed
    assert "broker_smoke_persisted" in failed
    assert "paper_lifecycle_persisted" in failed
    assert "paper_soak_report_valid" in failed
    assert "backup_restore_manifest_present" in failed
    assert not readiness.readiness_payload(report)["passed"]


@pytest.mark.asyncio
async def test_readiness_rejects_malformed_paper_soak(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    soak = tmp_path / "paper-soak.json"
    soak.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        readiness,
        "build_performance_repository",
        lambda _dsn: _Repo(),
    )

    report = await readiness.build_readiness_report(
        _live_settings(tmp_path),
        profile=ProductionProfile.LIVE,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        broker_checked=False,
    )

    failed = {check.name for check in report.failures}
    assert "paper_soak_report_valid" in failed


@pytest.mark.asyncio
async def test_live_readiness_passes_with_required_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    soak = tmp_path / "paper-soak.json"
    backup = tmp_path / "backup.json"
    soak.write_text(_soak_payload(as_of), encoding="utf-8")
    backup.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        readiness,
        "build_performance_repository",
        lambda _dsn: _Repo(
            heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
            broker=BrokerHealthObservation(
                observed_at=as_of,
                status="connected",
                latency_ms=5.0,
            ),
            smoke=BrokerSmokeObservation(
                observed_at=as_of,
                status="connected",
                host="host.docker.internal",
                port=4002,
                client_id=991,
                latency_ms=5.0,
                account_status="ok",
                positions_status="ok",
                open_orders_status="ok",
            ),
            lifecycle=PaperLifecycleObservation(
                observed_at=as_of,
                status="passed",
                host="host.docker.internal",
                port=4002,
                client_id=992,
                instrument_id=uuid.uuid4(),
                broker_order_id="100",
                max_notional_usd=Decimal("100"),
                limit_price=Decimal("50"),
                quantity=1,
                ack_status="ok",
                cancel_status="ok",
                stale_open_order_count=0,
            ),
        ),
    )
    signal = SignalGateStatus(
        signal_name="xsec",
        signal_type="classical",
        as_of=as_of,
        observations=30,
        rolling_ic=0.08,
        negative_streak=0,
        max_drawdown=-0.02,
        max_turnover=0.2,
        min_observations=20,
        min_ic=0.05,
        max_negative_streak=3,
        drawdown_limit=-0.10,
        turnover_limit=1.0,
    )

    report = await readiness.build_readiness_report(
        _live_settings(tmp_path),
        profile=ProductionProfile.LIVE,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        backup_manifest=backup,
        signal_status=signal,
        broker_checked=True,
    )

    assert report.passed
    assert readiness.readiness_payload(report)["state"] == "ready"
