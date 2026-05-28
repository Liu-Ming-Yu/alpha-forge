"""Tests for the machine-owned paper-soak generator."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from quant_platform.config import (
    BrokerSettings,
    PlatformSettings,
    StorageSettings,
)
from quant_platform.core.domain.production import (
    BrokerHealthObservation,
    BrokerSmokeObservation,
    ForecastEvidence,
    NavSnapshot,
    PaperLifecycleObservation,
    SignalGateStatus,
)
from quant_platform.services.governance_service import paper_soak

if TYPE_CHECKING:
    from pathlib import Path


def _settings(tmp_path: Path) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        broker=BrokerSettings(paper_trading=True),
        storage=StorageSettings(
            postgres_dsn="",
            redis_url="",
            event_bus_backend="in_memory",
            object_store_root=str(tmp_path / "parquet"),
        ),
    )


@dataclass
class _Repo:
    health: BrokerHealthObservation | None = None
    smoke: BrokerSmokeObservation | None = None
    lifecycle: PaperLifecycleObservation | None = None
    nav: NavSnapshot | None = None

    async def latest_broker_health(self) -> BrokerHealthObservation | None:
        return self.health

    async def latest_broker_smoke(self) -> BrokerSmokeObservation | None:
        return self.smoke

    async def latest_paper_lifecycle(self) -> PaperLifecycleObservation | None:
        return self.lifecycle

    async def list_nav_snapshots(self, _run_id: uuid.UUID, *, limit: int = 252):
        return [self.nav] if self.nav is not None else []

    async def forecast_evidence(
        self,
        source: str,
        *,
        model_version: str | None = None,
        as_of: datetime,
        stale_after_hours: int = 24,
        min_confidence: float = 0.0,
        limit: int = 500,
    ) -> ForecastEvidence:
        return ForecastEvidence(
            source=source,
            model_version=model_version or "",
            as_of=as_of,
            horizon="",
            observations=0,
            mean_confidence=0.0,
            latest_prediction_at=None,
            stale_after=timedelta(hours=stale_after_hours),
            blockers=("no prediction evidence recorded",),
        )


def _signal_status(passed: bool, as_of: datetime) -> SignalGateStatus:
    return SignalGateStatus(
        signal_name="xsec",
        signal_type="classical",
        as_of=as_of,
        observations=30,
        rolling_ic=0.08 if passed else -0.05,
        negative_streak=0 if passed else 4,
        max_drawdown=-0.02,
        max_turnover=0.18,
        min_observations=20,
        min_ic=0.05,
        max_negative_streak=3,
        drawdown_limit=-0.10,
        turnover_limit=1.0,
    )


@pytest.mark.asyncio
async def test_paper_soak_missing_evidence_marks_sections_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        paper_soak,
        "build_performance_repository",
        lambda _dsn: _Repo(),
    )

    payload = await paper_soak.build_paper_soak_report(
        _settings(tmp_path),
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        strategy_run_id=uuid.uuid4(),
    )
    assert payload["broker_health"]["passed"] is False
    assert payload["lifecycle_result"]["passed"] is False
    assert payload["nav_snapshot"]["passed"] is False
    assert payload["data_health"]["passed"] is False
    assert payload["signal_gate"]["passed"] is False
    assert payload["order_latency"]["passed"] is False
    assert payload["reconciliation"]["drift_detected"] is False


@pytest.mark.asyncio
async def test_paper_soak_passes_with_fresh_persisted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    run_id = uuid.uuid4()
    repo = _Repo(
        health=BrokerHealthObservation(
            observed_at=as_of - timedelta(minutes=5),
            status="connected",
            latency_ms=8.0,
            last_heartbeat_at=as_of - timedelta(seconds=30),
        ),
        lifecycle=PaperLifecycleObservation(
            observed_at=as_of - timedelta(minutes=10),
            status="passed",
            host="127.0.0.1",
            port=4002,
            client_id=42,
            instrument_id=uuid.uuid4(),
            broker_order_id="bo-1",
            max_notional_usd=Decimal("100"),
            limit_price=Decimal("1"),
            quantity=1,
            ack_status="ok",
            cancel_status="ok",
            stale_open_order_count=0,
        ),
        nav=NavSnapshot(
            snapshot_id=uuid.uuid4(),
            strategy_run_id=run_id,
            as_of=as_of - timedelta(minutes=15),
            net_asset_value=Decimal("100000"),
            gross_exposure=Decimal("60000"),
            cash=Decimal("40000"),
        ),
    )
    monkeypatch.setattr(paper_soak, "build_performance_repository", lambda _dsn: repo)

    async def _signal(_settings, *, signal_name, signal_type, as_of):
        return _signal_status(passed=True, as_of=as_of)

    monkeypatch.setattr(paper_soak, "signal_gate_status", _signal)

    payload = await paper_soak.build_paper_soak_report(
        _settings(tmp_path),
        as_of=as_of,
        strategy_run_id=run_id,
        signal_name="xsec",
    )

    assert payload["broker_health"]["passed"] is True
    assert payload["lifecycle_result"]["passed"] is True
    assert payload["nav_snapshot"]["passed"] is True
    assert payload["signal_gate"]["passed"] is True
    assert payload["reconciliation"]["drift_detected"] is False
    # data_health and order_latency still need durable infra to pass.
    assert payload["data_health"]["passed"] is False
    assert payload["order_latency"]["passed"] is False


@pytest.mark.asyncio
async def test_paper_soak_marks_health_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 1, 2, tzinfo=UTC)
    monkeypatch.setattr(
        paper_soak,
        "build_performance_repository",
        lambda _dsn: _Repo(
            health=BrokerHealthObservation(
                observed_at=as_of - timedelta(days=10),
                status="connected",
                latency_ms=8.0,
                last_heartbeat_at=as_of - timedelta(days=10),
            )
        ),
    )

    payload = await paper_soak.build_paper_soak_report(
        _settings(tmp_path),
        as_of=as_of,
        strategy_run_id=uuid.uuid4(),
    )
    assert payload["broker_health"]["passed"] is False
    assert payload["broker_health"]["fresh"] is False


@pytest.mark.asyncio
async def test_paper_soak_writes_canonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        paper_soak,
        "build_performance_repository",
        lambda _dsn: _Repo(),
    )

    payload = await paper_soak.build_paper_soak_report(
        _settings(tmp_path),
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
        strategy_run_id=uuid.uuid4(),
    )
    output = tmp_path / "soak" / "soak.json"
    written = paper_soak.write_paper_soak_report(payload, output)

    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk["version"] == paper_soak.SOAK_REPORT_VERSION
    assert on_disk["broker_health"]["passed"] is False
