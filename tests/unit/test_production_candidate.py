"""Unit tests for the aggregated production-candidate gate.

Covers the new ``services.governance_service.production_candidate`` module.
The tests build a minimal :class:`PlatformSettings` aligned with live-grade
settings, monkey-patch the performance repository factory, and write a
campaign manifest under the configured ``object_store_root`` so the gate
sees real on-disk evidence.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from quant_platform.application.features.admission import ordered_feature_schema_hash
from quant_platform.config import (
    AlphaSettings,
    ApiSettings,
    BrokerSettings,
    ExecutionSettings,
    LiquiditySettings,
    LLMSettings,
    PlatformSettings,
    ProductionSettings,
    RegimeSettings,
    RiskSettings,
    StorageSettings,
    V2Settings,
)
from quant_platform.core.domain.production import (
    BrokerHealthObservation,
    BrokerSmokeObservation,
    ForecastEvidence,
    PaperLifecycleObservation,
    PreflightCheck,
    ProductionProfile,
    PromotionMode,
    ReadinessState,
    RuntimeHeartbeat,
    ShadowPaperParityStatus,
    SignalGateStatus,
)
from quant_platform.services.governance_service import production_candidate, readiness
from quant_platform.services.governance_service.production_candidate.diagnostics import (
    render_production_candidate_diagnostics,
)
from quant_platform.services.research_service.text.model_manifest import (
    write_text_model_manifest,
)


def _live_settings(tmp_path: Path) -> PlatformSettings:
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
        production=ProductionSettings(
            text_gate_min_observations=20,
            text_gate_min_ic=0.05,
            text_gate_max_negative_streak=3,
            signal_gate_max_drawdown=-0.10,
            signal_gate_max_turnover=1.0,
        ),
        alpha=AlphaSettings(
            ensemble_mode="paper",
            source_weights={"classical": 0.70, "xgboost": 0.25, "text": 0.05},
            paper_max_non_classical_weight=0.30,
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


def _paper_text_settings(tmp_path: Path) -> PlatformSettings:
    base = _live_settings(tmp_path)
    return base.model_copy(
        update={
            "alpha": AlphaSettings(
                ensemble_mode="paper",
                source_weights={
                    "classical": 0.95,
                    "text": 0.05,
                    "xgboost": 0.0,
                    "event": 0.0,
                    "intraday": 0.0,
                },
                paper_max_non_classical_weight=0.05,
                fail_closed_on_promoted_source_error=True,
                require_promotion_gate=True,
            )
        }
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
    """Minimal performance repository stub used by readiness + signal gate."""

    def __init__(
        self,
        *,
        heartbeat: RuntimeHeartbeat | None = None,
        broker: BrokerHealthObservation | None = None,
        smoke: BrokerSmokeObservation | None = None,
        lifecycle: PaperLifecycleObservation | None = None,
        signal_statuses: dict[tuple[str, str], SignalGateStatus] | None = None,
        forecast_evidence: dict[str, ForecastEvidence] | None = None,
        parity_status: ShadowPaperParityStatus | None = None,
    ) -> None:
        self._heartbeat = heartbeat
        self._broker = broker
        self._smoke = smoke
        self._lifecycle = lifecycle
        self._signal_statuses = signal_statuses or {}
        self._forecast_evidence = forecast_evidence or {}
        self._parity_status = parity_status

    async def latest_runtime_heartbeat(self, _component: str) -> RuntimeHeartbeat | None:
        return self._heartbeat

    async def latest_broker_health(self) -> BrokerHealthObservation | None:
        return self._broker

    async def latest_broker_smoke(self) -> BrokerSmokeObservation | None:
        return self._smoke

    async def latest_paper_lifecycle(self) -> PaperLifecycleObservation | None:
        return self._lifecycle

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
        existing = self._signal_statuses.get((signal_name, signal_type))
        if existing is not None:
            return existing
        return SignalGateStatus(
            signal_name=signal_name,
            signal_type=signal_type,
            as_of=as_of,
            observations=0,
            rolling_ic=0.0,
            negative_streak=10,
            max_drawdown=0.0,
            max_turnover=0.0,
            min_observations=min_observations,
            min_ic=min_ic,
            max_negative_streak=max_negative_streak,
            drawdown_limit=drawdown_limit,
            turnover_limit=turnover_limit,
        )

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
        existing = self._forecast_evidence.get(source)
        if existing is not None:
            return existing
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

    async def shadow_paper_parity_status(
        self,
        signal_name: str,
        signal_type: str,
        *,
        as_of: datetime,
        min_trading_days: int = 20,
        max_target_weight_diff_bps: float = 1.0,
    ) -> ShadowPaperParityStatus:
        if self._parity_status is not None:
            return self._parity_status
        return ShadowPaperParityStatus(
            signal_name=signal_name,
            signal_type=signal_type,
            as_of=as_of,
            observations=0,
            trading_days=0,
            min_trading_days=min_trading_days,
            max_target_weight_diff_bps=0.0,
            max_allowed_target_weight_diff_bps=max_target_weight_diff_bps,
            missing_instruments=0,
            order_side_mismatches=0,
        )


def _green_signal_status(
    *,
    signal_name: str,
    signal_type: str,
    as_of: datetime,
) -> SignalGateStatus:
    return SignalGateStatus(
        signal_name=signal_name,
        signal_type=signal_type,
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


def _green_forecast_evidence(
    source: str,
    as_of: datetime,
    *,
    feature_schema_hashes: tuple[str, ...] = (),
) -> ForecastEvidence:
    return ForecastEvidence(
        source=source,
        model_version=f"{source}-v1",
        as_of=as_of,
        horizon="21d",
        observations=25,
        mean_confidence=0.72,
        latest_prediction_at=as_of,
        stale_after=timedelta(hours=24),
        calibration_buckets=("passive|0.5-2pct_adv|tight_spread|fresh|limit|open",),
        feature_schema_hashes=feature_schema_hashes,
    )


def _write_campaign_manifest(
    settings: PlatformSettings,
    *,
    created_at: datetime,
    passed: bool = True,
    paper_source_weights: dict[str, float] | None = None,
    eligibility_failed: tuple[str, ...] = (),
) -> Path:
    root = Path(settings.storage.object_store_root) / "research" / "walk_forward" / "test-run"
    root.mkdir(parents=True, exist_ok=True)
    weights = paper_source_weights or {"classical": 0.70, "xgboost": 0.25, "text": 0.05}
    payload: dict[str, object] = {
        "run_id": str(uuid.uuid4()),
        "created_at": created_at.isoformat(),
        "model_version": "test",
        "feature_set_version": "v1",
        "passed": passed,
        "metrics": {"oos_rolling_ic": 0.06},
        "eligibility": {
            "passed": passed,
            "checks": [
                {"name": "oos_rolling_ic", "passed": "oos_rolling_ic" not in eligibility_failed},
            ],
        },
        "artifacts": {},
        "selected_weights": {"factor_a": 1.0},
        "paper_source_weights": weights,
        "git_commit": "abcdef",
        "next_allowed_paper_mode": "paper_ensemble" if passed else "shadow_only",
        "command": "test",
    }
    path = root / "campaign_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_live_text_artifacts(
    settings: PlatformSettings,
    tmp_path: Path,
    as_of: datetime,
    *,
    audit_state: str = "live",
) -> Path:
    feature_name = "live_text_alpha"
    feature_set = "text-live-v1"
    card_dir = tmp_path / "cards"
    card_dir.mkdir(exist_ok=True)
    (card_dir / f"{feature_name}.json").write_text(
        json.dumps({"feature": feature_name, "state": "live"}, sort_keys=True),
        encoding="utf-8",
    )
    audit_dir = (
        Path(settings.storage.object_store_root)
        / "research"
        / "feature_audits"
        / feature_name
        / feature_set
        / str(uuid.uuid4())
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "audit_id": str(uuid.uuid4()),
        "generated_at": as_of.isoformat(),
        "sample_start": (as_of - timedelta(days=30)).isoformat(),
        "sample_end": as_of.isoformat(),
        "feature_set_version": feature_set,
        "feature": {"name": feature_name, "version": feature_set, "state": audit_state},
        "passed": True,
        "metrics": {"rolling_ic": 0.08},
        "gate_results": {"ic_gate": True},
        "schema_hash": ordered_feature_schema_hash((feature_name,)),
        "blockers": [],
    }
    (audit_dir / "feature_audit_manifest.json").write_text(
        json.dumps(audit_payload),
        encoding="utf-8",
    )
    extraction = tmp_path / "text_extraction_manifest.json"
    extraction.write_text("{}", encoding="utf-8")
    campaign = tmp_path / "text_campaign_manifest.json"
    campaign.write_text("{}", encoding="utf-8")
    return write_text_model_manifest(
        output_root=Path(settings.storage.object_store_root),
        model_version="text-v1",
        feature_set_version=feature_set,
        feature_names=(feature_name,),
        weights={feature_name: 1.0},
        provider=settings.llm.provider,
        llm_model=settings.llm.model,
        prompt_version=settings.llm.text_prompt_version,
        campaign_manifest=campaign,
        source_data_manifest=None,
        extraction_manifest=extraction,
        feature_card_dir=card_dir,
        created_at=as_of,
    )


def _live_llm_settings(
    tmp_path: Path,
    as_of: datetime,
    *,
    rehearsal: bool = False,
) -> PlatformSettings:
    feature_name = "live_text_alpha"
    feature_set = "text-live-v1"
    base = _live_settings(tmp_path)
    settings = base.model_copy(
        update={
            "broker": BrokerSettings(paper_trading=True) if rehearsal else base.broker,
            "alpha": AlphaSettings(
                ensemble_mode="live",
                source_weights={
                    "classical": 0.99,
                    "text": 0.01,
                    "xgboost": 0.0,
                    "event": 0.0,
                    "intraday": 0.0,
                },
                max_non_classical_weight=0.01,
                live_ramp_initial=Decimal("0.01"),
                paper_max_non_classical_weight=0.30,
            ),
            "llm": LLMSettings(
                live_mode_enabled=True,
                live_rehearsal_enabled=rehearsal,
                shadow_mode_enabled=True,
                text_feature_weights={feature_name: 1.0},
                text_feature_versions={feature_name: feature_set},
                text_feature_set_version=feature_set,
                text_feature_card_dir=str(tmp_path / "cards"),
            ),
        }
    )
    manifest = _write_live_text_artifacts(
        settings,
        tmp_path,
        as_of,
        audit_state="paper" if rehearsal else "live",
    )
    return settings.model_copy(
        update={"llm": settings.llm.model_copy(update={"text_model_manifest": str(manifest)})}
    )


def _green_parity(as_of: datetime, *, passed: bool = True) -> ShadowPaperParityStatus:
    return ShadowPaperParityStatus(
        signal_name="text",
        signal_type="text",
        as_of=as_of,
        observations=20,
        trading_days=20 if passed else 19,
        min_trading_days=20,
        max_target_weight_diff_bps=0.5,
        max_allowed_target_weight_diff_bps=1.0,
        missing_instruments=0,
        order_side_mismatches=0,
    )


async def _passing_quorum_check(
    _settings: PlatformSettings,
    *,
    as_of: datetime,
    profile: ProductionProfile,
    dataset_kind: str = "bars_eod",
) -> PreflightCheck:
    return PreflightCheck(
        name="v2_dataset_quorum_evidence_fresh",
        passed=True,
        detail=f"stub as_of={as_of.isoformat()} profile={profile.value} dataset={dataset_kind}",
    )


def _write_soak_and_backup(tmp_path: Path, as_of: datetime) -> tuple[Path, Path]:
    soak = tmp_path / f"soak-{uuid.uuid4()}.json"
    backup = tmp_path / f"backup-{uuid.uuid4()}.json"
    soak.write_text(
        json.dumps(
            {
                "generated_at": as_of.isoformat(),
                "broker_health": {"passed": True},
                "lifecycle_result": {"passed": True},
                "nav_snapshot": {"net_asset_value": "100000"},
                "data_health": {"passed": True},
                "signal_gate": {"passed": True},
                "prediction_quality": [{"source": "text", "passed": True}],
                "reconciliation": {"drift_detected": False},
                "order_latency": {"p95_ms": 25.0},
            }
        ),
        encoding="utf-8",
    )
    backup.write_text("{}", encoding="utf-8")
    return soak, backup


def _patch_repos(
    monkeypatch: pytest.MonkeyPatch,
    repo: _Repo,
) -> None:
    monkeypatch.setattr(readiness, "build_performance_repository", lambda _dsn: repo)
    monkeypatch.setattr(production_candidate, "build_performance_repository", lambda _dsn: repo)
    monkeypatch.setattr(
        "quant_platform.services.governance_service.gates.signal_gate.build_performance_repository",
        lambda _dsn: repo,
    )


@pytest.mark.asyncio
async def test_paper_candidate_passes_with_full_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _write_campaign_manifest(settings, created_at=as_of)

    statuses = {
        ("classical", "classical"): _green_signal_status(
            signal_name="classical", signal_type="classical", as_of=as_of
        ),
        ("test", "xgboost"): _green_signal_status(
            signal_name="test", signal_type="xgboost", as_of=as_of
        ),
        ("xgboost", "xgboost"): _green_signal_status(
            signal_name="xgboost", signal_type="xgboost", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        broker=BrokerHealthObservation(observed_at=as_of, status="connected", latency_ms=4.0),
        smoke=BrokerSmokeObservation(
            observed_at=as_of,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            latency_ms=4.0,
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
        signal_statuses=statuses,
        forecast_evidence={
            "xgboost": _green_forecast_evidence("xgboost", as_of),
            "text": _green_forecast_evidence("text", as_of),
        },
    )
    _patch_repos(monkeypatch, repo)

    soak = tmp_path / "soak.json"
    backup = tmp_path / "backup.json"
    soak.write_text(
        json.dumps(
            {
                "generated_at": as_of.isoformat(),
                "broker_health": {"passed": True},
                "lifecycle_result": {"passed": True},
                "nav_snapshot": {"net_asset_value": "100000"},
                "data_health": {"passed": True},
                "signal_gate": {"passed": True},
                "prediction_quality": [
                    {"source": "xgboost", "passed": True},
                    {"source": "text", "passed": True},
                ],
                "reconciliation": {"drift_detected": False},
                "order_latency": {"p95_ms": 25.0},
            }
        ),
        encoding="utf-8",
    )
    backup.write_text("{}", encoding="utf-8")

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        backup_manifest=backup,
        broker_checked=True,
    )

    assert report.passed
    assert report.state == ReadinessState.READY
    assert report.next_allowed_mode == PromotionMode.PAPER_ENSEMBLE
    assert report.promotion_blockers == ()
    payload = production_candidate.production_candidate_payload(report)
    assert payload["passed"] is True
    assert payload["next_allowed_mode"] == "paper_ensemble"
    assert payload["campaign_manifest"] is not None
    assert {"classical", "xgboost", "text"} - {
        gate["signal_name"] for gate in payload["signal_gates"]
    } == {"classical"}, "classical is folded into the readiness gate, not the multi-source list"
    assert payload["campaign_manifest_path"]
    assert payload["representative_signal_gate"]["signal_name"] == "classical"


@pytest.mark.asyncio
async def test_paper_target_classical_text_promotes_when_all_evidence_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _paper_text_settings(tmp_path)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={
            "classical": 0.95,
            "text": 0.05,
            "xgboost": 0.0,
            "event": 0.0,
            "intraday": 0.0,
        },
    )

    statuses = {
        ("classical", "classical"): _green_signal_status(
            signal_name="classical", signal_type="classical", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        broker=BrokerHealthObservation(observed_at=as_of, status="connected", latency_ms=4.0),
        smoke=BrokerSmokeObservation(
            observed_at=as_of,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            latency_ms=4.0,
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
        signal_statuses=statuses,
        forecast_evidence={"text": _green_forecast_evidence("text", as_of)},
    )
    _patch_repos(monkeypatch, repo)
    soak, backup = _write_soak_and_backup(tmp_path, as_of)

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        backup_manifest=backup,
        broker_checked=True,
    )

    assert report.passed is True
    assert report.next_allowed_mode == PromotionMode.PAPER_ENSEMBLE
    assert report.promotion_blockers == ()
    assert report.representative_signal_gate_status is not None
    assert report.representative_signal_gate_status.signal_name == "classical"
    assert {status.signal_name for status in report.signal_gate_statuses} == {"text"}


@pytest.mark.asyncio
async def test_diagnostics_show_text_blockers_and_zero_weight_sources_stay_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _paper_text_settings(tmp_path)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={
            "classical": 0.95,
            "text": 0.05,
            "xgboost": 0.0,
            "event": 0.0,
            "intraday": 0.0,
        },
    )
    _patch_repos(monkeypatch, _Repo())

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=False,
    )
    rendered = render_production_candidate_diagnostics(report)
    check_names = {check.name for check in report.checks}

    assert "next_allowed_mode=shadow_only" in rendered
    assert "signal_gate_text_passed" in rendered
    assert "prediction_evidence_text_fresh" in rendered
    assert "signal_gate_xgboost_passed" not in check_names
    assert "signal_gate_event_passed" not in check_names
    assert "signal_gate_intraday_passed" not in check_names
    assert "prediction_evidence_xgboost_fresh" not in check_names
    assert "prediction_evidence_event_fresh" not in check_names
    assert "prediction_evidence_intraday_fresh" not in check_names


@pytest.mark.asyncio
async def test_paper_candidate_can_override_primary_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _write_campaign_manifest(settings, created_at=as_of)
    statuses = {
        ("test", "xgboost"): _green_signal_status(
            signal_name="test", signal_type="xgboost", as_of=as_of
        ),
        ("xgboost", "xgboost"): _green_signal_status(
            signal_name="xgboost", signal_type="xgboost", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        broker=BrokerHealthObservation(observed_at=as_of, status="connected", latency_ms=4.0),
        smoke=BrokerSmokeObservation(
            observed_at=as_of,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            latency_ms=4.0,
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
        signal_statuses=statuses,
        forecast_evidence={
            "xgboost": _green_forecast_evidence("xgboost", as_of),
            "text": _green_forecast_evidence("text", as_of),
        },
    )
    _patch_repos(monkeypatch, repo)
    soak, backup = _write_soak_and_backup(tmp_path, as_of)

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        backup_manifest=backup,
        broker_checked=True,
        primary_signal_name="test",
        primary_signal_type="xgboost",
    )

    assert report.passed
    signal_gate = next(check for check in report.checks if check.name == "signal_gate_passed")
    assert "xgboost/test state=ready" in signal_gate.detail


@pytest.mark.asyncio
async def test_classical_only_candidate_still_requires_classical_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path).model_copy(
        update={
            "alpha": AlphaSettings(
                ensemble_mode="paper",
                source_weights={"classical": 1.0},
            )
        }
    )
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 1.0},
    )
    _patch_repos(
        monkeypatch,
        _Repo(heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok")),
    )

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=False,
    )

    signal_gate = next(check for check in report.checks if check.name == "signal_gate_passed")
    assert not signal_gate.passed
    assert "classical/classical state=halted" in signal_gate.detail


@pytest.mark.asyncio
async def test_missing_campaign_manifest_blocks_paper_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _patch_repos(monkeypatch, _Repo())

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=False,
    )

    assert not report.passed
    assert report.next_allowed_mode == PromotionMode.SHADOW_ONLY
    assert "research_campaign_manifest_present" in report.promotion_blockers


@pytest.mark.asyncio
async def test_failing_eligibility_drops_to_shadow_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        passed=False,
        eligibility_failed=("oos_rolling_ic",),
    )
    _patch_repos(monkeypatch, _Repo())

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=False,
    )

    assert "research_campaign_eligibility_passed" in report.promotion_blockers
    assert report.next_allowed_mode == PromotionMode.SHADOW_ONLY


@pytest.mark.asyncio
async def test_non_classical_weight_above_cap_blocks_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 0.40, "xgboost": 0.40, "text": 0.20},
    )
    _patch_repos(monkeypatch, _Repo())

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=False,
    )

    assert "research_campaign_paper_weights_within_cap" in report.promotion_blockers


@pytest.mark.asyncio
async def test_live_requires_v2_account_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "v2": settings.v2.model_copy(update={"account_orchestrator_enabled": False}),
        }
    )
    _write_campaign_manifest(settings, created_at=as_of)
    _patch_repos(monkeypatch, _Repo())

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.LIVE,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=True,
    )

    assert "v2_account_orchestrator_is_live_submitter" in report.promotion_blockers
    assert report.next_allowed_mode == PromotionMode.SHADOW_ONLY


@pytest.mark.asyncio
async def test_signal_source_filter_overrides_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _write_campaign_manifest(settings, created_at=as_of)
    _patch_repos(monkeypatch, _Repo())

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        signal_sources=["classical"],  # only classical → no xgboost/text checks
        broker_checked=False,
    )

    check_names = {check.name for check in report.checks}
    assert "signal_gate_xgboost_passed" not in check_names
    assert "signal_gate_text_passed" not in check_names


@pytest.mark.asyncio
async def test_live_non_classical_source_requires_fresh_prediction_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _write_campaign_manifest(settings, created_at=as_of)
    statuses = {
        ("xgboost", "xgboost"): _green_signal_status(
            signal_name="xgboost", signal_type="xgboost", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    _patch_repos(monkeypatch, _Repo(signal_statuses=statuses))

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.LIVE,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=True,
    )

    assert "prediction_evidence_xgboost_fresh" in report.promotion_blockers
    assert "prediction_evidence_text_fresh" in report.promotion_blockers


@pytest.mark.asyncio
async def test_paper_text_source_requires_fresh_prediction_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_settings(tmp_path)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 0.70, "xgboost": 0.25, "text": 0.05},
    )
    statuses = {
        ("xgboost", "xgboost"): _green_signal_status(
            signal_name="xgboost", signal_type="xgboost", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    _patch_repos(
        monkeypatch,
        _Repo(
            signal_statuses=statuses,
            forecast_evidence={"xgboost": _green_forecast_evidence("xgboost", as_of)},
        ),
    )

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.PAPER,
        as_of=as_of,
        instrument_contracts=_contracts(),
        broker_checked=False,
    )

    assert "prediction_evidence_text_fresh" in report.promotion_blockers
    assert report.next_allowed_mode == PromotionMode.SHADOW_ONLY


@pytest.mark.asyncio
async def test_live_llm_candidate_passes_with_manifest_schema_and_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_llm_settings(tmp_path, as_of)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 0.99, "text": 0.01},
    )
    schema_hash = ordered_feature_schema_hash(tuple(settings.llm.text_feature_weights))
    statuses = {
        ("classical", "classical"): _green_signal_status(
            signal_name="classical", signal_type="classical", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        broker=BrokerHealthObservation(observed_at=as_of, status="connected", latency_ms=4.0),
        smoke=BrokerSmokeObservation(
            observed_at=as_of,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            latency_ms=4.0,
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
        signal_statuses=statuses,
        forecast_evidence={
            "text": _green_forecast_evidence(
                "text",
                as_of,
                feature_schema_hashes=(schema_hash,),
            )
        },
        parity_status=_green_parity(as_of),
    )
    _patch_repos(monkeypatch, repo)
    monkeypatch.setattr(
        production_candidate,
        "_v2_dataset_quorum_evidence_check",
        _passing_quorum_check,
    )
    soak, backup = _write_soak_and_backup(tmp_path, as_of)

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.LIVE,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        backup_manifest=backup,
        broker_checked=True,
    )

    assert report.passed
    assert report.next_allowed_mode == PromotionMode.LIVE_RAMP_INITIAL
    assert report.promotion_blockers == ()


@pytest.mark.asyncio
async def test_llm_live_rehearsal_candidate_passes_with_paper_audit_and_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_llm_settings(tmp_path, as_of, rehearsal=True)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 0.99, "text": 0.01},
    )
    schema_hash = ordered_feature_schema_hash(tuple(settings.llm.text_feature_weights))
    statuses = {
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        broker=BrokerHealthObservation(observed_at=as_of, status="connected", latency_ms=4.0),
        smoke=BrokerSmokeObservation(
            observed_at=as_of,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            latency_ms=4.0,
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
        signal_statuses=statuses,
        forecast_evidence={
            "text": _green_forecast_evidence(
                "text",
                as_of,
                feature_schema_hashes=(schema_hash,),
            )
        },
        parity_status=_green_parity(as_of),
    )
    _patch_repos(monkeypatch, repo)
    monkeypatch.setattr(
        production_candidate,
        "_v2_dataset_quorum_evidence_check",
        _passing_quorum_check,
    )

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.LLM_LIVE_REHEARSAL,
        as_of=as_of,
        instrument_contracts=_contracts(),
        signal_sources=["text"],
        broker_checked=True,
    )

    assert report.passed
    assert report.next_allowed_mode == PromotionMode.LLM_LIVE_REHEARSAL
    assert report.promotion_blockers == ()
    payload = production_candidate.production_candidate_payload(report)
    assert payload["profile"] == "llm_live_rehearsal"
    assert payload["next_allowed_mode"] == "llm_live_rehearsal"
    by_name = {check.name: check for check in report.checks}
    assert "minimum_state=paper" in by_name["llm_live_text_feature_audits_admitted"].detail


@pytest.mark.asyncio
async def test_llm_live_rehearsal_candidate_blocks_true_live_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_llm_settings(tmp_path, as_of, rehearsal=True)
    settings = settings.model_copy(update={"broker": BrokerSettings(paper_trading=False)})
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 0.99, "text": 0.01},
    )
    schema_hash = ordered_feature_schema_hash(tuple(settings.llm.text_feature_weights))
    statuses = {
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        signal_statuses=statuses,
        forecast_evidence={
            "text": _green_forecast_evidence(
                "text",
                as_of,
                feature_schema_hashes=(schema_hash,),
            )
        },
        parity_status=_green_parity(as_of),
    )
    _patch_repos(monkeypatch, repo)
    monkeypatch.setattr(
        production_candidate,
        "_v2_dataset_quorum_evidence_check",
        _passing_quorum_check,
    )

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.LLM_LIVE_REHEARSAL,
        as_of=as_of,
        instrument_contracts=_contracts(),
        signal_sources=["text"],
        broker_checked=True,
    )

    assert "llm_live_rehearsal_paper_broker" in report.promotion_blockers
    assert report.next_allowed_mode == PromotionMode.SHADOW_ONLY


@pytest.mark.asyncio
async def test_live_llm_candidate_blocks_missing_shadow_paper_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_llm_settings(tmp_path, as_of)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 0.99, "text": 0.01},
    )
    schema_hash = ordered_feature_schema_hash(tuple(settings.llm.text_feature_weights))
    statuses = {
        ("classical", "classical"): _green_signal_status(
            signal_name="classical", signal_type="classical", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        broker=BrokerHealthObservation(observed_at=as_of, status="connected", latency_ms=4.0),
        smoke=BrokerSmokeObservation(
            observed_at=as_of,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            latency_ms=4.0,
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
        signal_statuses=statuses,
        forecast_evidence={
            "text": _green_forecast_evidence(
                "text",
                as_of,
                feature_schema_hashes=(schema_hash,),
            )
        },
        parity_status=_green_parity(as_of, passed=False),
    )
    _patch_repos(monkeypatch, repo)
    monkeypatch.setattr(
        production_candidate,
        "_v2_dataset_quorum_evidence_check",
        _passing_quorum_check,
    )
    soak, backup = _write_soak_and_backup(tmp_path, as_of)

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.LIVE,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        backup_manifest=backup,
        broker_checked=True,
    )

    assert "llm_live_shadow_paper_parity_passed" in report.promotion_blockers
    assert report.next_allowed_mode == PromotionMode.SHADOW_ONLY


@pytest.mark.asyncio
async def test_live_llm_candidate_blocks_prediction_schema_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_of = datetime(2026, 4, 1, tzinfo=UTC)
    settings = _live_llm_settings(tmp_path, as_of)
    _write_campaign_manifest(
        settings,
        created_at=as_of,
        paper_source_weights={"classical": 0.99, "text": 0.01},
    )
    statuses = {
        ("classical", "classical"): _green_signal_status(
            signal_name="classical", signal_type="classical", as_of=as_of
        ),
        ("text", "text"): _green_signal_status(signal_name="text", signal_type="text", as_of=as_of),
    }
    repo = _Repo(
        heartbeat=RuntimeHeartbeat(component="supervisor", as_of=as_of, status="ok"),
        broker=BrokerHealthObservation(observed_at=as_of, status="connected", latency_ms=4.0),
        smoke=BrokerSmokeObservation(
            observed_at=as_of,
            status="connected",
            host="host.docker.internal",
            port=4002,
            client_id=991,
            latency_ms=4.0,
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
        signal_statuses=statuses,
        forecast_evidence={
            "text": _green_forecast_evidence(
                "text",
                as_of,
                feature_schema_hashes=("wrong",),
            )
        },
        parity_status=_green_parity(as_of),
    )
    _patch_repos(monkeypatch, repo)
    monkeypatch.setattr(
        production_candidate,
        "_v2_dataset_quorum_evidence_check",
        _passing_quorum_check,
    )
    soak, backup = _write_soak_and_backup(tmp_path, as_of)

    report = await production_candidate.build_production_candidate_report(
        settings,
        profile=ProductionProfile.LIVE,
        as_of=as_of,
        instrument_contracts=_contracts(),
        soak_report=soak,
        backup_manifest=backup,
        broker_checked=True,
    )

    assert "llm_live_prediction_schema_hash_matches" in report.promotion_blockers
    assert report.next_allowed_mode == PromotionMode.SHADOW_ONLY
