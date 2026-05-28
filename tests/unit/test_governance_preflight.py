from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

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
from quant_platform.core.domain.production import ProductionProfile
from quant_platform.services.governance_service.preflight import evaluate_preflight

if TYPE_CHECKING:
    from pathlib import Path


def _live_settings(tmp_path: Path) -> PlatformSettings:
    return PlatformSettings(
        _env_file=None,
        broker=BrokerSettings(paper_trading=False),
        storage=StorageSettings(
            postgres_dsn="postgresql+psycopg://u:p@localhost/db",
            redis_url="redis://localhost:6379/0",
            event_bus_backend="redis_streams",
            object_store_root=str(tmp_path / "parquet"),
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


def test_live_preflight_passes_when_production_profile_is_strict(tmp_path: Path) -> None:
    report = evaluate_preflight(
        _live_settings(tmp_path),
        profile=ProductionProfile.LIVE,
        instrument_contracts=_contracts(),
    )

    assert report.passed
    assert report.failures == ()


def test_live_preflight_fails_permissive_controls(tmp_path: Path) -> None:
    settings = _live_settings(tmp_path)
    settings.liquidity.allow_missing_profile = True
    settings.api.operator_api_key = ""

    report = evaluate_preflight(
        settings,
        profile=ProductionProfile.LIVE,
        instrument_contracts=_contracts(),
    )

    failed_names = {check.name for check in report.failures}
    assert "strict_adv" in failed_names
    assert "operator_api_key" in failed_names
    assert not report.passed


def test_live_preflight_requires_complete_contract_metadata(tmp_path: Path) -> None:
    report = evaluate_preflight(
        _live_settings(tmp_path),
        profile=ProductionProfile.LIVE,
        instrument_contracts={uuid.uuid4(): {"symbol": "AAPL", "exchange": "SMART"}},
    )

    contract_check = next(
        check for check in report.checks if check.name == "live_contracts_complete"
    )
    assert not contract_check.passed
    assert "con_id" in contract_check.detail
    assert "sector" in contract_check.detail
