"""Static configuration checks for production preflight."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.core.domain.production import PreflightCheck, ProductionProfile

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings


def build_configuration_preflight_checks(
    settings: PlatformSettings,
    *,
    profile: ProductionProfile,
    instrument_contracts: dict[uuid.UUID, dict[str, object]],
) -> tuple[PreflightCheck, ...]:
    """Return static configuration and contract checks for a deployment profile."""
    live = profile == ProductionProfile.LIVE
    checks: list[PreflightCheck] = []

    def add(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        checks.append(PreflightCheck(name=name, passed=passed, detail=detail, severity=severity))

    add("postgres_configured", bool(settings.storage.postgres_dsn), "Postgres DSN is required")
    add("redis_configured", bool(settings.storage.redis_url), "Redis URL is required")
    add(
        "redis_streams_enabled",
        settings.storage.event_bus_backend == "redis_streams",
        "QP__STORAGE__EVENT_BUS_BACKEND must be redis_streams",
    )
    add(
        "operator_api_key",
        bool(settings.api.operator_api_key.strip()),
        "QP__API__OPERATOR_API_KEY must be set",
    )
    add(
        "operator_api_authenticated",
        not settings.api.allow_unauthenticated
        and not settings.api.acknowledge_unauthenticated_risk,
        "unauthenticated operator API escape hatch must be disabled",
    )
    add(
        "strict_adv",
        not settings.liquidity.allow_missing_profile,
        "QP__LIQUIDITY__ALLOW_MISSING_PROFILE must be false",
    )
    add(
        "sector_mapping_required",
        settings.risk.require_sector_mapping,
        "QP__RISK__REQUIRE_SECTOR_MAPPING must be true",
    )
    add(
        "trading_hours_enforced",
        settings.execution.trading_hours_enforced,
        "QP__EXECUTION__TRADING_HOURS_ENFORCED must be true",
        "error" if live else "warning",
    )
    add(
        "market_proxy_configured",
        valid_uuid(settings.regime.market_proxy_instrument_id),
        "QP__REGIME__MARKET_PROXY_INSTRUMENT_ID must be a UUID",
    )
    add(
        "market_regime_seed_required",
        settings.regime.require_seed_on_cycle,
        "QP__REGIME__REQUIRE_SEED_ON_CYCLE must be true",
        "error" if live else "warning",
    )
    add(
        "model_registry_match_required",
        settings.risk.require_registered_model_match,
        "QP__RISK__REQUIRE_REGISTERED_MODEL_MATCH must be true",
    )
    add(
        "object_store_root_valid",
        object_store_root_valid(settings.storage.object_store_root),
        "object-store root parent must exist and be writable",
    )
    add(
        "llm_live_manifest_explicit",
        (not settings.llm.live_mode_enabled) or bool(settings.llm.text_model_manifest.strip()),
        "QP__LLM__TEXT_MODEL_MANIFEST is required when QP__LLM__LIVE_MODE_ENABLED=true",
    )
    add(
        "llm_live_provider_limits_configured",
        (
            not settings.llm.live_mode_enabled
            or (
                settings.llm.replay_only_live
                and settings.llm.max_request_latency_seconds <= settings.llm.timeout_seconds
                and settings.llm.estimated_cost_per_call_usd
                <= settings.llm.max_daily_estimated_cost_usd
            )
        ),
        "live LLM mode requires replay-only extraction and bounded provider latency/cost",
    )
    add(
        "v2_enabled",
        settings.v2.enabled,
        "QP__V2__ENABLED must be true for V2 production readiness",
        "error" if live else "warning",
    )
    add(
        "v2_account_orchestrator_enabled",
        settings.v2.account_orchestrator_enabled,
        "central account orchestrator must be the only live submitter",
        "error" if live else "warning",
    )
    add(
        "v2_security_master_required",
        settings.v2.require_security_master,
        "durable point-in-time security master must be required",
        "error" if live else "warning",
    )
    add(
        "v2_feature_datasets_required",
        settings.v2.require_feature_datasets,
        "live scoring must require versioned FeatureDataset manifests",
        "error" if live else "warning",
    )
    add(
        "v2_event_sourced_oms_required",
        settings.v2.require_event_sourced_oms,
        "live execution must require event-sourced OMS state",
        "error" if live else "warning",
    )
    add(
        "v2_dataset_quorum_required",
        settings.v2.require_dataset_quorum and bool(settings.v2.third_eod_vendor.strip()),
        "third independent EOD vendor and dataset quorum must be configured",
        "error" if live else "warning",
    )
    add(
        "v2_readiness_snapshot_required",
        settings.v2.readiness_snapshot_required,
        "persisted readiness snapshots must be required for V2 live",
        "error" if live else "warning",
    )

    if live:
        add("live_contracts_present", bool(instrument_contracts), "live profile requires contracts")
        incomplete = contract_issues(instrument_contracts)
        add(
            "live_contracts_complete",
            not incomplete,
            "; ".join(incomplete[:5]) if incomplete else "all contracts complete",
        )
        add(
            "broker_live_mode_explicit",
            not settings.broker.paper_trading,
            "live profile requires QP__BROKER__PAPER_TRADING=false",
        )
        add(
            "dev_defaults_disabled",
            not settings.allow_dev_defaults,
            "QP__ALLOW_DEV_DEFAULTS must be false",
        )
    else:
        add(
            "paper_contracts_present",
            bool(instrument_contracts),
            "paper contracts are recommended",
            "warning",
        )

    return tuple(checks)


def valid_uuid(raw: str) -> bool:
    try:
        uuid.UUID(raw.strip())
    except (ValueError, AttributeError):
        return False
    return True


def object_store_root_valid(raw: str) -> bool:
    if not raw.strip():
        return False
    path = Path(raw).expanduser()
    parent = path if path.exists() and path.is_dir() else path.parent
    return parent.exists() and parent.is_dir()


def contract_issues(contracts: dict[uuid.UUID, dict[str, object]]) -> list[str]:
    issues: list[str] = []
    for instrument_id, spec in contracts.items():
        symbol = str(spec.get("symbol", "")).strip()
        exchange = str(spec.get("exchange", "")).strip()
        con_id = spec.get("con_id")
        sector = str(spec.get("sector", "")).strip()
        adv = spec.get("adv_shares_20d")
        last_close = spec.get("last_close")
        missing = []
        if not symbol:
            missing.append("symbol")
        if not exchange:
            missing.append("exchange")
        if not (isinstance(con_id, int) and con_id > 0):
            missing.append("con_id")
        if not sector:
            missing.append("sector")
        if adv is None:
            missing.append("adv_shares_20d")
        if last_close is None:
            missing.append("last_close")
        if missing:
            issues.append(f"{instrument_id}: missing {', '.join(missing)}")
    return issues
