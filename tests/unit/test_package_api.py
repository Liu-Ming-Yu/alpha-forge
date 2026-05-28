from __future__ import annotations

from pathlib import Path

from quant_platform.application.operator import serialization as bootstrap_serialization
from quant_platform.bootstrap import broker as bootstrap_broker
from quant_platform.bootstrap import data as bootstrap_data
from quant_platform.bootstrap import engine as bootstrap_engine
from quant_platform.bootstrap import governance as bootstrap_governance
from quant_platform.bootstrap import session as bootstrap_session
from quant_platform.bootstrap import text_events as bootstrap_text_events
from quant_platform.cli.commands import data as data_commands
from quant_platform.cli.commands import governance as governance_commands
from quant_platform.cli.commands import research as research_commands
from quant_platform.infrastructure import performance
from quant_platform.services.governance_service import llm_live_startup

ROOT = Path(__file__).resolve().parents[2]


def test_performance_package_exports_stable_public_api() -> None:
    expected = {
        "InMemoryPerformanceRepository",
        "PostgresPerformanceRepository",
        "build_performance_repository",
        "build_performance_report",
        "build_shadow_paper_parity_status",
        "build_signal_gate_status",
        "build_text_gate_status",
    }

    assert set(performance.__all__) == expected


def test_performance_implementation_modules_stay_inside_package() -> None:
    infrastructure_root = ROOT / "src" / "quant_platform" / "infrastructure"

    assert sorted(path.name for path in infrastructure_root.glob("performance_*.py")) == []
    assert (infrastructure_root / "performance" / "__init__.py").is_file()


def test_llm_live_startup_package_exports_stable_public_api() -> None:
    expected = {
        "LLM_LIVE_MAX_INITIAL_CAP",
        "LLM_LIVE_STARTUP_ASSERTION_SCHEMA_VERSION",
        "assert_llm_live_startup_allowed",
        "build_llm_live_evidence_checks",
        "expected_text_feature_schema_hash",
        "llm_extraction_artifact_root",
        "llm_live_startup_assertion_path",
        "text_model_manifest_path",
        "write_llm_live_startup_assertion",
    }

    assert set(llm_live_startup.__all__) == expected


def test_cli_command_families_are_packages() -> None:
    commands_root = ROOT / "src" / "quant_platform" / "cli" / "commands"

    assert not (commands_root / "data.py").exists()
    assert sorted(path.name for path in commands_root.glob("research_*.py")) == []
    assert sorted(path.name for path in commands_root.glob("governance_*.py")) == []
    assert (commands_root / "data" / "__init__.py").is_file()
    assert (commands_root / "research" / "__init__.py").is_file()
    assert (commands_root / "governance" / "__init__.py").is_file()


def test_cli_command_packages_export_stable_public_api() -> None:
    assert set(data_commands.__all__) == {
        "register",
        "register_event_bus",
        "register_maintenance",
    }
    assert set(research_commands.__all__) == {"register"}
    assert set(governance_commands.__all__) == {
        "dataset_quorum",
        "paper_soak",
        "performance",
        "production_candidate",
        "readiness",
        "register",
        "signal_gate",
        "simulator_calibration",
        "text_gate",
    }


def test_bootstrap_governance_package_exports_stable_public_api() -> None:
    expected = {
        "_json_default",
        "alpha_command",
        "dataset_quorum_command",
        "paper_soak_report_command",
        "performance_heartbeat_command",
        "performance_report_command",
        "performance_snapshot_command",
        "preflight_payload",
        "production_candidate_diagnostics_for_cli",
        "production_candidate_payload_for_cli",
        "readiness_payload_for_cli",
        "signal_gate_command",
        "simulator_calibration_command",
        "smoke_command",
        "text_gate_command",
    }

    assert set(bootstrap_governance.__all__) == expected


def test_bootstrap_governance_implementation_modules_stay_inside_package() -> None:
    bootstrap_root = ROOT / "src" / "quant_platform" / "bootstrap"

    assert sorted(path.name for path in bootstrap_root.glob("governance_*.py")) == []
    assert (bootstrap_root / "governance" / "__init__.py").is_file()


def test_bootstrap_data_broker_engine_packages_export_stable_public_api() -> None:
    assert set(bootstrap_data.__all__) == {
        "compute_features",
        "data_health_payload_for_contracts",
        "ingest_bars",
        "load_intraday_feature_series",
        "maintain_data",
        "reprocess_corporate_actions",
        "run_intraday_command",
    }
    assert set(bootstrap_broker.__all__) == {
        "broker_gate_settings",
        "broker_health",
        "broker_smoke_from_report",
        "classify_broker_probe_failure",
        "ib_gateway_smoke",
        "ib_paper_lifecycle",
        "paper_lifecycle_limit_price",
        "sweep_dead_letters",
    }
    assert set(bootstrap_engine.__all__) == {
        "latest_contract_market_prices",
        "load_budgets",
        "run_cycle_once",
        "run_engine_loop",
        "run_multi_engine_v2",
        "supervise_engine",
    }
    assert set(bootstrap_serialization.__all__) == {"_json_default"}
    assert set(bootstrap_text_events.__all__) == {
        "extract_text_features",
        "ingest_sec_text_events",
        "text_events_command",
    }
    assert set(bootstrap_session.__all__) == {
        "CycleResult",
        "Session",
        "SessionDrawdownGuard",
        "build_session",
        "create_ib_paper_session_impl",
        "create_live_session_impl",
        "create_paper_session_impl",
        "durable_kill_switch",
        "hydrate_session_state",
        "maybe_attach_v2_orchestrator",
        "model_registry_preflight",
        "record_nav_snapshot",
        "run_strategy_cycle",
        "run_strategy_cycle_unlocked",
        "strategy_cycle_lock",
    }


def test_bootstrap_operation_families_are_packages() -> None:
    bootstrap_root = ROOT / "src" / "quant_platform" / "bootstrap"

    assert sorted(path.name for path in bootstrap_root.glob("data_*.py")) == []
    assert sorted(path.name for path in bootstrap_root.glob("broker_*.py")) == []
    assert sorted(path.name for path in bootstrap_root.glob("engine_*_ops.py")) == []
    assert sorted(path.name for path in bootstrap_root.glob("session_*.py")) == []
    assert sorted(path.name for path in bootstrap_root.glob("text_event_*.py")) == []
    assert not (bootstrap_root / "engine_ops.py").exists()
    assert (bootstrap_root / "data" / "__init__.py").is_file()
    assert (bootstrap_root / "broker" / "__init__.py").is_file()
    assert (bootstrap_root / "engine" / "__init__.py").is_file()
    assert (bootstrap_root / "session" / "__init__.py").is_file()
    assert (bootstrap_root / "text_events" / "__init__.py").is_file()
