"""CLI entrypoint tests at the application boundary."""

from __future__ import annotations

import ast
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import quant_platform.__main__ as package_entrypoint
from quant_platform.application.data import DataHealthRequest
from quant_platform.application.governance import (
    DatasetQuorumRequest,
    PaperSoakReportRequest,
    PerformanceHeartbeatRequest,
    PerformanceReportRequest,
    PerformanceSnapshotRequest,
    ProductionCandidateRequest,
    ReadinessRequest,
    SignalGateRequest,
    SimulatorCalibrationRequest,
    TextGateRequest,
)
from quant_platform.application.operator.requests import (
    BrokerContractsRequest,
    IngestNewsTextEventsRequest,
    NoInputRequest,
    RunCycleRequest,
    RunEngineRequest,
    SuperviseRequest,
)
from quant_platform.application.results import ResultPresentation, UseCaseResult
from quant_platform.cli import app
from quant_platform.cli import context as cli_context
from quant_platform.cli.registry import BoundCommand, dispatch

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings

ROOT = Path(__file__).resolve().parents[2]


def test_package_entrypoint_exports_only_main() -> None:
    assert package_entrypoint.__all__ == ["main"]
    assert package_entrypoint.main is app.main


def test_help_does_not_instantiate_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_settings() -> PlatformSettings:
        raise AssertionError("help should exit before settings are loaded")

    monkeypatch.setattr(cli_context, "PlatformSettings", _fail_settings)

    with pytest.raises(SystemExit) as exc:
        app.main(["--help"])

    assert exc.value.code == 0


@pytest.mark.parametrize(
    ("argv", "use_case_name", "expected_request"),
    [
        (
            ["run-cycle", "--initial-cash", "12345.67"],
            "runtime.run_cycle",
            RunCycleRequest(initial_cash=Decimal("12345.67")),
        ),
        (
            ["health"],
            "runtime.health",
            NoInputRequest(),
        ),
        (
            ["ib-gateway-smoke", "--contracts-file", "contracts.json"],
            "broker.ib_gateway_smoke",
            BrokerContractsRequest(contracts_file="contracts.json"),
        ),
        (
            [
                "supervise",
                "--engine",
                "etf_macro_allocator",
                "--mode",
                "paper",
                "--execution-backend",
                "ib-paper",
                "--contracts-file",
                "contracts.json",
                "--interval",
                "60",
                "--max-cycles",
                "2",
            ],
            "runtime.supervise",
            SuperviseRequest(
                mode="paper",
                initial_cash=Decimal("50000"),
                interval_seconds=60.0,
                max_cycles=2,
                contracts_file="contracts.json",
                engine_name="etf_macro_allocator",
                execution_backend="ib-paper",
            ),
        ),
        (
            [
                "run-engine",
                "--engine",
                "etf_macro_allocator",
                "--mode",
                "paper",
                "--execution-backend",
                "ib-paper",
                "--contracts-file",
                "contracts.json",
            ],
            "engine.run",
            RunEngineRequest(
                mode="paper",
                initial_cash=Decimal("50000"),
                cycles=1,
                contracts_file="contracts.json",
                engine_name="etf_macro_allocator",
                execution_backend="ib-paper",
            ),
        ),
        (
            [
                "data-health",
                "--contracts-file",
                "contracts.json",
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-02",
            ],
            "data.health",
            DataHealthRequest(
                contracts_file=Path("contracts.json"),
                start=date(2026, 1, 1),
                end=date(2026, 1, 2),
                bar_seconds=86400,
            ),
        ),
    ],
)
def test_command_descriptors_build_typed_requests(
    argv: list[str],
    use_case_name: str,
    expected_request: object,
) -> None:
    args = app.build_parser().parse_args(argv)

    command = args._command

    assert isinstance(command, BoundCommand)
    assert command.use_case_name == use_case_name
    assert command.request_factory(args) == expected_request
    assert isinstance(command.request_factory(args), command.request_type)


@pytest.mark.parametrize(
    ("argv", "use_case_name", "expected_request"),
    [
        (
            [
                "performance",
                "snapshot",
                "--strategy-run-id",
                "00000000-0000-4000-8000-000000000001",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
                "--nav",
                "1000",
            ],
            "governance.performance",
            PerformanceSnapshotRequest(
                strategy_run_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                nav=Decimal("1000"),
                gross_exposure=Decimal("0"),
                cash=Decimal("0"),
                source="operator",
            ),
        ),
        (
            [
                "performance",
                "report",
                "--strategy-run-id",
                "00000000-0000-4000-8000-000000000001",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
            ],
            "governance.performance",
            PerformanceReportRequest(
                strategy_run_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                window=90,
            ),
        ),
        (
            [
                "performance",
                "heartbeat",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
            ],
            "governance.performance",
            PerformanceHeartbeatRequest(
                component="supervisor",
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                status="ok",
                detail="",
            ),
        ),
        (
            [
                "signal-gate",
                "record",
                "--signal-name",
                "alpha",
                "--signal-type",
                "classical",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
                "--daily-ic",
                "0.1",
            ],
            "governance.signal_gate",
            SignalGateRequest(
                command="record",
                signal_name="alpha",
                signal_type="classical",
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                daily_ic=0.1,
            ),
        ),
        (
            [
                "text-gate",
                "status",
                "--strategy-name",
                "text-alpha",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
            ],
            "governance.text_gate",
            TextGateRequest(
                command="status",
                strategy_name="text-alpha",
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            ),
        ),
        (
            [
                "readiness",
                "assert",
                "--profile",
                "paper",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
            ],
            "governance.readiness",
            ReadinessRequest(
                command="assert",
                profile="paper",
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                contracts_file=None,
                component="supervisor",
                soak_report=None,
                backup_manifest=None,
                signal_name="",
                signal_type="classical",
                check_broker=False,
            ),
        ),
        (
            [
                "production-candidate",
                "diagnose",
                "--profile",
                "paper",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
                "--signal-source",
                "alpha",
            ],
            "governance.production_candidate",
            ProductionCandidateRequest(
                command="diagnose",
                profile="paper",
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                contracts_file=None,
                component="supervisor",
                soak_report=None,
                backup_manifest=None,
                signal_sources=("alpha",),
                primary_signal_name="",
                primary_signal_type="",
                campaign_max_age_days=None,
                clean_live_days=0,
                check_broker=False,
            ),
        ),
        (
            [
                "paper-soak",
                "report",
                "--strategy-run-id",
                "00000000-0000-4000-8000-000000000001",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
            ],
            "governance.paper_soak",
            PaperSoakReportRequest(
                profile="paper",
                strategy_run_id=uuid.UUID("00000000-0000-4000-8000-000000000001"),
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                contracts_file=None,
                signal_name="",
                signal_type="classical",
                bar_seconds=86400,
                data_health_window_days=5,
                order_latency_window_days=7,
                output=None,
            ),
        ),
        (
            [
                "dataset-quorum",
                "latest",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
            ],
            "governance.dataset_quorum",
            DatasetQuorumRequest(
                command="latest",
                dataset_kind="bars_eod",
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
            ),
        ),
        (
            [
                "simulator-calibration",
                "report",
                "--as-of",
                "2026-01-01T00:00:00+00:00",
            ],
            "governance.simulator_calibration",
            SimulatorCalibrationRequest(
                as_of=datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                lookback_days=30,
                floor_bps=1.0,
                min_sample_count=20,
                p90_safety_margin=0.0,
                output=None,
            ),
        ),
    ],
)
def test_governance_command_descriptors_build_explicit_requests(
    argv: list[str],
    use_case_name: str,
    expected_request: object,
) -> None:
    args = app.build_parser().parse_args(argv)

    command = args._command

    assert isinstance(command, BoundCommand)
    assert command.use_case_name == use_case_name
    assert command.request_factory(args) == expected_request
    assert isinstance(command.request_factory(args), command.request_type)


def test_dispatch_runs_use_case_and_renders_result(capsys: pytest.CaptureFixture[str]) -> None:
    args = app.build_parser().parse_args(["health"])
    calls: list[tuple[str, object]] = []

    class _FakeContext:
        async def run(self, use_case_name: str, request: object) -> UseCaseResult[dict[str, str]]:
            calls.append((use_case_name, request))
            return UseCaseResult(
                payload={"status": "ok"},
                presentation=ResultPresentation.KEY_VALUE,
            )

    assert dispatch(args, _FakeContext()) == 0

    assert calls == [("runtime.health", NoInputRequest())]
    assert "status: ok" in capsys.readouterr().out


def test_text_events_ingest_news_builds_tws_request() -> None:
    args = app.build_parser().parse_args(
        [
            "text-events",
            "ingest-news",
            "--vendor",
            "tws",
            "--contracts-file",
            "contracts.json",
            "--start",
            "2026-04-29T00:00:00+00:00",
            "--end",
            "2026-04-30T00:00:00+00:00",
            "--provider-codes",
            "BRFG+DJNL",
            "--total-results-per-symbol",
            "25",
            "--headline-only",
        ]
    )

    command = args._command

    assert isinstance(command, BoundCommand)
    assert command.use_case_name == "text_events"
    assert command.request_factory(args) == IngestNewsTextEventsRequest(
        vendor="tws",
        contracts_file="contracts.json",
        start=datetime.fromisoformat("2026-04-29T00:00:00+00:00"),
        end=datetime.fromisoformat("2026-04-30T00:00:00+00:00"),
        provider_codes=("BRFG+DJNL",),
        total_results_per_symbol=25,
        include_article_text=False,
        artifact_root=None,
    )


def test_cli_source_does_not_import_retired_operations() -> None:
    offenders: list[str] = []
    for path in (ROOT / "src" / "quant_platform" / "cli").rglob("*.py"):
        if path.name == "operations.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "quant_platform.cli.operations":
                        offenders.append(path.as_posix())
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module == "quant_platform.cli"
                and any(alias.name == "operations" for alias in node.names)
            ):
                offenders.append(path.as_posix())

    assert offenders == []
