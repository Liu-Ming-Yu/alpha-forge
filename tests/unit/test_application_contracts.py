from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from quant_platform.application.data import (
    ComputeFeaturesRequest,
    ComputeFeaturesUseCase,
    DataHealthRequest,
    DataHealthUseCase,
    IngestRequest,
    IngestUseCase,
    IntradayDataRequest,
    IntradayDataUseCase,
    MaintainDataRequest,
    MaintainDataUseCase,
    ReprocessCorporateActionsRequest,
    ReprocessCorporateActionsUseCase,
)
from quant_platform.application.governance import (
    DatasetQuorumRequest,
    PaperSoakReportRequest,
    PerformanceRequest,
    PerformanceSnapshotRequest,
    ProductionCandidateRequest,
    ReadinessRequest,
    SignalGateRequest,
    SimulatorCalibrationRequest,
    TextGateRequest,
)
from quant_platform.application.operator import payload_coercion
from quant_platform.application.operator.requests import (
    BrokerContractsRequest,
    EventBusSweepRequest,
    ExtractTextFeaturesRequest,
    FactorsCalibrateRequest,
    IntradayCommandRequest,
    IntradayValidateRequest,
    NoInputRequest,
    PaperLifecycleRequest,
    PassiveRepriceRequest,
    PreflightRequest,
    RunCycleRequest,
    RunEngineRequest,
    RunMultiEngineRequest,
    ServeApiRequest,
    SuperviseRequest,
    TearsheetRequest,
    TextEventsRequest,
)
from quant_platform.application.operator_use_cases import (
    register_broker_use_cases,
    register_data_use_cases,
    register_engine_use_cases,
    register_governance_use_cases,
    register_infra_use_cases,
    register_research_use_cases,
    register_runtime_use_cases,
)
from quant_platform.application.research import (
    AlphaRequest,
    BacktestEvidenceAssertRequest,
    BacktestRequest,
    BoostingRequest,
    CampaignRequest,
    CampaignRunRequest,
    FeaturesBuildSamplesRequest,
    FeaturesRequest,
    FeaturesRetentionRequest,
    ModelRegistryRequest,
    WalkForwardRequest,
)
from quant_platform.application.research import evidence as research_evidence
from quant_platform.application.research.calibration_artifacts import (
    load_calibration_recommendation_bps,
)
from quant_platform.application.results import ResultPresentation, UseCaseResult
from quant_platform.application.use_cases import UseCaseRegistry


class _OperatorPorts:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def gateway_smoke(self, request: BrokerContractsRequest) -> dict[str, object]:
        self.calls.append("gateway_smoke")
        return {"passed": True, "contracts_file": request.contracts_file}

    async def paper_lifecycle(self, request: PaperLifecycleRequest) -> dict[str, object]:
        self.calls.append("paper_lifecycle")
        return {"passed": False, "instrument_id": str(request.instrument_id)}

    async def passive_reprice(self, request: PassiveRepriceRequest) -> dict[str, object]:
        self.calls.append("passive_reprice")
        return {"mode": request.mode}

    async def sweep_dead_letters(self, request: EventBusSweepRequest) -> tuple[int, int]:
        self.calls.append("sweep_dead_letters")
        return 3, 1

    async def compute_features(self, request: ComputeFeaturesRequest) -> None:
        self.calls.append("compute_features")

    async def ingest(self, request: IngestRequest) -> None:
        self.calls.append("ingest")

    async def maintain(self, request: MaintainDataRequest) -> None:
        self.calls.append("maintain")

    async def reprocess_ca(self, request: ReprocessCorporateActionsRequest) -> None:
        self.calls.append("reprocess_ca")

    async def intraday(self, request: IntradayCommandRequest) -> None:
        self.calls.append(f"intraday:{type(request).__name__}")

    async def health(self, request: DataHealthRequest | NoInputRequest) -> object:
        if isinstance(request, NoInputRequest):
            self.calls.append("runtime_health")
            return {"status": "ok"}
        self.calls.append("health")
        return {"passed": False}, False

    async def run_engine(self, request: RunEngineRequest) -> None:
        self.calls.append(f"run_engine:{request.engine_name}")

    async def run_multi_engine(self, request: RunMultiEngineRequest) -> None:
        self.calls.append(f"run_multi:{len(request.engine_names)}")

    def preflight(self, request: PreflightRequest) -> tuple[dict[str, object], bool]:
        self.calls.append(f"preflight:{request.profile}")
        return {"passed": False}, False

    async def performance(self, request: PerformanceRequest) -> None:
        self.calls.append("performance")

    async def signal_gate(self, request: SignalGateRequest) -> None:
        self.calls.append("signal_gate")

    async def text_gate(self, request: TextGateRequest) -> None:
        self.calls.append("text_gate")

    async def readiness(
        self,
        request: ReadinessRequest,
    ) -> tuple[dict[str, object], bool]:
        self.calls.append(f"readiness:{request.command}")
        return {"passed": False}, False

    async def production_candidate_diagnostics(
        self,
        request: ProductionCandidateRequest,
    ) -> str:
        self.calls.append("production_candidate_diagnostics")
        return "diagnostics"

    async def production_candidate_payload(
        self,
        request: ProductionCandidateRequest,
    ) -> tuple[dict[str, object], bool]:
        self.calls.append(f"production_candidate_payload:{request.command}")
        return {"passed": False}, False

    async def paper_soak(self, request: PaperSoakReportRequest) -> None:
        self.calls.append("paper_soak")

    async def simulator_calibration(self, request: SimulatorCalibrationRequest) -> None:
        self.calls.append("simulator_calibration")

    async def dataset_quorum(self, request: DatasetQuorumRequest) -> None:
        self.calls.append("dataset_quorum")

    async def text_events(self, request: TextEventsRequest) -> None:
        self.calls.append("text_events")

    def migrate(self, request: NoInputRequest) -> str:
        self.calls.append("migrate")
        return "head"

    def migrations_check(self, request: NoInputRequest) -> str:
        self.calls.append("migrations_check")
        return "head"

    async def verify_schema(self, request: NoInputRequest) -> None:
        self.calls.append("verify_schema")

    async def factors_calibrate(self, request: FactorsCalibrateRequest) -> None:
        self.calls.append("factors_calibrate")

    async def tearsheet(self, request: TearsheetRequest) -> None:
        self.calls.append("tearsheet")

    async def model_registry(self, request: ModelRegistryRequest) -> None:
        self.calls.append("model_registry")

    async def boosting(self, request: BoostingRequest) -> None:
        self.calls.append("boosting")

    async def alpha(self, request: AlphaRequest) -> None:
        self.calls.append("alpha")

    async def walk_forward(self, request: WalkForwardRequest) -> None:
        self.calls.append("walk_forward")

    async def features(self, request: FeaturesRequest) -> None:
        self.calls.append("features")

    async def features_retention(self, request: FeaturesRetentionRequest) -> None:
        self.calls.append("features_retention")

    async def campaign(self, request: CampaignRequest) -> None:
        self.calls.append("campaign")

    async def backtest(self, request: BacktestRequest) -> None:
        self.calls.append("backtest")

    async def run_cycle(self, request: RunCycleRequest) -> None:
        self.calls.append("run_cycle")

    async def supervise(self, request: SuperviseRequest) -> None:
        self.calls.append("supervise")

    async def serve_api(self, request: ServeApiRequest) -> UseCaseResult[str]:
        self.calls.append(f"serve_api:{request.host}:{request.port}")
        return UseCaseResult(message="serving", presentation=ResultPresentation.TEXT)

    async def smoke(self, request: NoInputRequest) -> None:
        self.calls.append("smoke")


@pytest.mark.asyncio
async def test_operator_use_case_registrations_preserve_results() -> None:
    ports = _OperatorPorts()
    registry = UseCaseRegistry()

    register_broker_use_cases(registry, ports)
    register_data_use_cases(registry, ports)
    register_engine_use_cases(registry, ports)
    register_governance_use_cases(registry, ports)
    register_infra_use_cases(registry, ports)
    register_research_use_cases(registry, ports)
    register_runtime_use_cases(registry, ports)

    today = date(2024, 1, 2)
    as_of = datetime(2024, 1, 2, tzinfo=UTC)
    instrument_id = uuid4()
    readiness_request = ReadinessRequest(
        command="assert",
        profile="paper",
        as_of=as_of,
        contracts_file=None,
        component="supervisor",
        soak_report=None,
        backup_manifest=None,
        signal_name="",
        signal_type="classical",
        check_broker=False,
    )
    production_assert = ProductionCandidateRequest(
        command="assert",
        profile="paper",
        as_of=as_of,
        contracts_file=None,
        component="supervisor",
        soak_report=None,
        backup_manifest=None,
        signal_sources=(),
        primary_signal_name="",
        primary_signal_type="",
        campaign_max_age_days=None,
        clean_live_days=0,
        check_broker=False,
    )

    results = [
        await registry.run("broker.ib_gateway_smoke", BrokerContractsRequest("contracts.json")),
        await registry.run(
            "broker.ib_paper_lifecycle",
            PaperLifecycleRequest("contracts.json", instrument_id, Decimal("1000")),
        ),
        await registry.run(
            "broker.passive_reprice_once",
            PassiveRepriceRequest("dry-run", "contracts.json", Decimal("100000")),
        ),
        await registry.run("event_bus.sweep_dead_letters", EventBusSweepRequest("orders")),
        await registry.run("data.compute_features", ComputeFeaturesRequest(Path("contracts.json"))),
        await registry.run(
            "data.ingest",
            IngestRequest(today, today, Path("contracts.json"), 86400),
        ),
        await registry.run("data.maintain", MaintainDataRequest(60.0, None, None, None)),
        await registry.run("data.reprocess_ca", ReprocessCorporateActionsRequest(instrument_id)),
        await registry.run(
            "data.intraday",
            IntradayValidateRequest(Path("bars.csv"), "polygon", "contracts.json", as_of),
        ),
        await registry.run(
            "engine.run",
            RunEngineRequest("paper", Decimal("1"), 1, None, "shadow", "paper"),
        ),
        await registry.run(
            "engine.run_multi",
            RunMultiEngineRequest("paper", ("a", "b"), None, 1, Decimal("1"), None),
        ),
        await registry.run("governance.preflight", PreflightRequest("paper", None)),
        await registry.run(
            "governance.performance",
            PerformanceSnapshotRequest(
                strategy_run_id=uuid4(),
                as_of=as_of,
                nav=Decimal("1"),
                gross_exposure=Decimal("0"),
                cash=Decimal("1"),
                source="test",
            ),
        ),
        await registry.run(
            "governance.signal_gate",
            SignalGateRequest("status", "alpha", "classical", as_of),
        ),
        await registry.run(
            "governance.text_gate",
            TextGateRequest("status", "text-alpha", as_of),
        ),
        await registry.run("governance.readiness", readiness_request),
        await registry.run(
            "governance.production_candidate",
            ProductionCandidateRequest(
                command="diagnose",
                profile=production_assert.profile,
                as_of=production_assert.as_of,
                contracts_file=production_assert.contracts_file,
                component=production_assert.component,
                soak_report=production_assert.soak_report,
                backup_manifest=production_assert.backup_manifest,
                signal_sources=production_assert.signal_sources,
                primary_signal_name=production_assert.primary_signal_name,
                primary_signal_type=production_assert.primary_signal_type,
                campaign_max_age_days=production_assert.campaign_max_age_days,
                clean_live_days=production_assert.clean_live_days,
                check_broker=production_assert.check_broker,
            ),
        ),
        await registry.run("governance.production_candidate", production_assert),
        await registry.run(
            "governance.paper_soak",
            PaperSoakReportRequest(
                profile="paper",
                strategy_run_id=uuid4(),
                as_of=as_of,
                contracts_file=None,
                signal_name="",
                signal_type="classical",
                bar_seconds=86400,
                data_health_window_days=5,
                order_latency_window_days=7,
                output=None,
            ),
        ),
        await registry.run(
            "governance.simulator_calibration",
            SimulatorCalibrationRequest(
                as_of=as_of,
                lookback_days=30,
                floor_bps=1.0,
                min_sample_count=20,
                p90_safety_margin=0.0,
                output=None,
            ),
        ),
        await registry.run(
            "governance.dataset_quorum",
            DatasetQuorumRequest("latest", "bars_eod", as_of),
        ),
        await registry.run(
            "text_events",
            ExtractTextFeaturesRequest(
                start=as_of,
                end=as_of,
                prompt_version="v1",
                document_role="all",
                source_data_manifest=None,
                artifact_root=None,
            ),
        ),
        await registry.run("infra.migrate", NoInputRequest()),
        await registry.run("infra.migrations_check", NoInputRequest()),
        await registry.run("infra.verify_schema", NoInputRequest()),
        await registry.run(
            "research.factors_calibrate",
            FactorsCalibrateRequest(Path("samples.csv"), Path("out"), 5, 0.1, 1.0),
        ),
        await registry.run("research.tearsheet", TearsheetRequest(uuid4(), Path("root"))),
        await registry.run("research.model_registry", ModelRegistryRequest(command="list")),
        await registry.run("research.boosting", BoostingRequest(command="gpu-check")),
        await registry.run("research.alpha", AlphaRequest(command="ramp", clean_live_days=1)),
        await registry.run(
            "research.walk_forward",
            WalkForwardRequest(command="run", samples=Path("s.json"), model_version="m"),
        ),
        await registry.run(
            "research.features",
            FeaturesBuildSamplesRequest(
                command="build-samples",
                contracts_file="contracts.json",
                start=as_of,
                end=as_of,
                output=Path("samples.json"),
            ),
        ),
        await registry.run(
            "research.features_retention",
            FeaturesRetentionRequest(command="retention", keep_days=30),
        ),
        await registry.run(
            "research.campaign",
            CampaignRunRequest(
                command="run",
                contracts_file="contracts.json",
                start=as_of,
                end=as_of,
                model_version="m",
            ),
        ),
        await registry.run(
            "research.backtest",
            BacktestEvidenceAssertRequest(command="assert", manifest=Path("m.json")),
        ),
        await registry.run("runtime.run_cycle", RunCycleRequest(Decimal("100"))),
        await registry.run("runtime.supervise", SuperviseRequest(Decimal("100"), 1.0)),
        await registry.run("runtime.health", NoInputRequest()),
        await registry.run("runtime.serve_api", ServeApiRequest(Decimal("100"), "127.0.0.1", 8000)),
        await registry.run("runtime.smoke", NoInputRequest()),
    ]

    assert registry.names()[0] == "broker.ib_gateway_smoke"
    assert results[1].status == "blocked"
    assert results[1].exit_code == 2
    assert results[3].message == "moved=3 dlq_stream=orders.dlq depth=1"
    assert results[15].exit_code == 2
    assert results[16].message == "diagnostics"
    assert results[37].presentation is ResultPresentation.KEY_VALUE
    assert "serve_api:127.0.0.1:8000" in ports.calls
    with pytest.raises(RuntimeError, match="unknown use case"):
        await registry.run("missing", NoInputRequest())


@pytest.mark.asyncio
async def test_domain_data_use_cases_return_use_case_results() -> None:
    calls: list[object] = []

    async def run_only(request: object) -> None:
        calls.append(request)

    async def ingest_report(request: IngestRequest) -> dict[str, object]:
        calls.append(request)
        return {"rows": 10}

    async def assertion_report(request: object) -> tuple[dict[str, object], bool]:
        calls.append(request)
        return {"passed": False}, False

    today = date(2024, 1, 2)
    instrument_id = uuid4()

    ingest = await IngestUseCase(reporter=ingest_report).run(
        IngestRequest(today, today, Path("contracts.json")),
    )
    compute = await ComputeFeaturesUseCase(runner=run_only).run(ComputeFeaturesRequest())
    maintain = await MaintainDataUseCase(runner=run_only).run(MaintainDataRequest(60.0))
    reprocess = await ReprocessCorporateActionsUseCase(runner=run_only).run(
        ReprocessCorporateActionsRequest(instrument_id),
    )
    health = await DataHealthUseCase(reporter=assertion_report).run(
        DataHealthRequest(Path("contracts.json"), today, today, 86400),
    )
    intraday = await IntradayDataUseCase(reporter=assertion_report).run(
        IntradayDataRequest("replay", {"symbol": "ABC"}),
    )

    assert ingest.payload == {"rows": 10}
    assert compute.payload == {"passed": True}
    assert maintain.status == "ok"
    assert reprocess.status == "ok"
    assert health.status == "blocked"
    assert intraday.exit_code == 2
    assert len(calls) == 6


def test_use_case_result_rejects_unknown_status() -> None:
    assert UseCaseResult(status="failed").passed is False
    with pytest.raises(ValueError, match="invalid UseCaseResult.status"):
        UseCaseResult(status="passed")


def test_payload_coercion_helpers_validate_json_shapes() -> None:
    assert payload_coercion.optional_mapping(None, name="payload") == {}
    assert payload_coercion.optional_mapping({1: "one"}, name="payload") == {"1": "one"}
    with pytest.raises(TypeError, match="payload"):
        payload_coercion.optional_mapping([], name="payload")

    assert payload_coercion.optional_sequence(None, name="items") == ()
    assert payload_coercion.optional_sequence([1, 2], name="items") == [1, 2]
    with pytest.raises(TypeError, match="items"):
        payload_coercion.optional_sequence("abc", name="items")

    assert payload_coercion.require_float("1.25", name="threshold") == 1.25
    with pytest.raises(TypeError, match="threshold"):
        payload_coercion.require_float(object(), name="threshold")


def test_calibration_artifact_reader_handles_artifact_states(tmp_path: Path) -> None:
    now = datetime(2024, 1, 5, tzinfo=UTC)
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    no_bps = tmp_path / "no_bps.json"
    no_bps.write_text('{"overall": {"recommended_bps": 0}}', encoding="utf-8")
    insufficient = tmp_path / "insufficient.json"
    insufficient.write_text(
        '{"overall": {"recommended_bps": 1}, "insufficient_data": true}',
        encoding="utf-8",
    )
    stale = tmp_path / "stale.json"
    stale.write_text(
        '{"overall": {"recommended_bps": 2}, "generated_at": "2024-01-01T00:00:00+00:00"}',
        encoding="utf-8",
    )
    fresh = tmp_path / "fresh.json"
    fresh.write_text(
        '{"overall": {"recommended_bps": 3}, "sample_count": 8, '
        '"generated_at": "2024-01-04T00:00:00"}',
        encoding="utf-8",
    )

    assert load_calibration_recommendation_bps(None)[0] is None
    assert load_calibration_recommendation_bps(missing)[0] is None
    assert load_calibration_recommendation_bps(invalid)[1]["error"].startswith("unreadable")
    assert load_calibration_recommendation_bps(no_bps)[1]["error"] == "no overall.recommended_bps"
    assert load_calibration_recommendation_bps(insufficient)[1]["error"] == "insufficient_data"
    assert (
        load_calibration_recommendation_bps(stale, max_age_days=1, as_of=now)[1]["error"] == "stale"
    )
    bps, metadata = load_calibration_recommendation_bps(
        fresh,
        max_age_days=2,
        as_of=now + timedelta(hours=1),
    )
    assert bps == 3.0
    assert metadata["sample_count"] == 8


def test_research_evidence_readers_use_canonical_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    root = tmp_path / "objects"
    manifest_root = root / "walk_forward"

    monkeypatch.setattr(
        research_evidence,
        "walk_forward_object_root",
        lambda path: Path(path) / "walk_forward",
    )
    monkeypatch.setattr(
        research_evidence,
        "list_campaign_manifests",
        lambda path, limit=20: (
            calls.append(("list", (path, limit)))
            or [{"artifact_root": str(path / "run-1"), "run_id": "run-1"}]
        ),
    )
    monkeypatch.setattr(
        research_evidence,
        "read_campaign_manifest",
        lambda path: calls.append(("read", path)) or {"run_id": path.parent.name},
    )
    monkeypatch.setattr(
        research_evidence,
        "assert_backtest_evidence",
        lambda path: calls.append(("validate", path)),
    )

    assert research_evidence.campaign_manifest_root(root) == manifest_root
    assert research_evidence.list_campaign_evidence(root, limit=5)[0]["run_id"] == "run-1"
    assert research_evidence.read_campaign_evidence(root, "missing") is None
    missing_latest = research_evidence.latest_campaign_manifest_evidence(root)
    assert missing_latest.path is None

    manifest = manifest_root / "run-2" / "campaign_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    assert research_evidence.read_campaign_evidence(root, "run-2") == {"run_id": "run-2"}

    manifest_root.mkdir(parents=True, exist_ok=True)
    latest = research_evidence.latest_campaign_manifest_evidence(root)
    assert latest.path == manifest_root / "run-1" / "campaign_manifest.json"
    assert latest.payload is not None

    research_evidence.validate_backtest_evidence_manifest("manifest.json")
    assert calls[-1] == ("validate", Path("manifest.json"))
