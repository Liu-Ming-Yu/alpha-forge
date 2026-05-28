"""Governance operator adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.operator.cli_inputs import load_instrument_contracts
from quant_platform.bootstrap.operator_adapters.common import latest_paper_soak
from quant_platform.bootstrap.operator_adapters.governance_commands import (
    run_dataset_quorum_request,
    run_paper_soak_request,
    run_performance_request,
    run_signal_gate_request,
    run_simulator_calibration_request,
    run_text_gate_request,
)

if TYPE_CHECKING:
    import uuid

    from quant_platform.application.governance import (
        DatasetQuorumRequest,
        PaperSoakReportRequest,
        PerformanceRequest,
        ProductionCandidateRequest,
        ReadinessRequest,
        SignalGateRequest,
        SimulatorCalibrationRequest,
        TextGateRequest,
    )
    from quant_platform.application.operator.requests import (
        PreflightRequest,
        TextEventsRequest,
    )
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.production import BrokerSmokeObservation


class GovernanceAdapters:
    """Concrete governance adapters backed by governance bootstrap helpers.

    Each method translates a typed governance DTO into explicit keyword
    arguments for a bootstrap helper.  No ``SimpleNamespace`` bridge is
    used: the typed request is the contract end to end.
    """

    def __init__(self, settings: PlatformSettings) -> None:
        self._settings = settings

    def preflight(self, request: PreflightRequest) -> tuple[dict[str, object], bool]:
        from quant_platform.bootstrap.governance.commands import preflight_payload

        return preflight_payload(
            self._settings,
            profile=request.profile,
            instrument_contracts=(
                load_instrument_contracts(request.contracts_file) if request.contracts_file else {}
            ),
        )

    async def performance(self, request: PerformanceRequest) -> UseCaseResult[dict[str, object]]:
        return await run_performance_request(self._settings, request)

    async def signal_gate(self, request: SignalGateRequest) -> UseCaseResult[dict[str, object]]:
        return await run_signal_gate_request(self._settings, request)

    async def text_gate(self, request: TextGateRequest) -> UseCaseResult[dict[str, object]]:
        return await run_text_gate_request(self._settings, request)

    async def readiness(
        self,
        request: ReadinessRequest,
    ) -> tuple[dict[str, object], bool]:
        from quant_platform.bootstrap.governance.readiness import readiness_payload_for_cli

        contracts, broker_smoke = await self._readiness_inputs(
            contracts_file=request.contracts_file,
            check_broker=request.check_broker,
        )
        return await readiness_payload_for_cli(
            self._settings,
            profile=request.profile,
            as_of=request.as_of,
            signal_name=request.signal_name,
            signal_type=request.signal_type,
            backup_manifest=request.backup_manifest,
            component=request.component,
            check_broker=request.check_broker,
            instrument_contracts=contracts,
            soak_report=request.soak_report or latest_paper_soak(self._settings),
            broker_smoke=broker_smoke,
        )

    async def production_candidate_diagnostics(
        self,
        request: ProductionCandidateRequest,
    ) -> str:
        from quant_platform.bootstrap.governance.readiness import (
            production_candidate_diagnostics_for_cli,
        )

        contracts, broker_smoke = await self._readiness_inputs(
            contracts_file=request.contracts_file,
            check_broker=request.check_broker,
        )
        diagnostics, _passed = await production_candidate_diagnostics_for_cli(
            self._settings,
            profile=request.profile,
            as_of=request.as_of,
            backup_manifest=request.backup_manifest,
            component=request.component,
            check_broker=request.check_broker,
            signal_sources=request.signal_sources,
            primary_signal_name=request.primary_signal_name,
            primary_signal_type=request.primary_signal_type,
            campaign_max_age_days=request.campaign_max_age_days,
            clean_live_days=request.clean_live_days,
            instrument_contracts=contracts,
            soak_report=request.soak_report or latest_paper_soak(self._settings),
            broker_smoke=broker_smoke,
        )
        return diagnostics

    async def production_candidate_payload(
        self,
        request: ProductionCandidateRequest,
    ) -> tuple[dict[str, object], bool]:
        from quant_platform.bootstrap.governance.readiness import (
            production_candidate_payload_for_cli,
        )

        contracts, broker_smoke = await self._readiness_inputs(
            contracts_file=request.contracts_file,
            check_broker=request.check_broker,
        )
        return await production_candidate_payload_for_cli(
            self._settings,
            command=request.command,
            profile=request.profile,
            as_of=request.as_of,
            backup_manifest=request.backup_manifest,
            component=request.component,
            check_broker=request.check_broker,
            signal_sources=request.signal_sources,
            primary_signal_name=request.primary_signal_name,
            primary_signal_type=request.primary_signal_type,
            campaign_max_age_days=request.campaign_max_age_days,
            clean_live_days=request.clean_live_days,
            instrument_contracts=contracts,
            soak_report=request.soak_report or latest_paper_soak(self._settings),
            broker_smoke=broker_smoke,
        )

    async def paper_soak(self, request: PaperSoakReportRequest) -> UseCaseResult[dict[str, object]]:
        return await run_paper_soak_request(self._settings, request)

    async def simulator_calibration(
        self, request: SimulatorCalibrationRequest
    ) -> UseCaseResult[dict[str, object]]:
        return await run_simulator_calibration_request(self._settings, request)

    async def dataset_quorum(
        self, request: DatasetQuorumRequest
    ) -> UseCaseResult[dict[str, object]]:
        return await run_dataset_quorum_request(self._settings, request)

    async def text_events(self, request: TextEventsRequest) -> UseCaseResult[dict[str, object]]:
        from quant_platform.bootstrap.text_events import text_events_command

        return await text_events_command(self._settings, request=request)

    async def _readiness_inputs(
        self,
        *,
        contracts_file: str | None,
        check_broker: bool,
    ) -> tuple[dict[uuid.UUID, dict[str, object]], BrokerSmokeObservation | None]:
        from quant_platform.bootstrap.broker import (
            broker_health,
            broker_smoke_from_report,
            ib_gateway_smoke,
        )

        contracts = load_instrument_contracts(contracts_file) if contracts_file else {}
        broker_smoke = None
        if check_broker:
            if contracts_file:
                report = await ib_gateway_smoke(self._settings, contracts)
                broker_smoke = broker_smoke_from_report(report)
            else:
                await broker_health(self._settings)
        return contracts, broker_smoke


__all__ = ["GovernanceAdapters"]
