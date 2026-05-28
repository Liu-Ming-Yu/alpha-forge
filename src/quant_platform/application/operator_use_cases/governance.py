"""Governance operator use cases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quant_platform.application.results import (
    ResultPresentation,
    UseCaseResult,
    UseCaseStatus,
)
from quant_platform.application.use_cases import CallableUseCase, UseCaseRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

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


class GovernanceUseCasePorts(Protocol):
    """Governance adapters required by operator use cases."""

    def preflight(self, request: PreflightRequest) -> tuple[dict[str, object], bool]: ...

    def performance(
        self, request: PerformanceRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def signal_gate(
        self, request: SignalGateRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def text_gate(
        self, request: TextGateRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def readiness(
        self,
        request: ReadinessRequest,
    ) -> Awaitable[tuple[dict[str, object], bool]]: ...

    def production_candidate_diagnostics(
        self,
        request: ProductionCandidateRequest,
    ) -> Awaitable[str]: ...

    def production_candidate_payload(
        self,
        request: ProductionCandidateRequest,
    ) -> Awaitable[tuple[dict[str, object], bool]]: ...

    def paper_soak(
        self, request: PaperSoakReportRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def simulator_calibration(
        self, request: SimulatorCalibrationRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def dataset_quorum(
        self, request: DatasetQuorumRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...

    def text_events(
        self, request: TextEventsRequest
    ) -> Awaitable[UseCaseResult[dict[str, object]]]: ...


def register_governance_use_cases(
    registry: UseCaseRegistry,
    ports: GovernanceUseCasePorts,
) -> None:
    """Register governance and text-event use cases."""

    def preflight(request: PreflightRequest) -> UseCaseResult[dict[str, object]]:
        payload, passed = ports.preflight(request)
        return _json_assertion(payload, passed)

    async def performance(request: PerformanceRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.performance(request)

    async def signal_gate(request: SignalGateRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.signal_gate(request)

    async def text_gate(request: TextGateRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.text_gate(request)

    async def readiness(request: ReadinessRequest) -> UseCaseResult[dict[str, object]]:
        payload, passed = await ports.readiness(request)
        return _json_assertion(payload, passed, assert_mode=request.command == "assert")

    async def production_candidate(
        request: ProductionCandidateRequest,
    ) -> UseCaseResult[dict[str, object] | str]:
        if request.command == "diagnose":
            diagnostics = await ports.production_candidate_diagnostics(request)
            return UseCaseResult(message=diagnostics, presentation=ResultPresentation.TEXT)
        payload, passed = await ports.production_candidate_payload(request)
        return _json_assertion(payload, passed, assert_mode=request.command == "assert")

    async def paper_soak(request: PaperSoakReportRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.paper_soak(request)

    async def simulator_calibration(
        request: SimulatorCalibrationRequest,
    ) -> UseCaseResult[dict[str, object]]:
        return await ports.simulator_calibration(request)

    async def dataset_quorum(request: DatasetQuorumRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.dataset_quorum(request)

    async def text_events(request: TextEventsRequest) -> UseCaseResult[dict[str, object]]:
        return await ports.text_events(request)

    registry.register("governance.preflight", CallableUseCase(preflight))
    registry.register("governance.performance", CallableUseCase(performance))
    registry.register("governance.signal_gate", CallableUseCase(signal_gate))
    registry.register("governance.text_gate", CallableUseCase(text_gate))
    registry.register("governance.readiness", CallableUseCase(readiness))
    registry.register("governance.production_candidate", CallableUseCase(production_candidate))
    registry.register("governance.paper_soak", CallableUseCase(paper_soak))
    registry.register("governance.simulator_calibration", CallableUseCase(simulator_calibration))
    registry.register("governance.dataset_quorum", CallableUseCase(dataset_quorum))
    registry.register("text_events", CallableUseCase(text_events))


def _json_assertion(
    payload: dict[str, object],
    passed: bool,
    *,
    assert_mode: bool = True,
) -> UseCaseResult[dict[str, object]]:
    return UseCaseResult(
        status=UseCaseStatus.OK if passed else UseCaseStatus.BLOCKED,
        payload=payload,
        exit_code=2 if assert_mode and not passed else 0,
        presentation=ResultPresentation.JSON,
    )


__all__ = ["GovernanceUseCasePorts", "register_governance_use_cases"]
