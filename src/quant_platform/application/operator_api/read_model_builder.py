"""Builder implementation for operator-facing read models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.operator_api.read_model_lifecycle import LifecycleReadModelMixin
from quant_platform.application.operator_api.read_model_multi_engine import (
    MultiEngineReadModelMixin,
)
from quant_platform.application.operator_api.read_model_runtime import RuntimeReadModelMixin

if TYPE_CHECKING:
    from quant_platform.application.operator_api.read_model_types import (
        CashLedgerViewPort,
        ThrottleStateViewPort,
    )
    from quant_platform.core.contracts import (
        BrokerSessionGateway,
        Clock,
        EventBus,
        MultiEngineGovernanceRepository,
        OrderRepository,
        PerformanceRepository,
        PositionRepository,
        PredictionEvidenceRepository,
        SignalContributionRepository,
    )


class OperatorReadModelBuilder(
    RuntimeReadModelMixin,
    LifecycleReadModelMixin,
    MultiEngineReadModelMixin,
):
    """Build operator-facing read models from runtime state."""

    def __init__(
        self,
        clock: Clock,
        cash_ledger: CashLedgerViewPort,
        throttle: ThrottleStateViewPort,
        order_repo: OrderRepository,
        position_repo: PositionRepository,
        performance_repo: PerformanceRepository | None = None,
        event_bus: EventBus | None = None,
        account_broker: BrokerSessionGateway | None = None,
        multi_engine_repo: MultiEngineGovernanceRepository | None = None,
        signal_contribution_repo: SignalContributionRepository | None = None,
        prediction_evidence_repo: PredictionEvidenceRepository | None = None,
    ) -> None:
        self._clock = clock
        self._cash = cash_ledger
        self._throttle = throttle
        self._orders = order_repo
        self._positions = position_repo
        self._performance = performance_repo
        self._events = event_bus
        self._account_broker = account_broker
        self._multi_engine = multi_engine_repo
        self._signal_contributions = signal_contribution_repo
        self._prediction_evidence = prediction_evidence_repo
