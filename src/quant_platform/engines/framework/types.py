"""Public engine runner configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

    from quant_platform.config import PlatformSettings
    from quant_platform.services.portfolio_service.portfolio_constructor import (
        LongOnlyPortfolioConstructor,
    )
    from quant_platform.services.signal_service.scoring import LinearWeightSignalModel


class RunMode(StrEnum):
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class ExecutionBackend(StrEnum):
    """Concrete execution adapter used by an engine run."""

    SIMULATED = "simulated"
    IB_PAPER = "ib-paper"


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for one strategy engine instance."""

    engine_name: str
    engine_version: str = "0.1.0"
    run_mode: RunMode = RunMode.SHADOW
    execution_backend: ExecutionBackend = ExecutionBackend.SIMULATED
    initial_cash: Decimal = Decimal("50000")
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {
            "momentum_1m": 0.20,
            "momentum_3m": 0.30,
            "momentum_12m_1m": 0.40,
            "vol_compression": 0.10,
        }
    )
    max_positions: int = 20
    rebalance_interval_seconds: float = 300.0
    instrument_contracts: dict[uuid.UUID, dict[str, object]] = field(default_factory=dict)
    plugin_name: str = ""
    feature_set_name: str = "classical"
    required_features: tuple[str, ...] = ()
    signal_model_factory: (
        Callable[
            [dict[str, float], str],
            LinearWeightSignalModel,
        ]
        | None
    ) = None
    portfolio_constructor_factory: (
        Callable[
            [PlatformSettings, int],
            LongOnlyPortfolioConstructor,
        ]
        | None
    ) = None

    @property
    def uses_order_capable_external_broker(self) -> bool:
        """True when the cycle can route orders outside the in-process simulator."""
        return self.run_mode == RunMode.LIVE or self.execution_backend == ExecutionBackend.IB_PAPER


@dataclass
class EngineRunResult:
    """Aggregated result of one or more engine cycles."""

    run_id: uuid.UUID
    engine_name: str
    run_mode: RunMode
    cycles_completed: int = 0
    total_signals: int = 0
    total_fills: int = 0
    total_submitted: int = 0
    total_rejected: int = 0
    shadow_only: bool = False
    errors: list[str] = field(default_factory=list)
