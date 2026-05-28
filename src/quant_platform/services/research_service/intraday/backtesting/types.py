"""Typed DTOs for intraday backtest evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime
    from decimal import Decimal
    from pathlib import Path

    from quant_platform.core.domain.orders import ExecutionTactic, OrderIntent


@dataclass(frozen=True)
class IntradayFillArtifact:
    """Fill-level TCA row produced by intraday tactic replay."""

    order_id: uuid.UUID
    instrument_id: uuid.UUID
    side: str
    tactic: str
    executed_at: datetime
    quantity: int
    requested_quantity: int
    residual_quantity: int
    arrival_price: Decimal
    decision_price: Decimal
    minute_vwap: Decimal
    fill_price: Decimal
    spread_bps: Decimal
    participation_rate: Decimal
    slippage_bps: Decimal
    implementation_shortfall_bps: Decimal
    commission: Decimal
    is_complete: bool


@dataclass(frozen=True)
class IntradayReplayOrderResult:
    """Replay result for one approved order intent."""

    intent: OrderIntent
    tactic: ExecutionTactic
    fills: tuple[IntradayFillArtifact, ...]
    residual_quantity: int
    comparable: bool


@dataclass(frozen=True)
class IntradayBacktestResult:
    """Canonical output of an intraday backtest engine."""

    strategy_run_id: uuid.UUID
    final_capital: Decimal
    total_return: Decimal
    max_drawdown: Decimal
    nav_curve: tuple[tuple[datetime, Decimal], ...]
    target_weights: Mapping[datetime, Mapping[uuid.UUID, Decimal]]
    eligible_universe: Mapping[datetime, tuple[uuid.UUID, ...]]
    fills: tuple[IntradayFillArtifact, ...]
    residual_order_count: int
    artifact_root: Path
    run_summary_uri: str
    execution_quality_uri: str
    fills_uri: str
    target_weights_uri: str
