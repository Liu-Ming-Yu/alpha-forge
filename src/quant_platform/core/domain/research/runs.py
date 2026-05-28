"""Research run lifecycle domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping
    from datetime import datetime


class RunType(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class StrategyRun:
    """A single execution record for a strategy, applicable to both live and backtest runs.

    Sharing this type ensures every order/fill can be attributed to a run and
    that the audit trail is uniform across research and production.

    Args:
        run_id: Stable system UUID.
        strategy_name: Human-readable strategy identifier (e.g. "cross_sectional_equity_v1").
        strategy_version: Semantic version of the strategy code.
        run_type: Whether this is a backtest, paper, or live run.
        status: Lifecycle status of the run.
        config_snapshot: JSON-serialisable dict of strategy parameters at run start.
            Must be captured at run creation and never mutated.
        created_at: UTC timestamp when the run was created.
        started_at: UTC timestamp when execution actually began.
        finished_at: UTC timestamp when the run completed or failed.
    """

    run_id: uuid.UUID
    strategy_name: str
    strategy_version: str
    run_type: RunType
    status: RunStatus
    config_snapshot: Mapping[str, object]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
