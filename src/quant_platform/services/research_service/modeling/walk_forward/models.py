"""Typed DTOs for walk-forward research evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from decimal import Decimal

    from quant_platform.core.domain.research import BacktestRun


@dataclass(frozen=True)
class WalkForwardConfig:
    """Parameters for rolling-origin train/test splits.

    **Unit gotcha.** ``train_window_days``, ``test_window_days``,
    ``step_days``, ``purge_days``, and ``embargo_days`` all count
    *calendar* days because the fold generator slices on a calendar
    timeline. ``label_horizon_days`` counts *trading* days because the
    forward-return labels live on a trading-day calendar
    (``close[t]`` → ``close[t+H]`` where ``t`` indexes trading days, not
    calendar days). The conservative purge-vs-horizon check uses the
    larger of the two units — ``purge_days >= label_horizon_days`` —
    which over-protects in calendar-day terms (21 trading days ≈ 31
    calendar days) but is the right cut on the trading-day axis once a
    sample-level purge (``run_sample_walk_forward``) tightens it.

    ``label_horizon_days`` is the forward-return horizon used to build
    the sample labels (e.g. ``21`` for a 21-trading-day forward return).
    When set, the config validates that ``purge_days >= label_horizon_days``
    so a training label cannot leak past the purge gap into the test
    window.

    The field is optional to preserve backward compatibility: existing
    callers that don't declare the horizon get the old validation
    (``purge_days >= 0`` only). Callers that do declare it opt into the
    stricter check.

    See ``docs/architecture/adr-003-return-accounting-separation.md``
    for the rationale.
    """

    train_window_days: int
    test_window_days: int
    step_days: int
    purge_days: int = 0
    embargo_days: int = 0
    min_folds: int = 1
    label_horizon_days: int | None = None

    def __post_init__(self) -> None:
        if self.train_window_days <= 0:
            raise ValueError("train_window_days must be > 0")
        if self.test_window_days <= 0:
            raise ValueError("test_window_days must be > 0")
        if self.step_days <= 0:
            raise ValueError("step_days must be > 0")
        if self.purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if self.min_folds < 1:
            raise ValueError("min_folds must be >= 1")
        if self.label_horizon_days is not None:
            if self.label_horizon_days <= 0:
                raise ValueError("label_horizon_days must be > 0 when set")
            if self.purge_days < self.label_horizon_days:
                raise ValueError(
                    "purge_days must be >= label_horizon_days; got "
                    f"purge_days={self.purge_days}, "
                    f"label_horizon_days={self.label_horizon_days}",
                )


@dataclass(frozen=True)
class WalkForwardFold:
    """Inclusive/exclusive boundaries for one fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        if self.train_end <= self.train_start:
            raise ValueError("train_end must be after train_start")
        if self.test_end <= self.test_start:
            raise ValueError("test_end must be after test_start")
        if self.test_start < self.train_end:
            raise ValueError("test_start must be >= train_end (purge must be non-negative)")


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate output of a walk-forward evaluation."""

    config: WalkForwardConfig
    folds: tuple[tuple[WalkForwardFold, BacktestRun], ...]
    combined_return: Decimal
    started_at: datetime
    finished_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def num_folds(self) -> int:
        return len(self.folds)


__all__ = ["WalkForwardConfig", "WalkForwardFold", "WalkForwardResult"]
