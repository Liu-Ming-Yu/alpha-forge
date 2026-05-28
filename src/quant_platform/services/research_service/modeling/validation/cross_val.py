"""Purged k-fold cross-validation with embargo.

Implements the De Prado-style split used to fit supervised models on
financial time-series data without label leakage.  Unlike
``WalkForwardRunner`` (which is strictly forward-walking), purged
k-fold folds the whole date span into ``k`` contiguous test slices and
uses the union of the other slices as the train set; the purge gap
drops training samples whose labels could overlap the test window, and
the embargo gap drops training samples that immediately follow the test
window.

Use ``PurgedKFold`` for factor-weight fitting (Phase 2.3) where the
full history is available at calibration time and the goal is an
unbiased weight estimate.  Use ``WalkForwardRunner`` for OOS backtest
evaluation where the order of folds matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PurgedKFoldConfig:
    """Configuration for a purged k-fold cross-validation run.

    Args:
        n_splits: Number of folds (k).  Must be >= 2.
        purge_days: Calendar days dropped from train set around each test window.
        embargo_days: Calendar days dropped from train set after each test window.
        label_horizon_days: Forward-label horizon in calendar days.  When set,
            ``embargo_days`` must be >= ``label_horizon_days`` to prevent the
            post-test embargo from being shorter than the label window.
    """

    n_splits: int = 5
    purge_days: int = 0
    embargo_days: int = 0
    label_horizon_days: int = 0

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if self.label_horizon_days < 0:
            raise ValueError("label_horizon_days must be >= 0")
        if self.label_horizon_days > 0 and self.embargo_days < self.label_horizon_days:
            raise ValueError(
                f"embargo_days ({self.embargo_days}) must be >= label_horizon_days "
                f"({self.label_horizon_days}) to prevent label leakage into training"
            )


@dataclass(frozen=True)
class CVSplit:
    """One train/test split from a purged k-fold run.

    ``train_indices`` and ``test_indices`` are positions into the
    caller-supplied timestamp sequence, sorted ascending.  The purge /
    embargo gaps are baked into ``train_indices`` (dropped) rather than
    represented separately so downstream consumers cannot accidentally
    use a leaked row.
    """

    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        overlap = set(self.train_indices).intersection(self.test_indices)
        if overlap:
            raise ValueError(
                f"PurgedKFold: {len(overlap)} indices appear in both train and "
                "test (invariant violation)."
            )


class PurgedKFold:
    """k-fold splitter with purge and embargo gaps.

    Args:
        n_splits: Number of folds (``k``).  Each sample lands in exactly
            one test slice across the ``k`` splits.  Must be >= 2.
        purge_days: Calendar days dropped from the train set on either
            side of each test slice.  Labels whose forward horizon
            overlaps the test window otherwise contaminate training.
        embargo_days: Calendar days dropped from the train set after
            each test slice.  Protects against next-day leakage when
            features include lagged returns.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 0,
        embargo_days: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        if embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        self._n_splits = n_splits
        self._purge = timedelta(days=purge_days)
        self._embargo = timedelta(days=embargo_days)

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def split(self, timestamps: Sequence[datetime]) -> Iterable[CVSplit]:
        """Yield ``n_splits`` CVSplit records over ``timestamps``.

        Timestamps must be non-decreasing and timezone-aware.  The yield
        order is the natural fold order (earliest test window first),
        which makes results deterministic for reproducible calibrations.
        """
        self._validate(timestamps)

        n = len(timestamps)
        fold_size = n // self._n_splits
        remainder = n - fold_size * self._n_splits

        # Distribute the remainder across the first ``remainder`` folds,
        # exactly as ``numpy.array_split`` would, so an n=11 / k=5 split
        # yields folds of sizes (3, 2, 2, 2, 2) rather than silently
        # discarding the trailing sample.
        fold_bounds: list[tuple[int, int]] = []
        cursor = 0
        for i in range(self._n_splits):
            extra = 1 if i < remainder else 0
            start = cursor
            stop = start + fold_size + extra
            fold_bounds.append((start, stop))
            cursor = stop

        for fold_index, (test_start_idx, test_stop_idx) in enumerate(fold_bounds):
            if test_start_idx == test_stop_idx:
                continue
            test_indices = tuple(range(test_start_idx, test_stop_idx))
            test_start_ts = timestamps[test_start_idx]
            test_end_ts = timestamps[test_stop_idx - 1]

            train_indices: list[int] = []
            for idx, ts in enumerate(timestamps):
                if test_start_idx <= idx < test_stop_idx:
                    continue  # the test slice itself
                if self._is_purged(ts, test_start_ts, test_end_ts):
                    continue
                train_indices.append(idx)

            yield CVSplit(
                fold_index=fold_index,
                train_indices=tuple(train_indices),
                test_indices=test_indices,
                test_start=test_start_ts,
                test_end=test_end_ts,
            )

    def _is_purged(
        self,
        ts: datetime,
        test_start: datetime,
        test_end: datetime,
    ) -> bool:
        """Return True if ``ts`` falls within the purge/embargo gap.

        Purge is symmetric (both sides of the test window); embargo is
        applied only after the test window.
        """
        if test_start - self._purge <= ts < test_start:
            return True
        if test_end < ts <= test_end + self._purge:
            return True
        return test_end < ts <= test_end + self._embargo

    def _validate(self, timestamps: Sequence[datetime]) -> None:
        if len(timestamps) < self._n_splits:
            raise ValueError(
                f"PurgedKFold.split: {len(timestamps)} samples < n_splits={self._n_splits}."
            )
        if any(ts.tzinfo is None for ts in timestamps):
            raise ValueError("PurgedKFold.split: timestamps must be timezone-aware")
        for prev, cur in zip(timestamps, timestamps[1:], strict=False):
            if cur < prev:
                raise ValueError("PurgedKFold.split: timestamps must be non-decreasing")
