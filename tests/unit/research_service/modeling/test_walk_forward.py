"""Unit tests for the walk-forward runner and its fold generator.

Focus: the leakage-safety invariants (train < purge-gap < test < embargo)
and the end-to-end stitching of per-fold ``BacktestRun`` records.  The
engine itself is mocked so we can assert on the slices it receives
without standing up a full paper session.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_platform.core.domain.research import (
    BacktestRun,
    RunStatus,
    RunType,
    StrategyRun,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardRunner,
    generate_folds,
)

_UTC = UTC


def _ts(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=_UTC)


def _strategy_run() -> StrategyRun:
    return StrategyRun(
        run_id=uuid.uuid4(),
        strategy_name="test",
        strategy_version="0.0",
        run_type=RunType.BACKTEST,
        status=RunStatus.PENDING,
        config_snapshot={},
        created_at=_ts(2024, 1, 1),
    )


class TestWalkForwardConfig:
    def test_rejects_nonpositive_windows(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardConfig(train_window_days=0, test_window_days=30, step_days=30)
        with pytest.raises(ValueError):
            WalkForwardConfig(train_window_days=30, test_window_days=0, step_days=30)

    def test_rejects_negative_purge(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardConfig(
                train_window_days=30,
                test_window_days=30,
                step_days=30,
                purge_days=-1,
            )


class TestGenerateFolds:
    def test_basic_non_overlapping(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=30,
            test_window_days=30,
            step_days=30,
        )
        folds = generate_folds(_ts(2024, 1, 1), _ts(2024, 6, 1), cfg)
        assert len(folds) >= 2
        for f in folds:
            assert f.test_start >= f.train_end
            assert f.test_end > f.test_start

    def test_test_windows_advance_by_step(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=60,
            test_window_days=30,
            step_days=30,
        )
        folds = generate_folds(_ts(2024, 1, 1), _ts(2024, 12, 1), cfg)
        for prev, cur in zip(folds, folds[1:], strict=False):
            assert cur.test_start == prev.test_start + timedelta(days=30)

    def test_purge_trims_train_tail_without_overlapping_test(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=60,
            test_window_days=30,
            step_days=30,
            purge_days=5,
        )
        folds = generate_folds(_ts(2024, 1, 1), _ts(2024, 6, 1), cfg)
        for f in folds:
            # train_end has been purged by 5 days; the 5-day purge gap
            # must sit between train_end and test_start.
            assert (f.test_start - f.train_end) == timedelta(days=5)

    def test_min_folds_enforced(self) -> None:
        cfg = WalkForwardConfig(
            train_window_days=200,
            test_window_days=200,
            step_days=30,
            min_folds=3,
        )
        with pytest.raises(ValueError, match="below min_folds"):
            generate_folds(_ts(2024, 1, 1), _ts(2024, 6, 1), cfg)


class TestWalkForwardFold:
    def test_test_before_train_rejected(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardFold(
                fold_index=0,
                train_start=_ts(2024, 2, 1),
                train_end=_ts(2024, 3, 1),
                test_start=_ts(2024, 2, 15),
                test_end=_ts(2024, 4, 1),
            )


class _StubEngine:
    """Collects ``run_with_data`` calls and returns a zero-return BacktestRun."""

    def __init__(self, per_fold_returns: list[Decimal] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._returns = per_fold_returns or []

    async def run_with_data(
        self,
        strategy_run,
        start,
        end,
        initial_capital,
        rebalance_timestamps,
        feature_series,
        price_series,
    ):
        self.calls.append(
            {
                "start": start,
                "end": end,
                "rebalances": list(rebalance_timestamps),
                "feature_ts": set(feature_series.keys()),
                "price_ts": set(price_series.keys()),
            }
        )
        total_return = (
            self._returns[len(self.calls) - 1]
            if len(self.calls) <= len(self._returns)
            else Decimal("0")
        )
        return BacktestRun(
            backtest_id=uuid.uuid4(),
            strategy_run_id=strategy_run.run_id,
            universe_snapshot_id=uuid.uuid4(),
            start_date=start,
            end_date=end,
            initial_capital=initial_capital,
            final_capital=initial_capital * (Decimal("1") + total_return),
            total_return=total_return,
            annualised_sharpe=Decimal("0"),
            max_drawdown=Decimal("0"),
            artifact_uri="",
            created_at=end,
        )


@pytest.mark.asyncio
async def test_runner_forwards_only_test_slice() -> None:
    cfg = WalkForwardConfig(
        train_window_days=30,
        test_window_days=30,
        step_days=30,
    )
    start = _ts(2024, 1, 1)
    end = _ts(2024, 6, 1)

    # rebalance every 7 days across the full window
    rebalances = []
    cursor = start
    while cursor < end:
        rebalances.append(cursor)
        cursor += timedelta(days=7)

    feature_series = {ts: {uuid.uuid4(): {"f": 1.0}} for ts in rebalances}
    price_series = {ts: {} for ts in rebalances}

    stub = _StubEngine()
    runner = WalkForwardRunner(stub)  # type: ignore[arg-type]
    result = await runner.run(
        strategy_run=_strategy_run(),
        start=start,
        end=end,
        initial_capital=Decimal("10000"),
        rebalance_timestamps=rebalances,
        feature_series=feature_series,
        price_series=price_series,
        config=cfg,
    )

    assert result.num_folds == len(stub.calls)
    for call, (fold, _run) in zip(stub.calls, result.folds, strict=False):
        for ts in call["rebalances"]:
            assert fold.test_start <= ts < fold.test_end


@pytest.mark.asyncio
async def test_combined_return_geometric_chain() -> None:
    cfg = WalkForwardConfig(
        train_window_days=30,
        test_window_days=30,
        step_days=30,
    )
    rebalances = [_ts(2024, 2, 1), _ts(2024, 3, 1), _ts(2024, 4, 1)]
    feature_series = {ts: {} for ts in rebalances}
    price_series = {ts: {} for ts in rebalances}

    stub = _StubEngine(per_fold_returns=[Decimal("0.1"), Decimal("-0.05"), Decimal("0.2")])
    runner = WalkForwardRunner(stub)  # type: ignore[arg-type]
    result = await runner.run(
        strategy_run=_strategy_run(),
        start=_ts(2024, 1, 1),
        end=_ts(2024, 6, 1),
        initial_capital=Decimal("10000"),
        rebalance_timestamps=rebalances,
        feature_series=feature_series,
        price_series=price_series,
        config=cfg,
    )

    # (1.1 * 0.95 * 1.2) - 1 = 0.254
    expected = Decimal("1.1") * Decimal("0.95") * Decimal("1.2") - Decimal("1")
    assert abs(result.combined_return - expected) < Decimal("0.0001")


@pytest.mark.asyncio
async def test_calibrator_runs_once_per_fold() -> None:
    cfg = WalkForwardConfig(
        train_window_days=30,
        test_window_days=30,
        step_days=30,
    )
    rebalances = [_ts(2024, 2, 15), _ts(2024, 3, 15), _ts(2024, 4, 15)]
    feature_series = {ts: {} for ts in rebalances}
    price_series = {ts: {} for ts in rebalances}

    calls: list[int] = []

    async def calibrator(fold):
        calls.append(fold.fold_index)
        return None

    stub = _StubEngine()
    runner = WalkForwardRunner(stub, calibrator=calibrator)  # type: ignore[arg-type]
    result = await runner.run(
        strategy_run=_strategy_run(),
        start=_ts(2024, 1, 1),
        end=_ts(2024, 6, 1),
        initial_capital=Decimal("10000"),
        rebalance_timestamps=rebalances,
        feature_series=feature_series,
        price_series=price_series,
        config=cfg,
    )

    assert calls == list(range(result.num_folds))
    # Calibrator must see exactly the evaluated folds — no trailing empty
    # fold leaks through.  The fixture's rebalance spacing generates four
    # geometric fold windows; only three contain rebalances.
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rebalance_months, expected_evaluated_folds",
    [
        ([2, 3, 4], 3),  # each rebalance in its own fold
        ([2, 3], 2),  # last fold empty
        ([2], 1),  # only the first fold has work to do
    ],
)
async def test_calibrator_skips_empty_folds(
    rebalance_months: list[int],
    expected_evaluated_folds: int,
) -> None:
    """Calibrator invocation count must match evaluated (non-empty) folds."""
    cfg = WalkForwardConfig(
        train_window_days=30,
        test_window_days=30,
        step_days=30,
    )
    rebalances = [_ts(2024, m, 15) for m in rebalance_months]
    feature_series = {ts: {} for ts in rebalances}
    price_series = {ts: {} for ts in rebalances}

    calls: list[int] = []

    async def calibrator(fold):
        calls.append(fold.fold_index)
        return None

    stub = _StubEngine()
    runner = WalkForwardRunner(stub, calibrator=calibrator)  # type: ignore[arg-type]
    result = await runner.run(
        strategy_run=_strategy_run(),
        start=_ts(2024, 1, 1),
        end=_ts(2024, 6, 1),
        initial_capital=Decimal("10000"),
        rebalance_timestamps=rebalances,
        feature_series=feature_series,
        price_series=price_series,
        config=cfg,
    )
    assert len(calls) == expected_evaluated_folds
    assert result.num_folds == expected_evaluated_folds


def test_fold_config_immutable_via_replace() -> None:
    cfg = WalkForwardConfig(
        train_window_days=30,
        test_window_days=30,
        step_days=30,
    )
    # replace works on frozen dataclasses and preserves invariants
    cfg2 = replace(cfg, step_days=15)
    assert cfg2.step_days == 15
    assert cfg.step_days == 30
