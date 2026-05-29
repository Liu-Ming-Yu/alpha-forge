"""Tests for the within-worst-streak drawdown helper (ADR-004 Option D)."""

from __future__ import annotations

import pytest

from quant_platform.services.research_service.campaigns.evaluation.streak_containment import (
    max_drawdown_during_worst_streak,
    worst_negative_streak_window,
)


def test_worst_negative_streak_window_basic() -> None:
    assert worst_negative_streak_window([]) is None
    assert worst_negative_streak_window([0.1, 0.2]) is None
    # single longest run
    assert worst_negative_streak_window([0.1, -0.1, -0.2, 0.3]) == (1, 2)
    # ties resolve to the earliest run
    assert worst_negative_streak_window([-0.1, -0.2, 0.1, -0.3, -0.4]) == (0, 1)
    # zero is not negative
    assert worst_negative_streak_window([-0.1, 0.0, -0.1]) == (0, 0)


def test_drawdown_zero_when_no_negative_streak() -> None:
    assert max_drawdown_during_worst_streak([0.1, 0.2], [0.01, 0.02]) == 0.0


def test_drawdown_measured_over_streak_window_only() -> None:
    # Worst IC streak is folds 1-2; the big return loss in fold 3 is OUTSIDE
    # the streak window and must not count.
    fold_ics = [0.2, -0.1, -0.1, 0.3]
    fold_returns = [0.0, -0.10, -0.10, -0.50]
    dd = max_drawdown_during_worst_streak(fold_ics, fold_returns)
    # equity over folds 1-2: 0.90, then 0.81 -> drawdown -0.19
    assert dd == pytest.approx(-0.19)


def test_absorbed_episode_has_small_drawdown() -> None:
    # Arm-G-like: a strong IC inversion the construction rode out flat.
    fold_ics = [0.1, -0.25, -0.21, -0.13, -0.03, 0.02]
    fold_returns = [0.01, 0.0031, -0.0003, 0.0027, -0.0029, 0.005]
    dd = max_drawdown_during_worst_streak(fold_ics, fold_returns)
    assert dd > -0.01  # negligible drawdown despite the −0.25 IC fold
