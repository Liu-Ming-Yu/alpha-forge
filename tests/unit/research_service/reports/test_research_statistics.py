from __future__ import annotations

import math

import pytest

from quant_platform.services.research_service.reports.statistics import (
    average_ranks,
    bootstrap_mean_ci,
    negative_streak,
    spearman_ic,
    winsor_impact,
)


def test_spearman_ic_uses_average_tie_ranks() -> None:
    assert average_ranks([1.0, 1.0, 3.0, 4.0]) == [0.5, 0.5, 2.0, 3.0]
    assert spearman_ic([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_ic_can_return_nan_for_report_style_invalid_inputs() -> None:
    result = spearman_ic(
        [1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0],
        invalid_value=float("nan"),
        constant_value=float("nan"),
    )
    assert math.isnan(result)


def test_bootstrap_mean_ci_is_deterministic() -> None:
    assert bootstrap_mean_ci([0.01, 0.02, 0.03], seed=7, samples=50) == pytest.approx(
        bootstrap_mean_ci([0.01, 0.02, 0.03], seed=7, samples=50)
    )


def test_negative_streak_and_winsor_impact() -> None:
    assert negative_streak([0.1, -0.1, -0.2, 0.0, -0.3]) == 2
    values = [0.0] * 20 + [100.0]
    assert winsor_impact(values) > 0.0
