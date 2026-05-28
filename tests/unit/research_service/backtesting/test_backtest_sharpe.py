"""Unit tests for the annualised Sharpe helper.

Covers commit 8 / R-GOV-03: ``SimpleBacktestEngine.run_with_data`` used
to hardcode ``annualised_sharpe=Decimal("0")``, silently understating
both broken strategies (should be < 0) and legitimately strong ones
(should be > 0).  The helper extracted in commit 8 performs the
standard mean/std annualisation with a 252 trading-day factor and
returns ``None`` when the sample size is below 20.
"""

from __future__ import annotations

from decimal import Decimal

from quant_platform.services.research_service.backtesting.simple.backtest_engine import (
    _compute_annualised_sharpe,
)


def test_sharpe_none_below_twenty_samples() -> None:
    """Fewer than 20 returns → None (sample too small to annualise)."""
    curve = [Decimal("100") + Decimal(str(i)) for i in range(10)]
    assert _compute_annualised_sharpe(curve) is None


def test_sharpe_none_on_flat_curve() -> None:
    """Zero variance → None (division by zero guard)."""
    curve = [Decimal("100")] * 30
    assert _compute_annualised_sharpe(curve) is None


def test_sharpe_positive_for_monotonic_up_curve() -> None:
    """A smoothly-climbing NAV should report a large positive Sharpe."""
    curve = [Decimal("100") * (Decimal("1.001") ** i) for i in range(30)]
    # Add a little noise so variance isn't zero.
    curve[10] = curve[10] * Decimal("1.0005")
    sharpe = _compute_annualised_sharpe(curve)
    assert sharpe is not None
    assert sharpe > 0
    # Sanity: annualisation factor is 252.  A ~10 bps/day return with
    # tiny variance should blow the Sharpe comfortably above 1.
    assert sharpe > Decimal("1.0")


def test_sharpe_negative_for_losing_strategy() -> None:
    """A drifting-down NAV produces a negative Sharpe."""
    curve = [Decimal("100") * (Decimal("0.999") ** i) for i in range(30)]
    curve[5] = curve[5] * Decimal("1.0005")
    sharpe = _compute_annualised_sharpe(curve)
    assert sharpe is not None
    assert sharpe < 0


def test_sharpe_deterministic_given_fixed_returns() -> None:
    """The helper is deterministic for a fixed input curve."""
    curve = [Decimal(str(100 + i * 0.1)) for i in range(50)]
    a = _compute_annualised_sharpe(curve)
    b = _compute_annualised_sharpe(list(curve))
    assert a is not None
    assert a == b
