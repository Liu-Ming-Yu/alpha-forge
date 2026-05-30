"""Unit tests for ``scripts/calibrate_eligibility_thresholds.py``.

Exercises the importable helpers (streak, drawdown, fold split, grid sweep)
with synthetic fold series rather than the file-reading ``main()`` flow.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "calibrate_eligibility_thresholds.py"
)
_spec = importlib.util.spec_from_file_location("calibrate_eligibility_thresholds", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
calib = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_eligibility_thresholds"] = calib
_spec.loader.exec_module(calib)


def test_longest_negative_streak() -> None:
    assert calib.longest_negative_streak([]) == 0
    assert calib.longest_negative_streak([0.1, 0.2]) == 0
    assert calib.longest_negative_streak([-0.1, -0.2, 0.1, -0.3]) == 2
    assert calib.longest_negative_streak([0.1, -0.1, -0.1, -0.1, 0.2]) == 3
    # zero is not negative, so it breaks the run
    assert calib.longest_negative_streak([-0.1, 0.0, -0.1]) == 1


def test_max_drawdown_from_returns() -> None:
    assert calib.max_drawdown_from_returns([]) == 0.0
    assert calib.max_drawdown_from_returns([0.1, 0.1]) == 0.0  # only rises
    # +10% then -50% -> trough at 0.55 from peak 1.1 -> -0.5 drawdown
    dd = calib.max_drawdown_from_returns([0.1, -0.5])
    assert dd == pytest.approx(-0.5)


def test_gate_pass_boundaries() -> None:
    # streak at the bound passes; one over fails
    assert calib.gate_pass(4, -0.04, 4, -0.10) is True
    assert calib.gate_pass(5, -0.04, 4, -0.10) is False
    # DD at the bound passes; worse fails
    assert calib.gate_pass(4, -0.10, 4, -0.10) is True
    assert calib.gate_pass(4, -0.11, 4, -0.10) is False


def _fold(start: str, end: str, ic: float, ret: float) -> dict[str, object]:
    return {"test_start": start, "test_end": end, "mean_ic": ic, "total_return": ret}


def test_split_folds_partitions_by_cutoff() -> None:
    folds = [
        _fold("2022-01-03", "2022-02-01", 0.05, 0.01),
        _fold("2022-02-01", "2022-03-01", -0.02, 0.00),
        _fold("2024-06-17", "2024-07-15", -0.25, 0.00),
    ]
    cal, val, cutoff = calib.split_folds(folds, calibration_trading_days=60)
    # 60 business days from 2022-01-03 lands in early 2022 -> first two folds calibration
    assert len(cal) == 2
    assert len(val) == 1
    assert val[0]["mean_ic"] == -0.25
    assert cutoff.startswith("2022-03")


def test_window_metrics() -> None:
    folds = [
        _fold("2022-01-03", "2022-02-01", 0.05, 0.02),
        _fold("2022-02-01", "2022-03-01", -0.02, -0.01),
        _fold("2022-03-01", "2022-04-01", -0.03, -0.02),
    ]
    m = calib.window_metrics(folds)
    assert m.n_folds == 3
    assert m.streak == 2
    assert m.negative_fold_fraction == pytest.approx(2 / 3)
    assert m.max_drawdown < 0.0


def test_calibrate_flags_unstable_verdict() -> None:
    # Arm whose streak is fine in calibration but blows out in validation:
    # the verdict cannot be stable across the split.
    arms = {
        "G": {
            "canonical_name": "g",
            "reported_max_drawdown": -0.04,
            "reported_streak": 4.0,
            "folds": [
                _fold("2022-01-03", "2022-02-01", 0.05, 0.01),
                _fold("2022-02-01", "2022-03-01", 0.04, 0.01),
                _fold("2024-06-17", "2024-07-15", -0.2, -0.01),
                _fold("2024-07-15", "2024-08-15", -0.2, -0.01),
                _fold("2024-08-15", "2024-09-15", -0.2, -0.01),
                _fold("2024-09-15", "2024-10-15", -0.2, -0.01),
            ],
        }
    }
    report = calib.calibrate(
        arms,
        calibration_trading_days=60,
        streak_grid=(3, 4),
        drawdown_grid=(-0.10,),
    )
    g = report.per_arm["G"]
    assert g.calibration.streak == 0  # only positive folds in calibration
    assert g.validation.streak == 4  # the four-fold inversion
    # No grid cell can be stable: calibration admits G everywhere, validation
    # rejects it at streak<=3.
    cell_streak3 = next(c for c in report.grid if c.streak_threshold == 3)
    assert "G" in cell_streak3.admit_calibration
    assert "G" not in cell_streak3.admit_validation
    assert cell_streak3.verdict_stable is False
