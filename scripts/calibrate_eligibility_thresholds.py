"""Held-out calibration of the portfolio_candidate streak + DD eligibility gates.

Settles ADR-004 Action Items 12 and 15: the candidate thresholds
(``max_fold_negative_ic_streak <= 4``, ``max_drawdown >= -0.10``) were chosen to
attack the v4 audit gate, never validated against held-out data. This script
asks the only question that matters for promotion: *do those thresholds hold up
out-of-sample, or were they overfit to one regime episode?*

Method
------

For each ``portfolio_candidate`` arm in a canonical latest-stack evidence
directory, split its walk-forward folds chronologically into a **calibration**
window (the earliest ``--calibration-trading-days`` of OOS) and a
**validation** window (the remainder), then recompute the two
construction-interacting gate metrics on each window:

* ``longest_negative_streak`` — exact; ``fold_negative_ic_streak`` is computed
  from per-fold ``mean_ic`` and is path-independent, so a sub-window is a clean
  hold-out.
* ``max_drawdown`` — fold-granular, chained from per-fold realized returns. This
  is a proxy for the daily-mtm drawdown the live gate uses (intra-fold dips are
  smoothed), adequate for the cross-window *comparison* this script makes. The
  live gate's DD value is unchanged.

Then sweep the ``(streak, drawdown)`` grid and report, per grid point, the set
of arms admitted on the calibration window vs the validation window. A threshold
is *stable* iff it admits the same arms on both. Tuning evidence is written as
JSON.

Scope
-----

This calibrates the **eligibility gate thresholds** (applied post-hoc to the OOS
fold series). It does **not** tune the portfolio dial's ``kill_streak`` — that is
a construction parameter whose change would require a full backtest rerun, and
is tracked separately. No rerun is needed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: ADR-004 candidate-gate defaults (the values under test). The streak grid
#: extends to 9 because the dollar-volume-scoring fix (ADR-011) raised the
#: corrected negative-IC streaks to 5–9 (the artifact had suppressed them to ~4);
#: the drawdown grid adds −0.02 to probe the v2 "contained" band.
DEFAULT_STREAK_GRID: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9)
DEFAULT_DRAWDOWN_GRID: tuple[float, ...] = (-0.02, -0.05, -0.10, -0.15)
CANDIDATE_CATEGORY = "portfolio_candidate"


@dataclass(frozen=True)
class WindowMetrics:
    """Gate-relevant metrics over a contiguous window of folds."""

    n_folds: int
    streak: int
    max_drawdown: float
    mean_ic: float
    negative_fold_fraction: float


@dataclass(frozen=True)
class ArmReport:
    """Per-arm metrics on the calibration, validation, and full windows."""

    canonical_name: str
    calibration: WindowMetrics
    validation: WindowMetrics
    full: WindowMetrics
    reported_max_drawdown_daily_mtm: float


@dataclass(frozen=True)
class GridCell:
    """Admit sets for one ``(streak, drawdown)`` threshold pair."""

    streak_threshold: int
    drawdown_threshold: float
    admit_calibration: list[str]
    admit_validation: list[str]
    admit_full: list[str]
    verdict_stable: bool


@dataclass(frozen=True)
class CalibrationReport:
    """Full tuning-evidence record."""

    calibration_trading_days: int
    calibration_cutoff: str
    streak_grid: list[int]
    drawdown_grid: list[float]
    per_arm: dict[str, ArmReport]
    grid: list[GridCell]


def longest_negative_streak(ics: Sequence[float]) -> int:
    """Longest run of consecutive strictly-negative values."""
    best = current = 0
    for value in ics:
        current = current + 1 if value < 0 else 0
        best = max(best, current)
    return best


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    """Max peak-to-trough drawdown of the equity curve built from ``returns``.

    Returns ``0.0`` for an empty sequence or a curve that only rises.
    """
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for ret in returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def window_metrics(folds: Sequence[Mapping[str, object]]) -> WindowMetrics:
    """Streak, drawdown, and IC summary over a window of folds."""
    ics = [cast("float", f["mean_ic"]) for f in folds]
    rets = [cast("float", f["total_return"]) for f in folds]
    n = len(ics)
    n_neg = sum(1 for x in ics if x < 0)
    return WindowMetrics(
        n_folds=n,
        streak=longest_negative_streak(ics),
        max_drawdown=max_drawdown_from_returns(rets),
        mean_ic=(sum(ics) / n) if n else float("nan"),
        negative_fold_fraction=(n_neg / n) if n else float("nan"),
    )


def gate_pass(streak: int, drawdown: float, streak_thresh: int, dd_thresh: float) -> bool:
    """Apply the two construction-interacting gates (streak ≤ T, DD ≥ T)."""
    return streak <= streak_thresh and drawdown >= dd_thresh


def split_folds(
    folds: Sequence[Mapping[str, object]],
    calibration_trading_days: int,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]], str]:
    """Split folds at the Nth trading day after the earliest fold's test_start.

    A fold lands in calibration iff its ``test_end`` is on or before the cutoff.
    Returns ``(calibration, validation, cutoff_iso)``.
    """
    if not folds:
        return [], [], ""
    starts = pd.to_datetime([str(f["test_start"]) for f in folds])
    cutoff = pd.bdate_range(start=starts.min(), periods=calibration_trading_days)[-1]
    calibration: list[Mapping[str, object]] = []
    validation: list[Mapping[str, object]] = []
    for fold in folds:
        if pd.to_datetime(str(fold["test_end"])) <= cutoff:
            calibration.append(fold)
        else:
            validation.append(fold)
    return calibration, validation, cutoff.isoformat()


def load_candidate_arms(evidence_dir: Path) -> dict[str, dict[str, object]]:
    """Load each portfolio_candidate arm's folds from a latest-stack run dir."""
    manifest = json.loads((evidence_dir / "run_manifest.json").read_text(encoding="utf-8"))
    arms: dict[str, dict[str, object]] = {}
    for entry in manifest["completed_arms"]:
        if entry["category"] != CANDIDATE_CATEGORY:
            continue
        evidence = json.loads((evidence_dir / entry["evidence_file"]).read_text(encoding="utf-8"))
        arms[entry["cli_alias"]] = {
            "canonical_name": entry["canonical_name"],
            "folds": evidence["folds"],
            "reported_max_drawdown": entry["max_drawdown"],
        }
    return arms


def calibrate(
    arms: dict[str, dict[str, object]],
    *,
    calibration_trading_days: int,
    streak_grid: Sequence[int],
    drawdown_grid: Sequence[float],
) -> CalibrationReport:
    """Run the hold-out split + grid sweep over the candidate arms."""
    per_arm: dict[str, ArmReport] = {}
    cutoff_iso = ""
    for alias, data in arms.items():
        folds = cast("list[Mapping[str, object]]", data["folds"])
        cal, val, cutoff_iso = split_folds(folds, calibration_trading_days)
        per_arm[alias] = ArmReport(
            canonical_name=str(data["canonical_name"]),
            calibration=window_metrics(cal),
            validation=window_metrics(val),
            full=window_metrics(folds),
            reported_max_drawdown_daily_mtm=cast("float", data["reported_max_drawdown"]),
        )

    grid: list[GridCell] = []
    for streak_t in streak_grid:
        for dd_t in drawdown_grid:
            admit_cal, admit_val, admit_full = [], [], []
            for alias, report in per_arm.items():
                if gate_pass(
                    report.calibration.streak, report.calibration.max_drawdown, streak_t, dd_t
                ):
                    admit_cal.append(alias)
                if gate_pass(
                    report.validation.streak, report.validation.max_drawdown, streak_t, dd_t
                ):
                    admit_val.append(alias)
                if gate_pass(report.full.streak, report.full.max_drawdown, streak_t, dd_t):
                    admit_full.append(alias)
            grid.append(
                GridCell(
                    streak_threshold=streak_t,
                    drawdown_threshold=dd_t,
                    admit_calibration=sorted(admit_cal),
                    admit_validation=sorted(admit_val),
                    admit_full=sorted(admit_full),
                    verdict_stable=sorted(admit_cal) == sorted(admit_val),
                )
            )

    return CalibrationReport(
        calibration_trading_days=calibration_trading_days,
        calibration_cutoff=cutoff_iso,
        streak_grid=list(streak_grid),
        drawdown_grid=list(drawdown_grid),
        per_arm=per_arm,
        grid=grid,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("data/parquet/research/backtest_latest_stack_realized_v2"),
        help="Canonical latest-stack run directory (with run_manifest.json).",
    )
    parser.add_argument(
        "--calibration-trading-days",
        type=int,
        default=252,
        help="Earliest N OOS trading days form the calibration window (ADR-004: 252).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the tuning-evidence JSON. Defaults to "
        "<evidence-dir>/threshold_calibration.json.",
    )
    return parser


def _fmt_window(m: WindowMetrics) -> str:
    return f"streak={m.streak} dd={m.max_drawdown:+.4f} n={m.n_folds} ic={m.mean_ic:+.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    arms = load_candidate_arms(args.evidence_dir)
    report = calibrate(
        arms,
        calibration_trading_days=args.calibration_trading_days,
        streak_grid=DEFAULT_STREAK_GRID,
        drawdown_grid=DEFAULT_DRAWDOWN_GRID,
    )

    print(
        f"Calibration cutoff: {report.calibration_cutoff} "
        f"(earliest {args.calibration_trading_days} OOS trading days)"
    )
    print("\nPer-arm windowed metrics (streak is exact; DD is fold-granular proxy):")
    for alias in sorted(report.per_arm):
        arm = report.per_arm[alias]
        print(f"  {alias} ({arm.canonical_name}):")
        print(f"      calibration: {_fmt_window(arm.calibration)}")
        print(f"      validation : {_fmt_window(arm.validation)}")
        print(
            f"      full       : {_fmt_window(arm.full)}  "
            f"(daily-mtm DD {arm.reported_max_drawdown_daily_mtm:+.4f})"
        )

    cal_streaks = [arm.calibration.streak for arm in report.per_arm.values()]
    print(f"\nMax streak on calibration window across candidates: {max(cal_streaks)}")

    print("\nGrid sweep (admit sets per window; * = verdict stable across split):")
    for cell in report.grid:
        flag = "*" if cell.verdict_stable else " "
        print(
            f" {flag} streak<={cell.streak_threshold} dd>={cell.drawdown_threshold:+.2f}: "
            f"cal={cell.admit_calibration} val={cell.admit_validation} full={cell.admit_full}"
        )

    out_path = args.output or (args.evidence_dir / "threshold_calibration.json")
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\nTuning evidence written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
