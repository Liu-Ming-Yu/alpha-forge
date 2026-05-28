"""Registration for the main research-campaign run command."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.application.research import CampaignRunRequest
from quant_platform.cli.commands.research.request_factories import campaign_run_request
from quant_platform.cli.registry import bind_command


def register_campaign_run(rc_sub: Any) -> None:
    run_p = rc_sub.add_parser(
        "run",
        help="Build samples, run walk-forward, optionally train XGBoost, and write a manifest.",
    )
    run_p.add_argument("--contracts-file", required=True)
    run_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    run_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    run_p.add_argument("--feature-set-version", default="1.0.0")
    run_p.add_argument(
        "--date-policy",
        choices=["nyse-sessions", "calendar-days"],
        default="nyse-sessions",
        help="Research sample calendar policy; governed paper campaigns use NYSE sessions.",
    )
    run_p.add_argument("--horizon-days", type=int, default=21)
    run_p.add_argument("--model-version", required=True)
    run_p.add_argument(
        "--signal-type",
        choices=["auto", "classical", "text", "event", "intraday", "xgboost"],
        default="auto",
        help="Source signal gate to record for this campaign; auto derives the gate.",
    )
    run_p.add_argument("--output-root", type=Path, default=None)
    run_p.add_argument("--train-xgboost", action="store_true")
    run_p.add_argument("--xgboost-search", choices=["off", "conservative"], default="off")
    run_p.add_argument("--xgboost-device", choices=["auto", "cpu", "cuda"], default="auto")
    run_p.add_argument("--xgboost-require-gpu", action="store_true")
    run_p.add_argument(
        "--paper-source-weights-json",
        default=(
            '{"classical": 0.70, "xgboost": 0.15, "text": 0.05, "event": 0.05, "intraday": 0.05}'
        ),
        help="JSON object of paper-only ensemble source weights.",
    )
    run_p.add_argument("--bar-seconds", type=int, default=86400)
    run_p.add_argument("--max-feature-age-days", type=int, default=3)
    run_p.add_argument("--train-window-days", type=int, default=252)
    run_p.add_argument("--test-window-days", type=int, default=21)
    run_p.add_argument("--step-days", type=int, default=21)
    run_p.add_argument("--purge-days", type=int, default=21)
    run_p.add_argument("--embargo-days", type=int, default=0)
    run_p.add_argument("--min-folds", type=int, default=3)
    run_p.add_argument("--min-oos-rolling-ic", type=float, default=0.05)
    run_p.add_argument("--min-ic-60d", type=float, default=0.03)
    # Fold-level streak gate. Legacy ``--max-negative-ic-streak`` accepted as
    # an alias so existing scripts keep working, but the unit is fold-IC, not
    # daily-IC — see AlphaEligibilityThresholds.
    run_p.add_argument(
        "--max-fold-negative-ic-streak",
        "--max-negative-ic-streak",
        dest="max_fold_negative_ic_streak",
        type=int,
        default=2,
    )
    run_p.add_argument("--max-drawdown", type=float, default=-0.20)
    run_p.add_argument("--min-slippage-adjusted-sharpe", type=float, default=1.0)
    run_p.add_argument("--slippage-bps-per-turnover", type=float, default=10.0)
    run_p.add_argument(
        "--return-scale",
        type=float,
        default=1.0,
        help="Governed campaigns require 1.0; any other value fails closed.",
    )
    run_p.add_argument(
        "--campaign-portfolio-mode",
        choices=["runtime-long-only"],
        default="runtime-long-only",
    )
    run_p.add_argument("--campaign-top-n", type=int, default=10)
    run_p.add_argument("--campaign-vol-target", type=float, default=0.15)
    run_p.add_argument("--campaign-vol-floor", type=float, default=0.05)
    run_p.add_argument("--campaign-vol-lookback-days", type=int, default=63)
    run_p.add_argument("--campaign-max-gross-exposure", type=float, default=0.60)
    run_p.add_argument("--campaign-min-cash-buffer", type=float, default=0.05)
    run_p.add_argument("--campaign-max-single-name-weight", type=float, default=0.05)
    run_p.add_argument("--campaign-max-daily-turnover", type=float, default=0.20)
    run_p.add_argument("--campaign-max-position-change", type=float, default=0.05)
    run_p.add_argument(
        "--campaign-no-trade-band",
        type=float,
        default=0.0,
        help=(
            "Cost-aware hysteresis: hold the current weight when the desired "
            "change is below this band. 0.0 disables it. Sweep [0.002, 0.01] "
            "to cut turnover and lift slippage-adjusted Sharpe."
        ),
    )
    run_p.add_argument(
        "--campaign-rebalance-interval-days",
        type=int,
        default=1,
        help=(
            "Rebalance only every Nth observation day; carry weights forward "
            "between. 1 rebalances daily. Sweep {1,5,10,21} to cut turnover."
        ),
    )
    run_p.add_argument(
        "--max-calibration-age-days",
        type=float,
        default=14.0,
        help="Reject stale simulator calibration artifacts when deriving calibrated slippage.",
    )
    run_p.add_argument(
        "--require-calibration",
        action="store_true",
        help="Fail the campaign when a fresh simulator calibration artifact is unavailable.",
    )
    run_p.add_argument(
        "--feature-audit-mode",
        choices=["off", "shadow", "paper"],
        default="shadow",
        help="off skips feature audits; shadow records blockers; paper fails on audit gaps.",
    )
    run_p.add_argument("--feature-card-dir", type=Path, default=None)
    run_p.add_argument(
        "--feature-diagnostics",
        type=Path,
        default=None,
        help="Optional feature_direction_diagnostics.json path to record in manifests.",
    )
    run_p.add_argument(
        "--feature-family-file",
        type=Path,
        default=None,
        help="Optional feature family metadata for governed attribution preflight.",
    )
    run_p.add_argument(
        "--source-data-manifest",
        type=Path,
        default=None,
        help="Optional source-data manifest to link from text model manifests.",
    )
    run_p.add_argument(
        "--text-prompt-version",
        default="",
        help="Override QP__LLM__TEXT_PROMPT_VERSION in text model manifests.",
    )
    run_p.add_argument("--attribution-horizons", nargs="+", type=int, default=[5, 10, 21])
    run_p.add_argument("--attribution-permutation-seed", type=int, default=17)
    run_p.add_argument("--attribution-permutation-count", type=int, default=200)
    run_p.add_argument("--attribution-correlation-threshold", type=float, default=0.70)
    run_p.add_argument("--min-null-qualified-features", type=int, default=3)
    run_p.add_argument(
        "--feature-admission",
        choices=["passing", "all"],
        default="passing",
        help="Use only passing audited features, or require every audited feature to pass.",
    )
    run_p.add_argument("--min-admitted-features", type=int, default=3)
    run_p.add_argument(
        "--model-feature",
        action="append",
        default=None,
        help=(
            "Repeatable audited feature name to use as the campaign model input. "
            "When omitted, all admitted features are used."
        ),
    )
    run_p.add_argument("--fail-on-ineligible", action="store_true")
    bind_command(
        run_p,
        use_case_name="research.campaign",
        request_factory=campaign_run_request,
        request_type=CampaignRunRequest,
    )


__all__ = ["register_campaign_run"]
