"""Research command registrations."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.application.operator.requests import (
    FactorsCalibrateRequest,
    TearsheetRequest,
)
from quant_platform.application.research import (
    AlphaRequest,
    BoostingRequest,
    ModelRegistryRequest,
    WalkForwardRequest,
)
from quant_platform.cli.commands.research.backtest import register_backtest
from quant_platform.cli.commands.research.campaign import register_research_campaign
from quant_platform.cli.commands.research.features import register_features
from quant_platform.cli.commands.research.request_factories import (
    alpha_request,
    boosting_request,
    model_registry_request,
    walk_forward_request,
)
from quant_platform.cli.registry import bind_command
from quant_platform.config_signal_models.alpha import ALPHA_SOURCE_TYPES


def register(sub: Any) -> None:
    _register_factors(sub)
    _register_tearsheet(sub)
    _register_model_registry(sub)
    _register_boosting(sub)
    _register_alpha(sub)
    _register_walk_forward(sub)
    register_features(sub)
    register_research_campaign(sub)
    register_backtest(sub)


def _register_factors(sub: Any) -> None:
    fac_p = sub.add_parser("factors", help="Factor calibration utilities.")
    fac_sub = fac_p.add_subparsers(dest="factors_command", required=True)
    cal_p = fac_sub.add_parser(
        "calibrate",
        help="Fit NNLS+L2 factor weights from a samples JSON file.",
    )
    cal_p.add_argument("--samples", required=True, type=Path)
    cal_p.add_argument("--output-dir", type=Path, default=Path("data/calibration"))
    cal_p.add_argument("--horizon-days", type=int, default=21)
    cal_p.add_argument("--l2-lambda", type=float, default=1e-3)
    cal_p.add_argument("--momentum-scale", type=float, default=0.90)
    bind_command(
        cal_p,
        use_case_name="research.factors_calibrate",
        request_factory=lambda args: FactorsCalibrateRequest(
            samples_path=args.samples,
            output_dir=args.output_dir,
            horizon_days=args.horizon_days,
            l2_lambda=args.l2_lambda,
            momentum_scale=args.momentum_scale,
        ),
        request_type=FactorsCalibrateRequest,
    )


def _register_tearsheet(sub: Any) -> None:
    tear_p = sub.add_parser(
        "tearsheet",
        help="Render a Markdown tearsheet for a completed backtest run.",
    )
    tear_p.add_argument("--run-id", required=True, type=uuid.UUID)
    tear_p.add_argument("--root", type=Path, default=Path("data/backtest"))
    bind_command(
        tear_p,
        use_case_name="research.tearsheet",
        request_factory=lambda args: TearsheetRequest(run_id=args.run_id, root=args.root),
        request_type=TearsheetRequest,
    )


def _register_model_registry(sub: Any) -> None:
    mr_p = sub.add_parser(
        "model-registry",
        help="Inspect and manage the PostgresModelRegistry (R-GOV-02).",
    )
    mr_sub = mr_p.add_subparsers(dest="mr_command", required=True)
    list_p = mr_sub.add_parser("list", help="List all registered models.")
    _bind_model_registry(list_p)

    promote_p = mr_sub.add_parser("promote", help="Register + promote a new version.")
    promote_p.add_argument("--name", required=True)
    promote_p.add_argument("--version", required=True)
    promote_p.add_argument("--engine-version", required=True)
    promote_p.add_argument("--feature-set-version", required=True)
    promote_p.add_argument("--config-path", type=Path, default=None)
    promote_p.add_argument("--metadata-path", type=Path, default=None)
    promote_p.add_argument("--artifact-manifest", type=Path, default=None)
    _bind_model_registry(promote_p)

    retire_p = mr_sub.add_parser("retire", help="Retire the active version of a model.")
    retire_p.add_argument("--name", required=True)
    _bind_model_registry(retire_p)

    diff_p = mr_sub.add_parser("diff", help="Diff two registered model configs.")
    diff_p.add_argument("--name", required=True)
    diff_p.add_argument("--from-version", required=True)
    diff_p.add_argument("--to-version", required=True)
    _bind_model_registry(diff_p)

    rollback_p = mr_sub.add_parser(
        "rollback",
        help="Restore a prior version as the active row in one transaction.",
    )
    rollback_p.add_argument("--name", required=True)
    rollback_p.add_argument("--to-version", required=True)
    _bind_model_registry(rollback_p)


def _register_boosting(sub: Any) -> None:
    boost_p = sub.add_parser(
        "boosting",
        help="XGBoost boosted-tree training and GPU diagnostics.",
    )
    boost_sub = boost_p.add_subparsers(dest="boosting_command", required=True)
    gpu_p = boost_sub.add_parser(
        "gpu-check",
        help="Report NVIDIA/WSL visibility and run a tiny XGBoost CUDA smoke test.",
    )
    _bind_boosting(gpu_p)
    train_p = boost_sub.add_parser(
        "train",
        help="Train an XGBoost pairwise ranker from supervised feature samples.",
    )
    train_p.add_argument("--samples", required=True, type=Path)
    train_p.add_argument("--model-version", required=True)
    train_p.add_argument("--feature-set-version", default="1.0.0")
    train_p.add_argument("--output-root", type=Path, default=Path("data/models/xgboost"))
    train_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_p.add_argument("--require-gpu", action="store_true")
    train_p.add_argument("--validation-fraction", type=float, default=0.20)
    train_p.add_argument("--purge-days", type=int, default=21)
    train_p.add_argument("--num-boost-round", type=int, default=100)
    train_p.add_argument("--early-stopping-rounds", type=int, default=10)
    train_p.add_argument("--max-depth", type=int, default=4)
    train_p.add_argument("--eta", type=float, default=0.05)
    train_p.add_argument("--subsample", type=float, default=0.80)
    train_p.add_argument("--colsample-bytree", type=float, default=0.80)
    train_p.add_argument("--min-child-weight", type=float, default=1.0)
    train_p.add_argument("--random-seed", type=int, default=17)
    _bind_boosting(train_p)


def _register_alpha(sub: Any) -> None:
    alpha_p = sub.add_parser(
        "alpha",
        help="Governed XGBoost/text alpha promotion and live-ramp utilities.",
    )
    alpha_sub = alpha_p.add_subparsers(dest="alpha_command", required=True)
    alpha_assert_p = alpha_sub.add_parser("assert", help="Assert alpha promotion readiness.")
    alpha_assert_p.add_argument("--signal-name", required=True)
    alpha_assert_p.add_argument(
        "--signal-type",
        choices=list(ALPHA_SOURCE_TYPES),
        required=True,
    )
    alpha_assert_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    alpha_assert_p.add_argument("--artifact-manifest", type=Path, default=None)
    _bind_alpha(alpha_assert_p)

    alpha_promote_p = alpha_sub.add_parser("promote", help="Promote an alpha source model.")
    alpha_promote_p.add_argument("--signal-name", required=True)
    alpha_promote_p.add_argument(
        "--signal-type",
        choices=list(ALPHA_SOURCE_TYPES),
        required=True,
    )
    alpha_promote_p.add_argument("--model-version", required=True)
    alpha_promote_p.add_argument("--feature-set-version", required=True)
    alpha_promote_p.add_argument("--engine-version", required=True)
    alpha_promote_p.add_argument("--artifact-manifest", type=Path, default=None)
    alpha_promote_p.add_argument("--rollback-target", default="")
    alpha_promote_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    _bind_alpha(alpha_promote_p)

    alpha_rollback_p = alpha_sub.add_parser("rollback", help="Rollback an alpha source model.")
    alpha_rollback_p.add_argument("--signal-name", required=True)
    alpha_rollback_p.add_argument("--target-version", required=True)
    alpha_rollback_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    _bind_alpha(alpha_rollback_p)

    alpha_forecast_p = alpha_sub.add_parser(
        "materialize-forecasts",
        help="Materialize current promoted-source forecast evidence.",
    )
    alpha_forecast_p.add_argument("--contracts-file", required=True, type=Path)
    alpha_forecast_p.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    alpha_forecast_p.add_argument(
        "--source",
        action="append",
        choices=["text", "event", "intraday", "xgboost"],
        required=True,
    )
    alpha_forecast_p.add_argument("--horizon", default="21d")
    alpha_forecast_p.add_argument("--xgboost-manifest", type=Path, default=None)
    alpha_forecast_p.add_argument("--fail-on-missing", action="store_true")
    _bind_alpha(alpha_forecast_p)

    alpha_ramp_p = alpha_sub.add_parser("ramp", help="Compute allowed alpha live-ramp level.")
    alpha_ramp_p.add_argument("--clean-live-days", required=True, type=int)
    _bind_alpha(alpha_ramp_p)


def _register_walk_forward(sub: Any) -> None:
    wf_p = sub.add_parser(
        "walk-forward",
        help="Run purged sample-based walk-forward research evaluations.",
    )
    wf_sub = wf_p.add_subparsers(dest="walk_forward_command", required=True)
    run_p = wf_sub.add_parser("run", help="Evaluate supervised alpha samples.")
    run_p.add_argument("--samples", required=True, type=Path)
    run_p.add_argument("--output-root", type=Path, default=None)
    run_p.add_argument("--model-version", required=True)
    run_p.add_argument("--feature-set-version", default="1.0.0")
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
    bind_command(
        run_p,
        use_case_name="research.walk_forward",
        request_factory=walk_forward_request,
        request_type=WalkForwardRequest,
    )


def _bind_model_registry(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="research.model_registry",
        request_factory=model_registry_request,
        request_type=ModelRegistryRequest,
    )


def _bind_boosting(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="research.boosting",
        request_factory=boosting_request,
        request_type=BoostingRequest,
    )


def _bind_alpha(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="research.alpha",
        request_factory=alpha_request,
        request_type=AlphaRequest,
    )


__all__ = ["register"]
