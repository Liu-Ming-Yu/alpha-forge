"""Research feature command registrations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.application.research import (
    FeatureAuditRequest,
    FeaturesBackfillIntradayAlphaRequest,
    FeaturesBackfillRequest,
    FeaturesBuildSamplesRequest,
    FeaturesRetentionRequest,
)
from quant_platform.cli.commands.research.request_factories import (
    feature_audit_request,
    features_backfill_intraday_alpha_request,
    features_backfill_request,
    features_build_samples_request,
    features_retention_request,
)
from quant_platform.cli.registry import bind_command


def register_features(sub: Any) -> None:
    fr_p = sub.add_parser("features", help="Feature repository utilities.")
    fr_sub = fr_p.add_subparsers(dest="features_command", required=True)
    samples_p = fr_sub.add_parser(
        "build-samples",
        help="Build supervised alpha samples from feature vectors and forward returns.",
    )
    samples_p.add_argument("--contracts-file", required=True)
    samples_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    samples_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    samples_p.add_argument("--output", required=True, type=Path)
    samples_p.add_argument("--feature-set-version", default="1.0.0")
    samples_p.add_argument(
        "--date-policy",
        choices=["nyse-sessions", "calendar-days"],
        default="nyse-sessions",
    )
    samples_p.add_argument("--horizon-days", type=int, default=21)
    samples_p.add_argument("--bar-seconds", type=int, default=86400)
    samples_p.add_argument("--max-feature-age-days", type=int, default=3)
    bind_command(
        samples_p,
        use_case_name="research.features",
        request_factory=features_build_samples_request,
        request_type=FeaturesBuildSamplesRequest,
    )

    backfill_p = fr_sub.add_parser(
        "backfill",
        help="Backfill durable historical feature vectors for a contracts universe.",
    )
    backfill_p.add_argument("--contracts-file", required=True)
    backfill_p.add_argument("--start", required=True, type=datetime.fromisoformat)
    backfill_p.add_argument("--end", required=True, type=datetime.fromisoformat)
    backfill_p.add_argument("--feature-set-version", default="1.0.0")
    backfill_p.add_argument(
        "--date-policy",
        choices=["nyse-sessions", "calendar-days"],
        default="nyse-sessions",
    )
    backfill_p.add_argument("--bar-seconds", type=int, default=86400)
    backfill_p.add_argument("--lookback-days", type=int, default=380)
    backfill_p.add_argument("--source-data-manifest", type=Path, default=None)
    backfill_p.add_argument("--dry-run", action="store_true")
    bind_command(
        backfill_p,
        use_case_name="research.features",
        request_factory=features_backfill_request,
        request_type=FeaturesBackfillRequest,
    )

    intraday_backfill_p = fr_sub.add_parser(
        "backfill-intraday-alpha",
        help="Backfill immutable paper intraday-alpha feature vectors from frozen 1-minute bars.",
    )
    intraday_backfill_p.add_argument("--samples-file", type=Path, default=None)
    intraday_backfill_p.add_argument("--contracts-file", type=Path, required=True)
    intraday_backfill_p.add_argument("--start", type=datetime.fromisoformat, default=None)
    intraday_backfill_p.add_argument("--end", type=datetime.fromisoformat, default=None)
    intraday_backfill_p.add_argument("--context-feature-set-version", default="")
    intraday_backfill_p.add_argument(
        "--date-policy",
        choices=["nyse-sessions", "calendar-days"],
        default="nyse-sessions",
    )
    intraday_backfill_p.add_argument(
        "--intraday-file",
        action="append",
        required=True,
        help="Repeatable local intraday input in vendor=/path/to/file form.",
    )
    intraday_backfill_p.add_argument("--feature-family-file", type=Path, required=True)
    intraday_backfill_p.add_argument(
        "--feature-set-version",
        default="paper-alpha-intraday-microstructure-v2",
    )
    intraday_backfill_p.add_argument(
        "--candidate-set",
        choices=("microstructure-v2",),
        default="microstructure-v2",
    )
    intraday_backfill_p.add_argument("--artifact-uri", default="")
    intraday_backfill_p.add_argument("--dry-run", action="store_true")
    bind_command(
        intraday_backfill_p,
        use_case_name="research.features",
        request_factory=features_backfill_intraday_alpha_request,
        request_type=FeaturesBackfillIntradayAlphaRequest,
    )

    ret_p = fr_sub.add_parser(
        "retention",
        help="Prune feature-vector rows older than --keep-days (retires R-DAT-03).",
    )
    ret_p.add_argument("--keep-days", type=int, required=True)
    ret_p.add_argument("--dry-run", action="store_true")
    bind_command(
        ret_p,
        use_case_name="research.features_retention",
        request_factory=features_retention_request,
        request_type=FeaturesRetentionRequest,
    )

    audit_p = fr_sub.add_parser(
        "audit",
        help="Run and inspect institutional feature quality gates.",
    )
    audit_sub = audit_p.add_subparsers(dest="feature_audit_command", required=True)
    audit_run_p = audit_sub.add_parser("run", help="Run six-gate feature audit.")
    audit_run_p.add_argument("--feature-card", required=True, type=Path)
    audit_run_p.add_argument("--samples", type=Path, default=None)
    audit_run_p.add_argument("--contracts-file", default=None)
    audit_run_p.add_argument("--start", type=datetime.fromisoformat, default=None)
    audit_run_p.add_argument("--end", type=datetime.fromisoformat, default=None)
    audit_run_p.add_argument("--feature-set-version", default="1.0.0")
    audit_run_p.add_argument("--horizon-days", type=int, default=21)
    audit_run_p.add_argument("--bar-seconds", type=int, default=86400)
    audit_run_p.add_argument("--max-feature-age-days", type=int, default=3)
    audit_run_p.add_argument("--output-root", type=Path, default=None)
    audit_run_p.add_argument("--baseline-features", default="")
    audit_run_p.add_argument("--slippage-bps-per-turnover", type=float, default=10.0)
    audit_run_p.add_argument("--min-daily-groups", type=int, default=252)
    audit_run_p.add_argument("--min-coverage", type=float, default=0.95)
    audit_run_p.add_argument("--min-oos-ic", type=float, default=0.02)
    audit_run_p.add_argument("--min-icir", type=float, default=0.10)
    audit_run_p.add_argument("--max-negative-ic-streak", type=int, default=3)
    audit_run_p.add_argument("--max-turnover", type=float, default=4.0)
    audit_run_p.add_argument("--persist", action="store_true")
    _bind_audit(audit_run_p)

    audit_status_p = audit_sub.add_parser("status", help="List latest feature audit results.")
    audit_status_p.add_argument("--feature-name", default=None)
    audit_status_p.add_argument("--feature-version", default=None)
    audit_status_p.add_argument("--limit", type=int, default=20)
    audit_status_p.add_argument("--output-root", type=Path, default=None)
    _bind_audit(audit_status_p)

    audit_assert_p = audit_sub.add_parser("assert", help="Fail closed unless audit passed.")
    audit_assert_p.add_argument("--manifest", type=Path, default=None)
    audit_assert_p.add_argument("--feature-name", default=None)
    audit_assert_p.add_argument("--feature-version", default=None)
    audit_assert_p.add_argument(
        "--minimum-state",
        choices=["shadow", "paper", "live"],
        default="paper",
    )
    _bind_audit(audit_assert_p)

    audit_retire_p = audit_sub.add_parser("retire", help="Persist a retired feature audit row.")
    audit_retire_p.add_argument("--feature-name", required=True)
    audit_retire_p.add_argument("--feature-version", required=True)
    audit_retire_p.add_argument("--feature-set-version", default="1.0.0")
    audit_retire_p.add_argument("--reason", default="operator retired")
    _bind_audit(audit_retire_p)


def _bind_audit(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="research.features",
        request_factory=feature_audit_request,
        request_type=FeatureAuditRequest,
    )


__all__ = ["register_features"]
