"""Registration for research-campaign diagnostic commands."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.application.research import (
    CampaignAttributeFailuresRequest,
    CampaignDiagnoseFeaturesRequest,
)
from quant_platform.cli.commands.research.request_factories import (
    campaign_attribute_failures_request,
    campaign_diagnose_features_request,
)
from quant_platform.cli.registry import bind_command


def register_campaign_diagnostics(rc_sub: Any) -> None:
    diag_p = rc_sub.add_parser(
        "diagnose-features",
        help="Evaluate campaign features under positive and negative orientation without training.",
    )
    _add_common_diagnostic_args(diag_p)
    diag_p.add_argument("--horizon-days", type=int, default=21)
    diag_p.add_argument("--feature-card-dir", type=Path, required=True)
    bind_command(
        diag_p,
        use_case_name="research.campaign",
        request_factory=campaign_diagnose_features_request,
        request_type=CampaignDiagnoseFeaturesRequest,
    )

    attr_p = rc_sub.add_parser(
        "attribute-feature-failures",
        help="Attribute quarantined feature failures without training or promotion artifacts.",
    )
    _add_common_diagnostic_args(attr_p)
    attr_p.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 21])
    attr_p.add_argument("--official-horizon-days", type=int, default=21)
    attr_p.add_argument("--feature-card-dir", type=Path, required=True)
    attr_p.add_argument(
        "--feature-family-file",
        type=Path,
        default=Path("infra/config/feature_families/paper-alpha-catalyst-v10.json"),
    )
    attr_p.add_argument("--permutation-seed", type=int, default=17)
    attr_p.add_argument("--permutation-count", type=int, default=200)
    attr_p.add_argument("--correlation-threshold", type=float, default=0.70)
    bind_command(
        attr_p,
        use_case_name="research.campaign",
        request_factory=campaign_attribute_failures_request,
        request_type=CampaignAttributeFailuresRequest,
    )


def _add_common_diagnostic_args(parser: Any) -> None:
    parser.add_argument("--contracts-file", required=True)
    parser.add_argument("--start", required=True, type=datetime.fromisoformat)
    parser.add_argument("--end", required=True, type=datetime.fromisoformat)
    parser.add_argument("--feature-set-version", default="paper-alpha-catalyst-v10")
    parser.add_argument(
        "--date-policy",
        choices=["nyse-sessions", "calendar-days"],
        default="nyse-sessions",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--bar-seconds", type=int, default=86400)
    parser.add_argument("--max-feature-age-days", type=int, default=3)
    parser.add_argument("--slippage-bps-per-turnover", type=float, default=10.0)
    parser.add_argument("--max-calibration-age-days", type=float, default=14.0)
    parser.add_argument("--require-calibration", action="store_true")


__all__ = ["register_campaign_diagnostics"]
