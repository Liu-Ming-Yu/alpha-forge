"""Registration for research-campaign candidate screen and promotion commands."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from quant_platform.application.research import CampaignPromoteRequest, CampaignScreenRequest
from quant_platform.cli.commands.research.request_factories import (
    campaign_promote_request,
    campaign_screen_request,
)
from quant_platform.cli.registry import bind_command


def register_campaign_candidate_commands(rc_sub: Any) -> None:
    _register_text_screen(rc_sub)
    _register_text_promotion(rc_sub)
    _register_event_screen(rc_sub)
    _register_event_promotion(rc_sub)
    _register_intraday_screen(rc_sub)
    _register_intraday_promotion(rc_sub)


def _bind_screen(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="research.campaign",
        request_factory=campaign_screen_request,
        request_type=CampaignScreenRequest,
    )


def _bind_promote(parser: Any) -> None:
    bind_command(
        parser,
        use_case_name="research.campaign",
        request_factory=campaign_promote_request,
        request_type=CampaignPromoteRequest,
    )


def _register_text_screen(rc_sub: Any) -> None:
    screen_p = rc_sub.add_parser(
        "screen-text-candidates",
        help="Screen prospective text candidates before registering a new feature set.",
    )
    _add_common_screen_args(screen_p)
    screen_p.add_argument("--source-data-manifest", type=Path, required=True)
    screen_p.add_argument("--text-feature-set-version", default="text-v5")
    screen_p.add_argument("--promoted-feature-set-version", default="paper-alpha-catalyst-v10")
    screen_p.add_argument("--candidate-family", default="pre-v8")
    screen_p.add_argument(
        "--candidate-set",
        choices=("v10-alpha-quality",),
        default="v10-alpha-quality",
    )
    screen_p.add_argument("--lookback-days", type=int, default=21)
    _add_screen_threshold_args(screen_p)
    _bind_screen(screen_p)


def _register_event_screen(rc_sub: Any) -> None:
    event_screen_p = rc_sub.add_parser(
        "screen-event-candidates",
        help="Screen SEC-event/price-reaction candidates before registering a feature set.",
    )
    _add_common_screen_args(event_screen_p)
    event_screen_p.add_argument("--source-data-manifest", type=Path, required=True)
    event_screen_p.add_argument(
        "--event-feature-set-version",
        default="paper-alpha-event-reaction-v2",
    )
    event_screen_p.add_argument("--candidate-family", default="event-reaction-v2")
    event_screen_p.add_argument(
        "--candidate-set",
        choices=("seed", "event-reaction-v2"),
        default="seed",
    )
    _add_screen_threshold_args(event_screen_p)
    _bind_screen(event_screen_p)


def _register_intraday_screen(rc_sub: Any) -> None:
    intraday_screen_p = rc_sub.add_parser(
        "screen-intraday-candidates",
        help="Screen 1-minute intraday microstructure candidates before registering a feature set.",
    )
    _add_common_screen_args(intraday_screen_p)
    intraday_screen_p.add_argument("--contracts-file", type=Path, required=True)
    intraday_screen_p.add_argument(
        "--intraday-file",
        action="append",
        required=True,
        help="Repeatable local intraday input in vendor=/path/to/file form.",
    )
    intraday_screen_p.add_argument(
        "--intraday-feature-set-version",
        default="paper-alpha-intraday-microstructure-v2",
    )
    intraday_screen_p.add_argument("--candidate-family", default="intraday-microstructure-v2")
    intraday_screen_p.add_argument(
        "--candidate-set",
        choices=("seed", "microstructure-v2"),
        default="seed",
    )
    _add_screen_threshold_args(intraday_screen_p)
    _bind_screen(intraday_screen_p)


def _register_text_promotion(rc_sub: Any) -> None:
    parser = rc_sub.add_parser(
        "promote-text-candidates",
        help="Promote only text candidates shared by main, confirmation, and full screens.",
    )
    _add_promotion_args(
        parser,
        feature_card_dir=Path("infra/config/feature_cards/paper-alpha-catalyst-v10"),
        feature_family_file=Path("infra/config/feature_families/paper-alpha-catalyst-v10.json"),
    )
    _bind_promote(parser)


def _register_event_promotion(rc_sub: Any) -> None:
    parser = rc_sub.add_parser(
        "promote-event-candidates",
        help="Promote only candidates shared by main, confirmation, and full event screens.",
    )
    _add_promotion_args(
        parser,
        feature_card_dir=Path("infra/config/feature_cards/paper-alpha-event-reaction-v2"),
        feature_family_file=Path(
            "infra/config/feature_families/paper-alpha-event-reaction-v2.json"
        ),
    )
    _bind_promote(parser)


def _register_intraday_promotion(rc_sub: Any) -> None:
    parser = rc_sub.add_parser(
        "promote-intraday-candidates",
        help="Promote only candidates shared by main, confirmation, and full intraday screens.",
    )
    _add_promotion_args(
        parser,
        feature_card_dir=Path("infra/config/feature_cards/paper-alpha-intraday-microstructure-v2"),
        feature_family_file=Path(
            "infra/config/feature_families/paper-alpha-intraday-microstructure-v2.json"
        ),
    )
    _bind_promote(parser)


def _add_common_screen_args(parser: Any) -> None:
    parser.add_argument("--samples-file", type=Path, required=True)
    parser.add_argument("--sample-build-summary", type=Path, required=True)
    parser.add_argument("--sample-start", type=datetime.fromisoformat, default=None)
    parser.add_argument("--sample-end", type=datetime.fromisoformat, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--screen-name", default="")


def _add_screen_threshold_args(parser: Any) -> None:
    parser.add_argument("--min-source-density", type=float, default=0.05)
    parser.add_argument("--min-null-margin", type=float, default=0.0)
    parser.add_argument("--min-ic-mean", type=float, default=0.02)
    parser.add_argument("--min-icir", type=float, default=0.10)
    parser.add_argument("--max-negative-ic-streak", type=int, default=3)
    parser.add_argument("--min-passing-candidates", type=int, default=3)
    parser.add_argument("--permutation-seed", type=int, default=17)
    parser.add_argument("--permutation-count", type=int, default=200)


def _add_promotion_args(
    parser: Any,
    *,
    feature_card_dir: Path,
    feature_family_file: Path,
) -> None:
    parser.add_argument("--main-screen", type=Path, required=True)
    parser.add_argument("--confirmation-screen", type=Path, required=True)
    parser.add_argument("--full-screen", type=Path, required=True)
    parser.add_argument("--feature-card-dir", type=Path, default=feature_card_dir)
    parser.add_argument("--feature-family-file", type=Path, default=feature_family_file)
    parser.add_argument("--min-passing-candidates", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--screen-name", default="")


__all__ = ["register_campaign_candidate_commands"]
