"""Research campaign command registration."""

from __future__ import annotations

from typing import Any

from quant_platform.cli.commands.research.campaign.candidates import (
    register_campaign_candidate_commands,
)
from quant_platform.cli.commands.research.campaign.diagnostics import (
    register_campaign_diagnostics,
)
from quant_platform.cli.commands.research.campaign.run import register_campaign_run


def register_research_campaign(sub: Any) -> None:
    rc_p = sub.add_parser(
        "research-campaign",
        help="Run an end-to-end paper research campaign.",
    )
    rc_sub = rc_p.add_subparsers(dest="research_campaign_command", required=True)
    register_campaign_run(rc_sub)
    register_campaign_diagnostics(rc_sub)
    register_campaign_candidate_commands(rc_sub)


research_campaign = "research.campaign"

__all__ = ["register_research_campaign", "research_campaign"]
