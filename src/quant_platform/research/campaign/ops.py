"""Research-campaign operation dispatcher."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.campaign.run import _research_campaign_run

if TYPE_CHECKING:
    from quant_platform.application.research import (
        CampaignPromoteRequest,
        CampaignRequest,
        CampaignScreenRequest,
    )
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings

__all__ = ["_research_campaign", "_research_campaign_run"]


async def _research_campaign(
    settings: PlatformSettings,
    request: CampaignRequest,
) -> UseCaseResult[dict[str, object]]:
    """Dispatch end-to-end paper research campaign workflows."""
    from quant_platform.application.research import (
        CampaignAttributeFailuresRequest,
        CampaignDiagnoseFeaturesRequest,
        CampaignPromoteRequest,
        CampaignRunRequest,
        CampaignScreenRequest,
    )

    if isinstance(request, CampaignRunRequest):
        return await _research_campaign_run(settings, request)
    if isinstance(request, CampaignDiagnoseFeaturesRequest):
        from quant_platform.research.campaign import diagnostics

        return await diagnostics._research_campaign_diagnose_features(settings, request)
    if isinstance(request, CampaignAttributeFailuresRequest):
        from quant_platform.research.campaign import diagnostics

        return await diagnostics._research_campaign_attribute_feature_failures(settings, request)
    if isinstance(request, CampaignScreenRequest):
        return await _dispatch_screen(settings, request)
    if isinstance(request, CampaignPromoteRequest):
        return await _dispatch_promote(settings, request)
    raise OperatorUsageError(f"unknown research-campaign request: {type(request).__name__}")


async def _dispatch_screen(
    settings: PlatformSettings, request: CampaignScreenRequest
) -> UseCaseResult[dict[str, object]]:
    if request.command == "screen-text-candidates":
        from quant_platform.research.campaign import candidate_screens

        return await candidate_screens._research_campaign_screen_text_candidates(settings, request)
    if request.command == "screen-event-candidates":
        from quant_platform.research.campaign import candidate_screens

        return await candidate_screens._research_campaign_screen_event_candidates(settings, request)
    if request.command == "screen-intraday-candidates":
        from quant_platform.research.campaign import intraday_screen

        return await intraday_screen._research_campaign_screen_intraday_candidates(
            settings, request
        )
    raise OperatorUsageError(f"unknown research-campaign subcommand: {request.command}")


async def _dispatch_promote(
    settings: PlatformSettings, request: CampaignPromoteRequest
) -> UseCaseResult[dict[str, object]]:
    if request.command == "promote-text-candidates":
        from quant_platform.research.campaign import text_promotion

        return await text_promotion._research_campaign_promote_text_candidates(settings, request)
    if request.command == "promote-event-candidates":
        from quant_platform.research.campaign import event_promotion

        return await event_promotion._research_campaign_promote_event_candidates(settings, request)
    if request.command == "promote-intraday-candidates":
        from quant_platform.research.campaign import intraday_promotion

        return await intraday_promotion._research_campaign_promote_intraday_candidates(
            settings, request
        )
    raise OperatorUsageError(f"unknown research-campaign subcommand: {request.command}")
