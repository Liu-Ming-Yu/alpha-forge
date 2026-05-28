"""Research-campaign policy checks shared by orchestration helpers."""

from __future__ import annotations

from typing import Any


def governed_feature_audit_blocker(args: Any) -> dict[str, object] | None:
    """Return a fail-closed payload when governed paper audit inputs are missing."""
    from quant_platform.services.research_service.features.paper_alpha.composite import (
        PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
        PAPER_ALPHA_INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION,
    )
    from quant_platform.services.research_service.features.paper_alpha.event import (
        PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
    )
    from quant_platform.services.research_service.features.paper_alpha.text_features import (
        PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    )

    governed = {
        PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
        PAPER_ALPHA_EVENT_REACTION_V2_FEATURE_SET_VERSION,
        PAPER_ALPHA_INTRADAY_MICROSTRUCTURE_V2_FEATURE_SET_VERSION,
        PAPER_ALPHA_COMPOSITE_FEATURE_SET_VERSION,
    }
    if args.feature_set_version not in governed:
        return None
    if args.feature_audit_mode == "paper" and args.feature_card_dir is not None:
        return None
    return {
        "passed": False,
        "reason": f"{args.feature_set_version} requires paper feature audits and feature cards",
    }


__all__ = ["governed_feature_audit_blocker"]
