"""Feature-audit helpers for paper research campaigns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_platform.config import PlatformSettings

__all__ = ["_run_campaign_feature_audits"]


async def _run_campaign_feature_audits(
    *,
    settings: PlatformSettings,
    samples: Sequence[Any],
    feature_set_version: str,
    horizon_days: int,
    slippage_bps_per_turnover: float,
    mode: str,
    feature_card_dir: Path | None,
) -> list[dict[str, object]]:
    """Audit every candidate feature before the campaign model gate."""
    from quant_platform.application.features.governance import CampaignFeatureAuditRequest
    from quant_platform.research.feature_governance import build_feature_audit_use_case

    candidate_feature_names = _candidate_feature_names(
        feature_set_version=feature_set_version,
        mode=mode,
        feature_card_dir=feature_card_dir,
    )
    return await build_feature_audit_use_case(settings).audit_campaign_features(
        CampaignFeatureAuditRequest(
            samples=samples,
            feature_set_version=feature_set_version,
            horizon_days=horizon_days,
            slippage_bps_per_turnover=slippage_bps_per_turnover,
            mode=mode,
            feature_card_dir=feature_card_dir,
            candidate_feature_names=candidate_feature_names,
        )
    )


def _candidate_feature_names(
    *,
    feature_set_version: str,
    mode: str,
    feature_card_dir: Path | None,
) -> tuple[str, ...] | None:
    _ = feature_set_version
    if mode != "paper" or feature_card_dir is None:
        return None
    return tuple(sorted(path.stem for path in feature_card_dir.glob("*.json")))
