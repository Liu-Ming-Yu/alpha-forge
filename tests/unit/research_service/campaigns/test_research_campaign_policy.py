"""Current governed-campaign policy tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from quant_platform.research.campaign.policy import governed_feature_audit_blocker
from quant_platform.research.campaign.source_density import CATALYST_FEATURES_BY_SET
from quant_platform.research.campaign.stability_attribution import (
    _governed_family_files,
)
from quant_platform.services.research_service.features.paper_alpha.text_features import (
    PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
    TEXT_CATALYST_V10_ALPHA_FEATURES,
)


def _campaign_args(**overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        feature_set_version=PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION,
        feature_audit_mode="paper",
        feature_card_dir=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_current_governed_campaign_requires_feature_cards() -> None:
    blocker = governed_feature_audit_blocker(_campaign_args())

    assert blocker == {
        "passed": False,
        "reason": "paper-alpha-catalyst-v10 requires paper feature audits and feature cards",
    }


def test_current_governed_campaign_accepts_paper_cards() -> None:
    blocker = governed_feature_audit_blocker(
        _campaign_args(feature_card_dir=Path("infra/config/feature_cards/paper-alpha-catalyst-v10"))
    )

    assert blocker is None


def test_current_catalyst_source_density_uses_v10_features() -> None:
    assert CATALYST_FEATURES_BY_SET[PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION] == (
        TEXT_CATALYST_V10_ALPHA_FEATURES
    )


def test_current_stability_attribution_defaults_to_v10_family_file() -> None:
    assert _governed_family_files() == {
        PAPER_ALPHA_CATALYST_V10_FEATURE_SET_VERSION: (
            "infra/config/feature_families/paper-alpha-catalyst-v10.json"
        ),
        "paper-alpha-event-reaction-v2": (
            "infra/config/feature_families/paper-alpha-event-reaction-v2.json"
        ),
        "paper-alpha-intraday-microstructure-v2": (
            "infra/config/feature_families/paper-alpha-intraday-microstructure-v2.json"
        ),
        "paper-alpha-composite-v1": "infra/config/feature_families/paper-alpha-composite-v1.json",
    }
