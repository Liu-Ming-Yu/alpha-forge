"""Sample-based research campaign workflow API."""

from __future__ import annotations

from quant_platform.services.research_service.campaigns.evaluation.artifacts import (
    current_git_commit,
    list_campaign_manifests,
    read_campaign_manifest,
    write_campaign_manifest,
    write_model_comparison,
    write_walk_forward_artifacts,
)
from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.sampling.factory_metrics import (
    calibrated_slippage_bps_per_turnover,
)
from quant_platform.services.research_service.sampling.factory_models import (
    AlphaEligibilityThresholds,
    ResearchCampaignManifest,
    WalkForwardEvidence,
)
from quant_platform.services.research_service.sampling.sample_io import (
    load_supervised_samples,
    walk_forward_object_root,
)

__all__ = [
    "AlphaEligibilityThresholds",
    "ResearchCampaignManifest",
    "WalkForwardEvidence",
    "calibrated_slippage_bps_per_turnover",
    "current_git_commit",
    "list_campaign_manifests",
    "load_supervised_samples",
    "read_campaign_manifest",
    "run_sample_walk_forward",
    "walk_forward_object_root",
    "write_campaign_manifest",
    "write_model_comparison",
    "write_walk_forward_artifacts",
]
