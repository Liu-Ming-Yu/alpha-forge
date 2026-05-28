"""XGBoost training helper for research campaigns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_platform.services.research_service.boosting import BoostingTrainConfig
from quant_platform.services.research_service.boosting.search import (
    train_xgboost_ranker_with_search,
)
from quant_platform.services.research_service.campaigns.evaluation.feature_admission import (
    supervised_to_boosting_samples,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from quant_platform.services.research_service.boosting.artifacts import (
        BoostingDevice,
        BoostingManifest,
    )
    from quant_platform.services.research_service.boosting.search import XGBoostSearchMode
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def train_admitted_xgboost_ranker(
    *,
    samples: Sequence[SupervisedAlphaSample],
    admitted_features: Sequence[str],
    model_version: str,
    feature_set_version: str,
    feature_versions: Mapping[str, str],
    output_root: Path,
    device: BoostingDevice,
    require_gpu: bool,
    purge_days: int,
    search_mode: XGBoostSearchMode,
) -> tuple[BoostingManifest, Path, list[dict[str, object]]]:
    """Train XGBoost on admitted audited features only."""
    config = BoostingTrainConfig(
        model_version=model_version,
        feature_set_version=feature_set_version,
        output_root=output_root / "models" / "xgboost",
        device=device,
        require_gpu=require_gpu,
        purge_days=purge_days,
        feature_versions=dict(feature_versions),
    )
    return train_xgboost_ranker_with_search(
        supervised_to_boosting_samples(samples, admitted_features),
        config,
        search_mode=search_mode,
    )


__all__ = ["train_admitted_xgboost_ranker"]
