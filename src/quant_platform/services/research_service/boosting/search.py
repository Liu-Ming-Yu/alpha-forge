"""Small governed search space for XGBoost ranker training."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from quant_platform.services.research_service.boosting.ranker import train_xgboost_ranker

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from quant_platform.services.research_service.boosting.artifacts import (
        BoostingManifest,
        BoostingSample,
        BoostingTrainConfig,
    )


XGBoostSearchMode = Literal["off", "conservative"]


def train_xgboost_ranker_with_search(
    samples: Sequence[BoostingSample],
    config: BoostingTrainConfig,
    *,
    search_mode: XGBoostSearchMode = "off",
) -> tuple[BoostingManifest, Path, list[dict[str, object]]]:
    """Train one XGBoost model or a small conservative grid and select by validation IC."""
    if search_mode == "off":
        manifest, path = train_xgboost_ranker(samples, config)
        return manifest, path, [_search_row(manifest, path, selected=True)]
    if search_mode != "conservative":
        raise ValueError(f"unsupported xgboost search mode: {search_mode}")

    variants = (
        replace(config, model_version=f"{config.model_version}__search_0"),
        replace(
            config,
            model_version=f"{config.model_version}__search_1",
            max_depth=3,
            eta=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2.0,
            num_boost_round=150,
        ),
        replace(
            config,
            model_version=f"{config.model_version}__search_2",
            max_depth=2,
            eta=0.05,
            subsample=0.70,
            colsample_bytree=0.70,
            min_child_weight=3.0,
            num_boost_round=120,
        ),
    )

    trained: list[tuple[BoostingManifest, Path]] = []
    for variant in variants:
        trained.append(train_xgboost_ranker(samples, variant))

    best = max(trained, key=lambda item: _validation_ic(item[0]))
    rows = [_search_row(manifest, path, selected=path == best[1]) for manifest, path in trained]
    return best[0], best[1], rows


def _validation_ic(manifest: BoostingManifest) -> float:
    numeric = _metric_as_float(manifest.metrics.get("validation_ic"))
    if numeric is None:
        return -math.inf
    if not math.isfinite(numeric):
        return -math.inf
    return numeric


def _metric_as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _search_row(
    manifest: BoostingManifest,
    path: Path,
    *,
    selected: bool,
) -> dict[str, object]:
    return {
        "model_version": manifest.model_version,
        "manifest_path": str(path),
        "selected": selected,
        "validation_ic": _validation_ic(manifest),
        "train_samples": _metric_as_float(manifest.metrics.get("train_samples")) or 0.0,
        "validation_samples": _metric_as_float(manifest.metrics.get("validation_samples")) or 0.0,
    }


__all__ = ["XGBoostSearchMode", "train_xgboost_ranker_with_search"]
