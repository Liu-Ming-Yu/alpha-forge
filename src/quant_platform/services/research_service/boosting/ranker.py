"""XGBoost pairwise-ranker training orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import structlog

from quant_platform.services.research_service.boosting.artifacts import (
    BoostingManifest,
    BoostingSample,
    BoostingTrainConfig,
    feature_schema_hash,
)
from quant_platform.services.research_service.boosting.artifacts import (
    sha256_file as _sha256_file,
)
from quant_platform.services.research_service.boosting.device import (
    import_xgboost,
    resolve_xgboost_device,
)
from quant_platform.services.research_service.boosting.training import (
    build_matrix,
    infer_feature_names,
    split_samples,
    validation_ic,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

log = structlog.get_logger(__name__)


def train_xgboost_ranker(
    samples: Sequence[BoostingSample],
    config: BoostingTrainConfig,
) -> tuple[BoostingManifest, Path]:
    """Train an XGBoost pairwise ranker and write model artifacts."""
    if not samples:
        raise ValueError("boosting training requires at least one sample")
    xgb = import_xgboost()
    xgb_any = cast("Any", xgb)
    device = resolve_xgboost_device(config.device, require_gpu=config.require_gpu, xgb=xgb)
    feature_names = infer_feature_names(samples)
    train, valid, validation_start = split_samples(
        samples,
        validation_fraction=config.validation_fraction,
        purge_days=config.purge_days,
    )

    train_matrix = build_matrix(xgb, train, feature_names)
    valid_matrix = build_matrix(xgb, valid, feature_names)
    params = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        # Relevance labels are per-date rank indices (0..N-1); with a large
        # universe N exceeds 31, which XGBoost's exponential NDCG gain rejects.
        # Linear DCG gain handles rank-index relevance correctly.
        "ndcg_exp_gain": False,
        "tree_method": "hist",
        "device": device,
        "eta": config.eta,
        "max_depth": config.max_depth,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "seed": config.random_seed,
    }
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb_any.train(
        params,
        train_matrix,
        num_boost_round=config.num_boost_round,
        evals=[(train_matrix, "train"), (valid_matrix, "validation")],
        early_stopping_rounds=config.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False,
    )

    output_dir = config.output_root / config.model_version
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    booster.save_model(str(model_path))
    booster_hash = _sha256_file(model_path)

    metrics: dict[str, object] = {
        "train_samples": len(train),
        "validation_samples": len(valid),
        "train_groups": len({sample.as_of for sample in train}),
        "validation_groups": len({sample.as_of for sample in valid}),
        "feature_coverage": 1.0,
        "validation_start": validation_start.astimezone(UTC).isoformat(),
        "purge_days": config.purge_days,
        "validation_ic": validation_ic(xgb, booster, valid, valid_matrix),
        "best_iteration": getattr(booster, "best_iteration", None),
        "evals_result": evals_result,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    manifest = BoostingManifest(
        model_type="xgboost_ranker",
        model_version=config.model_version,
        feature_set_version=config.feature_set_version,
        booster_path="model.json",
        feature_names=feature_names,
        feature_schema_hash=feature_schema_hash(feature_names),
        xgboost_version=str(getattr(xgb, "__version__", "unknown")),
        objective="rank:pairwise",
        device=device,
        trained_at=datetime.now(tz=UTC).isoformat(),
        metrics_path="metrics.json",
        booster_sha256=booster_hash,
        random_seed=config.random_seed,
        metrics=metrics,
        feature_versions={
            name: str(config.feature_versions.get(name, config.feature_set_version))
            for name in feature_names
        },
    )
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    log.info(
        "boosting.train.complete",
        model_version=config.model_version,
        output_dir=str(output_dir),
        device=device,
        validation_ic=metrics["validation_ic"],
    )
    return manifest, manifest_path
