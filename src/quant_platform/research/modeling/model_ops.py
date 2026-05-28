"""Model-registry and boosted-tree research operation wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from quant_platform.application.errors import OperatorUsageError
from quant_platform.bootstrap.governance.repositories import build_model_registry
from quant_platform.research.common import (
    _verify_postgres_schema_if_configured,
    research_json_result,
)

if TYPE_CHECKING:
    import argparse

    from quant_platform.application.research import BoostingRequest, ModelRegistryRequest
    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


async def _model_registry(
    settings: PlatformSettings,
    request: ModelRegistryRequest,
) -> None:
    """Dispatch to the ``quant-platform model-registry`` subcommand."""
    from quant_platform.services.research_service.modeling.registry.cli import (
        dispatch,
    )

    await _verify_postgres_schema_if_configured(settings)
    # ModelRegistryRequest is structurally compatible with the namespace the
    # service-layer dispatch reads (name/version/.../to_version attributes).
    await dispatch(
        settings=settings,
        subcommand=request.command,
        args=cast("argparse.Namespace", request),
        registry=build_model_registry(settings.storage.postgres_dsn),
    )


async def _boosting(
    settings: PlatformSettings,
    request: BoostingRequest,
) -> UseCaseResult[dict[str, object]]:
    """Dispatch boosted-tree training and GPU diagnostics."""
    del settings
    from quant_platform.services.research_service.boosting import (
        BoostingTrainConfig,
        gpu_check,
        load_samples,
        train_xgboost_ranker,
    )

    if request.command == "gpu-check":
        return research_json_result({str(k): v for k, v in gpu_check().items()})
    if request.command == "train":
        if request.samples is None:
            raise OperatorUsageError("boosting train requires --samples")
        samples = load_samples(request.samples)
        manifest, manifest_path = train_xgboost_ranker(
            samples,
            BoostingTrainConfig(
                model_version=request.model_version,
                feature_set_version=request.feature_set_version,
                output_root=request.output_root,
                device=request.device,
                require_gpu=request.require_gpu,
                validation_fraction=request.validation_fraction,
                purge_days=request.purge_days,
                num_boost_round=request.num_boost_round,
                early_stopping_rounds=request.early_stopping_rounds,
                max_depth=request.max_depth,
                eta=request.eta,
                subsample=request.subsample,
                colsample_bytree=request.colsample_bytree,
                min_child_weight=request.min_child_weight,
                random_seed=request.random_seed,
            ),
        )
        return research_json_result(
            {
                "manifest_path": str(manifest_path),
                "model_version": manifest.model_version,
                "feature_set_version": manifest.feature_set_version,
                "feature_schema_hash": manifest.feature_schema_hash,
                "device": manifest.device,
                "validation_ic": manifest.metrics.get("validation_ic"),
            }
        )
    raise OperatorUsageError(f"unknown boosting subcommand: {request.command}")
