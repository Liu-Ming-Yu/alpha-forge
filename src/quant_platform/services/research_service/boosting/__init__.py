"""Public XGBoost boosted-tree training and shadow scoring API."""

from __future__ import annotations

from quant_platform.services.research_service.boosting.artifacts import (
    BoostingDevice,
    BoostingManifest,
    BoostingSample,
    BoostingTrainConfig,
    feature_schema_hash,
    load_samples,
    load_samples_async,
)
from quant_platform.services.research_service.boosting.device import (
    cuda_smoke as _device_cuda_smoke,
)
from quant_platform.services.research_service.boosting.device import (
    gpu_check,
)
from quant_platform.services.research_service.boosting.device import (
    import_xgboost as _import_xgboost,
)
from quant_platform.services.research_service.boosting.device import (
    nvidia_smi_status as _device_nvidia_smi_status,
)
from quant_platform.services.research_service.boosting.model import XGBoostRankSignalModel
from quant_platform.services.research_service.boosting.ranker import train_xgboost_ranker
from quant_platform.services.research_service.boosting.shadow import ShadowBoostingScorer

__all__ = [
    "BoostingDevice",
    "BoostingManifest",
    "BoostingSample",
    "BoostingTrainConfig",
    "ShadowBoostingScorer",
    "XGBoostRankSignalModel",
    "feature_schema_hash",
    "gpu_check",
    "load_samples",
    "load_samples_async",
    "resolve_xgboost_device",
    "train_xgboost_ranker",
]


def _cuda_smoke(xgb: object) -> tuple[bool, str]:
    """CUDA smoke hook for tests and diagnostics."""
    return _device_cuda_smoke(xgb)


def _nvidia_smi_status() -> dict[str, object]:
    """nvidia-smi diagnostic hook."""
    return _device_nvidia_smi_status()


def resolve_xgboost_device(
    requested: BoostingDevice,
    *,
    require_gpu: bool = False,
    xgb: object | None = None,
) -> str:
    """Resolve ``auto`` to ``cuda`` or ``cpu`` via XGBoost's CUDA probe."""
    if requested == "cpu":
        return "cpu"

    xgb_mod = xgb if xgb is not None else _import_xgboost()

    if requested in ("cuda", "auto"):
        ok, detail = _cuda_smoke(xgb_mod)
        if ok:
            return "cuda"
        if requested == "cuda":
            raise RuntimeError(f"XGBoost CUDA device requested but unavailable: {detail}")
        if require_gpu:
            raise RuntimeError(f"XGBoost CUDA device required but unavailable: {detail}")

    return "cpu"
