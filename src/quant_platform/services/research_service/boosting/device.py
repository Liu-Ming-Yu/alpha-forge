"""XGBoost runtime/device diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from quant_platform.services.research_service.boosting.artifacts import BoostingDevice


def import_xgboost() -> object:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - exercised in environments without ml extra
        raise RuntimeError(
            "xgboost is required for boosted-tree training/scoring. "
            "Install the project with the ml extra: pip install -e '.[ml]'."
        ) from exc
    return xgb


def resolve_xgboost_device(
    requested: BoostingDevice,
    *,
    require_gpu: bool = False,
    xgb: object | None = None,
) -> str:
    """Resolve ``auto`` to ``cuda`` or ``cpu`` via XGBoost's own CUDA probe."""
    if requested == "cpu":
        return "cpu"

    xgb_mod = xgb if xgb is not None else import_xgboost()

    if requested in ("cuda", "auto"):
        ok, detail = cuda_smoke(xgb_mod)
        if ok:
            return "cuda"
        if requested == "cuda":
            raise RuntimeError(f"XGBoost CUDA device requested but unavailable: {detail}")
        if require_gpu:
            raise RuntimeError(f"XGBoost CUDA device required but unavailable: {detail}")

    return "cpu"


def gpu_check() -> dict[str, object]:
    """Return a JSON-serialisable GPU diagnostic for operators."""
    result: dict[str, object] = {"nvidia_smi": nvidia_smi_status()}
    try:
        xgb = import_xgboost()
    except RuntimeError as exc:
        result["xgboost"] = {"available": False, "detail": str(exc)}
        result["cuda_smoke"] = {"ok": False, "detail": "xgboost unavailable"}
        return result
    result["xgboost"] = {"available": True, "version": getattr(xgb, "__version__", "unknown")}
    ok, detail = cuda_smoke(xgb)
    result["cuda_smoke"] = {"ok": ok, "detail": detail}
    return result


def nvidia_smi_status() -> dict[str, object]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {"available": False, "detail": "nvidia-smi not found"}
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return {"available": False, "detail": "nvidia-smi timed out"}
    detail = (proc.stdout or proc.stderr).strip()
    return {
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "detail": detail,
    }


def cuda_smoke(xgb: object) -> tuple[bool, str]:
    """Run a tiny XGBoost CUDA training pass."""
    xgb_any = cast("Any", xgb)
    try:
        x_values = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.9, 0.1]], dtype=float)
        y_values = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float)
        dmat = xgb_any.DMatrix(x_values, label=y_values, feature_names=["a", "b"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xgb_any.train(
                {"objective": "reg:squarederror", "tree_method": "hist", "device": "cuda"},
                dmat,
                num_boost_round=1,
                verbose_eval=False,
            )
        warning_text = "\n".join(str(item.message) for item in caught)
        if "No visible GPU" in warning_text or "Device is changed from GPU to CPU" in warning_text:
            return False, warning_text
        return True, "cuda smoke train completed"
    except Exception as exc:  # pragma: no cover - depends on host GPU state
        return False, str(exc)
