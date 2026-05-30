"""Read-only model / factor-family / alpha-library views for the console.

Feature families and the curated alpha library are introspected in-memory (no
DB needed). The model registry and auto-promoted alphas read from Postgres /
object-store when configured, degrading to empty + an ``error`` note otherwise.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quant_platform.config import PlatformSettings

_MAX_FEATURES_PER_FAMILY = 60


def _iso(value: object) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def feature_families() -> dict[str, Any]:
    """Introspect the registered feature factory families (in-memory)."""
    try:
        from quant_platform.research.features import get_global_registry

        registry = get_global_registry()
        families: list[dict[str, Any]] = []
        for name in registry.families():
            manifest = registry.get_family(name)
            specs = list(getattr(manifest, "feature_specs", ()) or ())
            families.append(
                {
                    "name": str(name),
                    "version": getattr(manifest, "version", ""),
                    "feature_count": len(specs),
                    "key_columns": list(getattr(manifest, "key_columns", ()) or ()),
                    "required_inputs": list(getattr(manifest, "required_input_columns", ()) or ()),
                    "features": [
                        {
                            "name": getattr(s, "name", ""),
                            "direction": str(getattr(s, "expected_direction", "unknown")),
                            "lookback_days": getattr(s, "lookback_days", None),
                            "description": getattr(s, "description", "") or "",
                        }
                        for s in specs[:_MAX_FEATURES_PER_FAMILY]
                    ],
                }
            )
    except Exception as exc:  # noqa: BLE001 - never 500 the dashboard
        return {"families": [], "total_families": 0, "total_features": 0, "error": str(exc)}
    families.sort(key=lambda f: f["feature_count"], reverse=True)
    return {
        "families": families,
        "total_families": len(families),
        "total_features": sum(f["feature_count"] for f in families),
    }


def alpha_library(settings: PlatformSettings) -> dict[str, Any]:
    """Alpha blend config + the effective formulaic alpha library."""
    alpha = settings.alpha
    out: dict[str, Any] = {
        "ensemble_mode": getattr(alpha, "ensemble_mode", None),
        "source_weights": {str(k): float(v) for k, v in dict(alpha.source_weights).items()},
        "alphas": [],
        "auto_promoted_count": 0,
    }
    try:
        from quant_platform.research.features.formulaic.features import EFFECTIVE_LIBRARY

        for a in EFFECTIVE_LIBRARY:
            spec = getattr(a, "spec", None) or a
            out["alphas"].append(
                {
                    "name": getattr(a, "name", None) or getattr(spec, "name", None),
                    "description": getattr(a, "description", "")
                    or getattr(spec, "description", "")
                    or "",
                    "expected_direction": str(getattr(spec, "expected_direction", "unknown")),
                    "lookback_days": getattr(spec, "lookback_days", None),
                    "required_inputs": list(getattr(spec, "required_inputs", ()) or ()),
                }
            )
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    with contextlib.suppress(Exception):  # missing/disabled auto-library is fine
        from quant_platform.research.features.formulaic.auto_library import load_promoted_library

        out["auto_promoted_count"] = len(load_promoted_library())
    return out


async def model_registry(settings: PlatformSettings) -> dict[str, Any]:
    """List registered/promoted models (Postgres-backed; empty without a DB)."""
    try:
        from quant_platform.bootstrap.governance.repositories import build_model_registry

        registry = build_model_registry(settings.storage.postgres_dsn or None)
        list_models = getattr(registry, "list_models", None)
        if list_models is None:
            return {
                "models": [],
                "count": 0,
                "error": "model registry requires a Postgres database (QP__STORAGE__POSTGRES_DSN)",
            }
        models = await list_models()
    except Exception as exc:  # noqa: BLE001
        return {"models": [], "count": 0, "error": str(exc)}
    rows = [
        {
            "model_id": str(getattr(m, "model_id", "")),
            "strategy_name": getattr(m, "strategy_name", None),
            "model_version": getattr(m, "model_version", None),
            "feature_set_version": getattr(m, "feature_set_version", None),
            "created_at": _iso(getattr(m, "created_at", None)),
            "active": bool(getattr(m, "active", False)),
            "metadata": dict(getattr(m, "metadata", {}) or {}),
        }
        for m in models
    ]
    return {"models": rows, "count": len(rows)}
