"""Live-LLM startup assertion writing and validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from quant_platform.application.research.text_model_manifest import read_text_model_manifest
from quant_platform.core.domain.production import ProductionProfile
from quant_platform.services.governance_service.llm_live_startup.constants import (
    LLM_LIVE_MAX_INITIAL_CAP,
    LLM_LIVE_STARTUP_ASSERTION_SCHEMA_VERSION,
)
from quant_platform.services.governance_service.llm_live_startup.helpers import (
    _list_payload,
    _parse_timestamp,
    _sha256_file,
    _source_weights_payload,
)
from quant_platform.services.governance_service.llm_live_startup.paths import (
    expected_text_feature_schema_hash,
    llm_live_startup_assertion_path,
    text_model_manifest_path,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from quant_platform.config import PlatformSettings


def write_llm_live_startup_assertion(
    settings: PlatformSettings,
    *,
    candidate_payload: Mapping[str, object],
    as_of: datetime,
) -> Path:
    """Write the startup token consumed by live session creation."""
    manifest_path = text_model_manifest_path(settings)
    if manifest_path is None or not manifest_path.is_file():
        raise RuntimeError("cannot write live LLM startup assertion without text manifest")
    manifest = read_text_model_manifest(manifest_path)
    profile = str(
        candidate_payload.get("profile")
        or (
            ProductionProfile.LLM_LIVE_REHEARSAL.value
            if settings.llm.live_rehearsal_enabled
            else ProductionProfile.LIVE.value
        )
    )
    payload = {
        "schema_version": LLM_LIVE_STARTUP_ASSERTION_SCHEMA_VERSION,
        "generated_at": as_of.astimezone(UTC).isoformat(),
        "profile": profile,
        "passed": bool(candidate_payload.get("passed")),
        "next_allowed_mode": str(candidate_payload.get("next_allowed_mode", "")),
        "text_model_manifest": str(manifest_path),
        "text_model_manifest_sha256": _sha256_file(manifest_path),
        "feature_schema_hash": str(manifest.get("feature_schema_hash", "")),
        "live_rehearsal_enabled": settings.llm.live_rehearsal_enabled,
        "broker_paper_trading": settings.broker.paper_trading,
        "ensemble_mode": settings.alpha.ensemble_mode,
        "source_weights": _source_weights_payload(settings),
        "provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "prompt_version": settings.llm.text_prompt_version,
        "live_cap": str(settings.alpha.max_non_classical_weight),
        "live_ramp_initial": str(settings.alpha.live_ramp_initial),
        "max_request_latency_seconds": settings.llm.max_request_latency_seconds,
        "max_daily_calls": settings.llm.max_daily_calls,
        "max_daily_estimated_cost_usd": settings.llm.max_daily_estimated_cost_usd,
        "estimated_cost_per_call_usd": settings.llm.estimated_cost_per_call_usd,
        "replay_only_live": settings.llm.replay_only_live,
        "checks": _list_payload(candidate_payload.get("checks")),
    }
    path = llm_live_startup_assertion_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def assert_llm_live_startup_allowed(
    settings: PlatformSettings,
    *,
    now: datetime,
) -> None:
    """Fail closed unless a fresh live production-candidate assertion exists."""
    if not settings.llm.live_mode_enabled:
        return
    if settings.alpha.ensemble_mode != "live":
        raise RuntimeError("QP__LLM__LIVE_MODE_ENABLED=true requires QP__ALPHA__ENSEMBLE_MODE=live")
    cap = Decimal(str(settings.alpha.max_non_classical_weight))
    initial = Decimal(str(settings.alpha.live_ramp_initial))
    if cap > LLM_LIVE_MAX_INITIAL_CAP or initial > LLM_LIVE_MAX_INITIAL_CAP:
        raise RuntimeError(
            "QP__LLM__LIVE_MODE_ENABLED=true requires "
            "QP__ALPHA__MAX_NON_CLASSICAL_WEIGHT and "
            f"QP__ALPHA__LIVE_RAMP_INITIAL <= {LLM_LIVE_MAX_INITIAL_CAP}"
        )

    manifest_path = text_model_manifest_path(settings)
    if manifest_path is None or not manifest_path.is_file():
        raise RuntimeError(
            "QP__LLM__LIVE_MODE_ENABLED=true requires "
            "QP__LLM__TEXT_MODEL_MANIFEST to point at a readable manifest"
        )
    assertion_path = llm_live_startup_assertion_path(settings)
    if not assertion_path.is_file():
        raise RuntimeError(
            "QP__LLM__LIVE_MODE_ENABLED=true requires a fresh "
            f"live startup assertion at {assertion_path}"
        )
    try:
        token = json.loads(assertion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid live LLM startup assertion {assertion_path}: {exc}") from exc
    if not isinstance(token, dict):
        raise RuntimeError(f"live LLM startup assertion must be a JSON object: {assertion_path}")

    generated_at = _parse_timestamp(str(token.get("generated_at", "")))
    stale_after = timedelta(hours=settings.llm.live_startup_assertion_stale_after_hours)
    if generated_at < now.astimezone(UTC) - stale_after:
        raise RuntimeError(
            "live LLM startup assertion is stale: "
            f"generated_at={generated_at.isoformat()} stale_after={stale_after}"
        )

    token_profile = str(token.get("profile", ""))
    if (
        token_profile == ProductionProfile.LLM_LIVE_REHEARSAL.value
        and not settings.broker.paper_trading
    ):
        raise RuntimeError(
            "LLM live rehearsal startup assertion cannot be used with "
            "QP__BROKER__PAPER_TRADING=false"
        )
    if settings.llm.live_rehearsal_enabled and not settings.broker.paper_trading:
        raise RuntimeError(
            "QP__LLM__LIVE_REHEARSAL_ENABLED=true requires QP__BROKER__PAPER_TRADING=true"
        )

    expected_profile = (
        ProductionProfile.LLM_LIVE_REHEARSAL.value
        if settings.llm.live_rehearsal_enabled
        else ProductionProfile.LIVE.value
    )
    expected_hash = expected_text_feature_schema_hash(settings)
    required = {
        "schema_version": LLM_LIVE_STARTUP_ASSERTION_SCHEMA_VERSION,
        "profile": expected_profile,
        "passed": True,
        "text_model_manifest": str(manifest_path),
        "text_model_manifest_sha256": _sha256_file(manifest_path),
        "feature_schema_hash": expected_hash,
        "live_rehearsal_enabled": settings.llm.live_rehearsal_enabled,
        "broker_paper_trading": settings.broker.paper_trading,
        "ensemble_mode": settings.alpha.ensemble_mode,
        "source_weights": _source_weights_payload(settings),
        "provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "prompt_version": settings.llm.text_prompt_version,
        "live_cap": str(settings.alpha.max_non_classical_weight),
        "live_ramp_initial": str(settings.alpha.live_ramp_initial),
        "max_request_latency_seconds": settings.llm.max_request_latency_seconds,
        "max_daily_calls": settings.llm.max_daily_calls,
        "max_daily_estimated_cost_usd": settings.llm.max_daily_estimated_cost_usd,
        "estimated_cost_per_call_usd": settings.llm.estimated_cost_per_call_usd,
        "replay_only_live": settings.llm.replay_only_live,
    }
    mismatches = [key for key, expected in required.items() if token.get(key) != expected]
    if mismatches:
        raise RuntimeError(
            "live LLM startup assertion does not match current settings/artifacts: "
            + ", ".join(sorted(mismatches))
        )
    next_allowed = str(token.get("next_allowed_mode", ""))
    if expected_profile == ProductionProfile.LLM_LIVE_REHEARSAL.value:
        if next_allowed != "llm_live_rehearsal":
            raise RuntimeError(
                "live LLM rehearsal startup assertion did not clear rehearsal mode: "
                f"next_allowed_mode={next_allowed!r}"
            )
    elif not next_allowed.startswith("live_ramp"):
        raise RuntimeError(
            "live LLM startup assertion did not clear a live ramp mode: "
            f"next_allowed_mode={next_allowed!r}"
        )
