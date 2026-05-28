"""Output helpers for governed research campaigns."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.config import PlatformSettings
    from quant_platform.services.research_service.sampling.factory_models import (
        WalkForwardEvidence,
    )
    from quant_platform.services.research_service.signal_gates import SignalGateStatus


def maybe_write_text_model_manifest(
    *,
    settings: PlatformSettings,
    args: Any,
    signal_type: str,
    output_root: Path,
    admission: Any,
    evidence: WalkForwardEvidence,
    campaign_manifest: Path,
) -> Path | None:
    if signal_type != "text":
        return None
    from quant_platform.services.research_service.text.model_manifest import (
        write_text_model_manifest,
    )

    return write_text_model_manifest(
        output_root=output_root,
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        feature_names=admission.admitted_features,
        weights=evidence.selected_weights,
        provider=settings.llm.provider,
        llm_model=settings.llm.model,
        prompt_version=str(
            getattr(args, "text_prompt_version", "") or settings.llm.text_prompt_version
        ),
        campaign_manifest=campaign_manifest,
        source_data_manifest=getattr(args, "source_data_manifest", None),
        extraction_manifest=getattr(args, "text_extraction_manifest", None),
        feature_card_dir=getattr(args, "feature_card_dir", None),
        created_at=datetime.now(tz=UTC),
    )


def campaign_run_payload(
    *,
    evidence: WalkForwardEvidence,
    manifest_path: Path,
    sample_build: dict[str, object],
    paper_weights: dict[str, float],
    feature_audits: Any,
    admission: Any,
    model_comparison_path: Path,
    text_manifest_path: Path | None,
    xgboost_manifest_path: Path | None,
    signal_status: SignalGateStatus,
    model_signal_status: SignalGateStatus,
) -> dict[str, object]:
    model_manifest_path = (
        str(text_manifest_path)
        if text_manifest_path is not None
        else str(xgboost_manifest_path)
        if xgboost_manifest_path is not None
        else str(evidence.artifact_root / "model_manifest.json")
        if evidence.artifact_root is not None
        else None
    )
    return {
        "passed": bool(evidence.eligibility["passed"]),
        "run_id": str(evidence.run_id),
        "artifact_root": str(evidence.artifact_root),
        "campaign_manifest": str(manifest_path),
        "sample_build": sample_build,
        "gate_metrics": dict(evidence.metrics),
        "eligibility_checks": evidence.eligibility["checks"],
        "selected_weights": dict(evidence.selected_weights),
        "paper_source_weights": paper_weights,
        "feature_audits": feature_audits,
        "feature_admission": admission.to_payload(),
        "model_comparison": str(model_comparison_path),
        "model_manifest_path": model_manifest_path,
        "signal_gate": {**vars(signal_status), "passed": signal_status.passed},
        "model_signal_gate": {
            **vars(model_signal_status),
            "passed": model_signal_status.passed,
        },
        "next_allowed_paper_mode": (
            "paper_ensemble" if evidence.eligibility["passed"] else "shadow_only"
        ),
    }


__all__ = ["campaign_run_payload", "maybe_write_text_model_manifest"]
