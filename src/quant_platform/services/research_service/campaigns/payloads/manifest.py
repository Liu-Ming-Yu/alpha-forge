"""Campaign manifest payload builder."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.payloads.evidence import (
    campaign_evidence_payload,
    replay_context_payload,
    require_non_null_campaign_evidence,
)
from quant_platform.services.research_service.sampling.factory_models import (
    ResearchCampaignManifest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from quant_platform.services.research_service.sampling.factory_models import (
        WalkForwardEvidence,
    )


def campaign_manifest_payload(
    evidence: WalkForwardEvidence,
    *,
    samples_path: Path,
    paper_source_weights: Mapping[str, float],
    git_commit: str,
    xgboost_manifest_path: Path | None,
    model_comparison_path: Path | None = None,
    feature_audits: Sequence[Mapping[str, object]] = (),
    campaign_context: Mapping[str, object] | None = None,
    command: str = "",
) -> dict[str, object]:
    """Build a JSON-safe campaign manifest payload."""
    if evidence.artifact_root is None:
        raise ValueError("walk-forward artifacts must be written before campaign manifest")
    run_dir = evidence.artifact_root
    passed = bool(evidence.eligibility["passed"])
    artifacts: dict[str, str | None] = {
        "samples": str(samples_path),
        "fold_metrics": str(run_dir / "fold_metrics.json"),
        "eligibility": str(run_dir / "eligibility.json"),
        "ic_report": str(run_dir / "ic_report.json"),
        "tearsheet": str(run_dir / "tearsheet.md"),
        "execution_quality": str(run_dir / "execution_quality.json"),
        "model_manifest": str(run_dir / "model_manifest.json"),
        "run_summary": str(run_dir / "run_summary.json"),
        "attribution": str(run_dir / "attribution.json"),
        "feature_stability": str(run_dir / "feature_stability.json"),
        "portfolio_config": str(run_dir / "portfolio_config.json")
        if evidence.portfolio_config
        else None,
        "portfolio_diagnostics": str(run_dir / "portfolio_diagnostics.json")
        if evidence.portfolio_diagnostics
        else None,
        "drawdown_diagnostics": str(run_dir / "drawdown_diagnostics.json")
        if evidence.drawdown_diagnostics
        else None,
        "backtest_evidence": str(run_dir / "backtest_evidence_manifest.json")
        if (run_dir / "backtest_evidence_manifest.json").exists()
        else None,
        "xgboost_manifest": str(xgboost_manifest_path) if xgboost_manifest_path else None,
        "model_comparison": str(model_comparison_path)
        if model_comparison_path
        else str(run_dir / "model_comparison.json")
        if (run_dir / "model_comparison.json").exists()
        else None,
        "feature_audits": None,
    }
    manifest = ResearchCampaignManifest(
        run_id=evidence.run_id,
        created_at=datetime.now(tz=UTC),
        model_version=evidence.model_version,
        feature_set_version=evidence.feature_set_version,
        passed=passed,
        metrics=dict(evidence.metrics),
        eligibility=dict(evidence.eligibility),
        artifacts=artifacts,
        selected_weights=dict(evidence.selected_weights),
        paper_source_weights=dict(paper_source_weights),
        git_commit=git_commit,
        next_allowed_paper_mode="paper_ensemble" if passed else "shadow_only",
    )
    payload = {
        "run_id": str(manifest.run_id),
        "created_at": manifest.created_at.isoformat(),
        "model_version": manifest.model_version,
        "feature_set_version": manifest.feature_set_version,
        "passed": manifest.passed,
        "metrics": dict(manifest.metrics),
        "eligibility": dict(manifest.eligibility),
        "artifacts": dict(manifest.artifacts),
        "selected_weights": dict(manifest.selected_weights),
        "paper_source_weights": dict(manifest.paper_source_weights),
        "git_commit": manifest.git_commit,
        "next_allowed_paper_mode": manifest.next_allowed_paper_mode,
        "feature_audits": list(feature_audits),
        "command": command,
    }
    context = campaign_context or {}
    if context:
        payload.update(dict(context))
    payload["replay_context"] = replay_context_payload(
        evidence,
        samples_path=samples_path,
        campaign_context=context,
    )
    payload["campaign_evidence"] = campaign_evidence_payload(
        evidence,
        passed=passed,
        campaign_context=context,
    )
    if context:
        require_non_null_campaign_evidence(payload)
    return payload


__all__ = ["campaign_manifest_payload"]
