"""Durable artifact IO for research campaigns."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.payloads import (
    campaign_manifest_payload,
    walk_forward_artifact_payloads,
)
from quant_platform.services.research_service.reports.tearsheet import render_tearsheet
from quant_platform.services.research_service.sampling.factory_models import (
    WalkForwardEvidence,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from quant_platform.core.contracts import ArtifactStore


def write_walk_forward_artifacts(
    evidence: WalkForwardEvidence,
    *,
    output_root: Path,
    artifact_store: ArtifactStore | None = None,
) -> WalkForwardEvidence:
    """Write standard research artifacts and render a tearsheet."""
    run_dir = output_root / str(evidence.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    for filename, payload in walk_forward_artifact_payloads(evidence).items():
        _write_json(run_dir / filename, payload, artifact_store=artifact_store)
    render_tearsheet(evidence.run_id, output_root)
    return WalkForwardEvidence(
        run_id=evidence.run_id,
        model_version=evidence.model_version,
        feature_set_version=evidence.feature_set_version,
        folds=evidence.folds,
        selected_weights=evidence.selected_weights,
        daily_returns=evidence.daily_returns,
        daily_ics=evidence.daily_ics,
        metrics=evidence.metrics,
        eligibility=evidence.eligibility,
        artifact_root=run_dir,
        daily_turnover=evidence.daily_turnover,
        feature_stability=evidence.feature_stability,
        bootstrap_ic_ci=evidence.bootstrap_ic_ci,
        attribution=evidence.attribution,
        slippage_bps_per_turnover=evidence.slippage_bps_per_turnover,
        portfolio_config=evidence.portfolio_config,
        portfolio_diagnostics=evidence.portfolio_diagnostics,
        drawdown_diagnostics=evidence.drawdown_diagnostics,
    )


def write_campaign_manifest(
    evidence: WalkForwardEvidence,
    *,
    samples_path: Path,
    paper_source_weights: Mapping[str, float],
    git_commit: str | None = None,
    xgboost_manifest_path: Path | None = None,
    model_comparison_path: Path | None = None,
    feature_audits: Sequence[Mapping[str, object]] = (),
    campaign_context: Mapping[str, object] | None = None,
    command: str = "",
    artifact_store: ArtifactStore | None = None,
) -> Path:
    """Write ``campaign_manifest.json`` beside walk-forward artifacts."""
    if evidence.artifact_root is None:
        raise ValueError("walk-forward artifacts must be written before campaign manifest")
    run_dir = evidence.artifact_root
    payload = campaign_manifest_payload(
        evidence,
        samples_path=samples_path,
        paper_source_weights=paper_source_weights,
        git_commit=git_commit if git_commit is not None else current_git_commit(),
        xgboost_manifest_path=xgboost_manifest_path,
        model_comparison_path=model_comparison_path,
        feature_audits=feature_audits,
        campaign_context=campaign_context,
        command=command,
    )
    path = run_dir / "campaign_manifest.json"
    _write_json(path, payload, artifact_store=artifact_store)
    return path


def write_model_comparison(
    evidence: WalkForwardEvidence,
    *,
    rows: Sequence[Mapping[str, object]],
    campaign_context: Mapping[str, object] | None = None,
    artifact_store: ArtifactStore | None = None,
) -> Path:
    """Write campaign model-comparison evidence beside walk-forward artifacts."""
    if evidence.artifact_root is None:
        raise ValueError("walk-forward artifacts must be written before model comparison")
    path = evidence.artifact_root / "model_comparison.json"
    payload: dict[str, object] = {
        "run_id": str(evidence.run_id),
        "model_version": evidence.model_version,
        "feature_set_version": evidence.feature_set_version,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "selected_candidate": _selected_candidate(rows),
        "candidates": [dict(row) for row in rows],
    }
    if campaign_context:
        payload.update(dict(campaign_context))
    _write_json(path, payload, artifact_store=artifact_store)
    return path


def read_campaign_manifest(path: Path) -> dict[str, object]:
    """Read one campaign manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"campaign manifest must be a JSON object: {path}")
    return dict(payload)


def list_campaign_manifests(root: Path, *, limit: int = 20) -> list[dict[str, object]]:
    """Return recent campaign manifests from the standard walk-forward root."""
    if not root.exists():
        return []
    manifests: list[dict[str, object]] = []
    for path in root.glob("*/campaign_manifest.json"):
        try:
            payload = read_campaign_manifest(path)
        except (OSError, json.JSONDecodeError):
            continue
        payload.setdefault("artifact_root", str(path.parent))
        manifests.append(payload)
    manifests.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return manifests[: max(1, limit)]


def current_git_commit() -> str:
    """Best-effort source revision for reproducible campaign manifests."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        proc = subprocess.run(
            [git, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else "unknown"


def _write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    artifact_store: ArtifactStore | None = None,
) -> None:
    if artifact_store is not None:
        artifact_store.write_json(str(path.resolve()), payload)
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _selected_candidate(rows: Sequence[Mapping[str, object]]) -> str | None:
    for row in rows:
        if bool(row.get("selected")):
            return str(row.get("candidate", ""))
    return None
