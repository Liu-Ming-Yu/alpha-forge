"""Artifact IO and validation for intraday backtest evidence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import (
    BacktestEvidenceManifest,
    BacktestReconciliationReport,
    IntradayBacktestSpec,
)
from quant_platform.services.research_service.intraday.evidence.payloads import (
    fill_payload,
    intraday_execution_quality_payload,
    intraday_run_summary_payload,
    manifest_payload,
    reconciliation_payload,
    stable_hash,
    target_weights_payload,
)
from quant_platform.services.research_service.sampling.factory import current_git_commit

if TYPE_CHECKING:
    from collections.abc import Mapping
    from decimal import Decimal

    from quant_platform.services.research_service.intraday.backtesting.types import (
        IntradayBacktestResult,
        IntradayFillArtifact,
    )


def write_reconciliation_report(report: BacktestReconciliationReport, path: Path) -> Path:
    """Persist reconciliation report JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(reconciliation_payload(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def write_backtest_evidence_manifest(
    *,
    spec: IntradayBacktestSpec,
    event_result: IntradayBacktestResult,
    vectorized_result: IntradayBacktestResult,
    reconciliation_report: BacktestReconciliationReport,
    reconciliation_report_path: Path,
    output_path: Path,
    config_payload: Mapping[str, object],
    universe_snapshot_id: uuid.UUID | None = None,
    feature_dataset_id: uuid.UUID | None = None,
    model_artifact_id: uuid.UUID | None = None,
    calibration_artifact_uri: str = "",
) -> Path:
    """Write the immutable evidence manifest required by promotion gates."""
    blockers = list(reconciliation_report.breaches)
    manifest = BacktestEvidenceManifest(
        manifest_id=uuid.uuid4(),
        strategy_run_id=event_result.strategy_run_id,
        created_at=datetime.now(tz=UTC),
        spec=spec,
        code_commit=current_git_commit(),
        config_hash=stable_hash(config_payload),
        dataset_ids=spec.dataset_ids,
        universe_snapshot_id=universe_snapshot_id
        or uuid.uuid5(uuid.NAMESPACE_URL, spec.universe_name),
        feature_dataset_id=feature_dataset_id,
        model_artifact_id=model_artifact_id,
        calibration_artifact_uri=calibration_artifact_uri,
        execution_quality_uri=event_result.execution_quality_uri,
        reconciliation_report_uri=reconciliation_report_path.resolve().as_uri(),
        event_driven_artifact_uri=event_result.run_summary_uri,
        vectorized_artifact_uri=vectorized_result.run_summary_uri,
        passed=reconciliation_report.passed and not blockers,
        blockers=tuple(blockers),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest_payload(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path


def assert_backtest_evidence(path: Path) -> dict[str, object]:
    """Load and fail-closed validate a backtest evidence manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "manifest_id",
        "strategy_run_id",
        "code_commit",
        "config_hash",
        "dataset_ids",
        "execution_quality_uri",
        "reconciliation_report_uri",
        "event_driven_artifact_uri",
        "vectorized_artifact_uri",
        "passed",
    }
    missing = sorted(required - set(payload))
    blockers = list(payload.get("blockers", []))
    if missing:
        blockers.append(f"missing_manifest_keys={','.join(missing)}")
    if not payload.get("passed", False):
        blockers.append("manifest_not_passed")
    rec_path = path_from_uri(str(payload.get("reconciliation_report_uri", "")))
    if rec_path is None or not rec_path.exists():
        blockers.append("reconciliation_report_missing")
    else:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if not rec.get("passed", False):
            blockers.append("reconciliation_report_not_passed")
    result = {"passed": not blockers, "blockers": blockers, "manifest": payload}
    if blockers:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def write_intraday_artifacts(
    root: Path,
    *,
    spec: IntradayBacktestSpec | None,
    strategy_run_id: uuid.UUID,
    nav_curve: list[tuple[datetime, Decimal]],
    fills: list[IntradayFillArtifact],
    target_weights: Mapping[datetime, Mapping[uuid.UUID, Decimal]],
    eligible_universe: Mapping[datetime, tuple[uuid.UUID, ...]],
    final_capital: Decimal,
    total_return: Decimal,
    max_drawdown: Decimal,
    engine_name: str = "intraday",
    engine_version: str = "internal",
    input_hash: str = "",
    cost_assumptions: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    fills_path = root / "intraday_fills.json"
    target_path = root / "target_weights.json"
    eq_path = root / "execution_quality.json"
    summary_path = root / "run_summary.json"
    fills_payload = [fill_payload(fill) for fill in fills]
    fills_path.write_text(json.dumps(fills_payload, indent=2, sort_keys=True), encoding="utf-8")
    target_path.write_text(
        json.dumps(
            target_weights_payload(
                target_weights=target_weights,
                eligible_universe=eligible_universe,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    eq_path.write_text(
        json.dumps(
            intraday_execution_quality_payload(fills),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            intraday_run_summary_payload(
                spec=spec,
                strategy_run_id=strategy_run_id,
                nav_curve=nav_curve,
                final_capital=final_capital,
                total_return=total_return,
                max_drawdown=max_drawdown,
                engine_name=engine_name,
                engine_version=engine_version,
                input_hash=input_hash,
                cost_assumptions=cost_assumptions,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "fills": fills_path,
        "target_weights": target_path,
        "execution_quality": eq_path,
        "run_summary": summary_path,
    }


def path_from_uri(uri: str) -> Path | None:
    if not uri:
        return None
    if uri.startswith("file://"):
        import os
        from urllib.parse import unquote, urlparse

        parsed = urlparse(uri)
        raw = unquote(parsed.path)
        # On Windows, urlparse yields a leading "/" before the drive letter
        # (e.g. "/C:/path"). Strip it so Path() resolves correctly.
        if os.name == "nt" and len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return Path(raw)
    return Path(uri)
