"""Shared helpers for research candidate-screen CLI workflows."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, cast

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.common import _json_default, research_json_result
from quant_platform.services.research_service.campaigns.screening.common import ensure_utc

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from pathlib import Path

    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings
    from quant_platform.core.domain.research import FeatureVector
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def write_screen_payload(
    *,
    screen: dict[str, object],
    output_root: Path,
    slug: str,
    report: str,
) -> UseCaseResult[dict[str, object]]:
    """Write candidate-screen diagnostics and return the CLI result payload."""
    diagnostics_root = output_root / "diagnostics" / slug
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    screen_path = diagnostics_root / "candidate_screen.json"
    screen_path.write_text(
        json.dumps(screen, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = diagnostics_root / "operator_report.md"
    report_path.write_text(report, encoding="utf-8")
    payload = {
        "passed": bool(screen["passed"]),
        "reason": screen["reason"],
        "candidate_screen": str(screen_path),
        "operator_report": str(report_path),
        "diagnostic_only": True,
        "promotion_artifacts_written": bool(screen.get("promotion_artifacts_written", False)),
        "passing_candidates": screen["passing_candidates"],
        "quarantined_candidates": screen["quarantined_candidates"],
        "screened_candidate_count": screen["screened_candidate_count"],
        "blockers": screen["blockers"],
        "feature_family_artifacts": screen.get("feature_family_artifacts", {"written": False}),
    }
    passed = bool(screen["passed"])
    if not passed:
        blocked_path = diagnostics_root / "blocked_candidate_screen_summary.json"
        blocked_path.write_text(
            json.dumps(screen, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        payload["blocked_candidate_screen_summary"] = str(blocked_path)
    return research_json_result(payload, passed=passed)


def load_json_mapping(path: Path) -> dict[str, object]:
    """Load a JSON object from disk for candidate-screen inputs."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorUsageError(f"failed to load JSON mapping {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OperatorUsageError(f"JSON mapping must be an object: {path}")
    return {str(key): value for key, value in payload.items()}


def load_sample_build_summary(path: Path) -> dict[str, object]:
    """Load the sample-build metadata object used by screen reports."""
    payload = load_json_mapping(path)
    raw = payload.get("sample_build", payload)
    if not isinstance(raw, dict):
        raise OperatorUsageError(f"sample build summary must be a JSON object: {path}")
    return {str(key): value for key, value in raw.items()}


def filter_samples_by_window(
    samples: tuple[SupervisedAlphaSample, ...],
    *,
    sample_start: datetime | None,
    sample_end: datetime | None,
) -> tuple[SupervisedAlphaSample, ...]:
    """Filter supervised samples by optional inclusive UTC timestamps."""
    start = ensure_utc(sample_start) if sample_start is not None else None
    end = ensure_utc(sample_end) if sample_end is not None else None
    if start is not None and end is not None and end < start:
        raise OperatorUsageError("--sample-end must be >= --sample-start")
    return tuple(
        sample
        for sample in samples
        if (start is None or ensure_utc(sample.as_of) >= start)
        and (end is None or ensure_utc(sample.as_of) <= end)
    )


def sample_filter_payload(
    all_samples: tuple[SupervisedAlphaSample, ...],
    samples: tuple[SupervisedAlphaSample, ...],
    args: Any,
) -> dict[str, object]:
    """Return machine-readable sample filtering details."""
    start = getattr(args, "sample_start", None)
    end = getattr(args, "sample_end", None)
    return {
        "sample_start": ensure_utc(start).isoformat() if start is not None else None,
        "sample_end": ensure_utc(end).isoformat() if end is not None else None,
        "loaded_sample_count": len(all_samples),
        "screened_sample_count": len(samples),
    }


async def load_text_feature_vectors(
    settings: PlatformSettings,
    *,
    feature_set_version: str,
    end: datetime,
) -> tuple[FeatureVector, ...]:
    """Load text feature vectors from the durable feature-vector table."""
    from sqlalchemy import text

    from quant_platform.infrastructure.postgres.support import create_pg_engine

    engine = create_pg_engine(settings.storage.postgres_dsn)
    try:
        async with engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text("""
                        SELECT vector_id, instrument_id, as_of, feature_set_version,
                               features, strategy_run_id, artifact_uri, available_at
                        FROM feature_vectors
                        WHERE feature_set_version = :feature_set_version
                          AND as_of <= :end
                        ORDER BY instrument_id, as_of
                    """),
                        {"feature_set_version": feature_set_version, "end": end},
                    )
                )
                .mappings()
                .all()
            )
    finally:
        await engine.dispose()
    return tuple(_feature_vector_from_row(row) for row in rows)


def screen_inputs(args: Any) -> dict[str, object]:
    """Return common candidate-screen input paths."""
    return {
        "samples_file": str(args.samples_file),
        "sample_build_summary": str(args.sample_build_summary),
        "source_data_manifest": str(args.source_data_manifest),
    }


def screen_slug(
    args: Any,
    *,
    samples: tuple[SupervisedAlphaSample, ...],
    feature_attr: str,
) -> str:
    """Return the explicit screen name or deterministic candidate-screen slug."""
    raw = str(getattr(args, "screen_name", "") or "").strip()
    if raw:
        return raw
    feature_set = str(getattr(args, feature_attr))
    if samples:
        start = min(sample.as_of for sample in samples)
        end = max(sample.as_of for sample in samples)
        return (
            f"{feature_set}_{args.candidate_family}_{args.candidate_set}_"
            f"{start:%Y-%m-%d}_{end:%Y-%m-%d}_candidate_screen"
        )
    return f"{feature_set}_{args.candidate_family}_{args.candidate_set}_candidate_screen"


def _feature_vector_from_row(row: object) -> FeatureVector:
    from quant_platform.core.domain.research import FeatureVector

    mapping = cast("Mapping[str, object]", row)
    return FeatureVector(
        vector_id=uuid.UUID(str(mapping["vector_id"])),
        instrument_id=uuid.UUID(str(mapping["instrument_id"])),
        as_of=cast("datetime", mapping["as_of"]),
        feature_set_version=str(mapping["feature_set_version"]),
        features=_coerce_feature_map(mapping["features"]),
        strategy_run_id=uuid.UUID(str(mapping["strategy_run_id"])),
        artifact_uri=str(mapping.get("artifact_uri", "") or ""),
        available_at=cast("datetime", mapping["available_at"]),
    )


def _coerce_feature_map(raw: object) -> dict[str, float]:
    loaded = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(loaded, dict):
        return {}
    features: dict[str, float] = {}
    for key, value in loaded.items():
        try:
            features[str(key)] = float(value)
        except (TypeError, ValueError, OverflowError):
            features[str(key)] = 0.0
    return features


__all__ = [
    "filter_samples_by_window",
    "load_json_mapping",
    "load_sample_build_summary",
    "load_text_feature_vectors",
    "sample_filter_payload",
    "screen_inputs",
    "screen_slug",
    "write_screen_payload",
]
