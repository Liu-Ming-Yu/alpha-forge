"""Pure shared helpers for research operations.

No composition or infrastructure dependencies: result wrapping, instrument-
contract loading, calibration-artifact discovery, durable-input guards, and
sample-result payload shaping. The composition-bound helpers (session and
sample building, Postgres schema verification, intraday feature loading) live
in ``research/common.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from quant_platform.application.errors import OperatorUsageError
from quant_platform.application.operator.cli_inputs import (
    instrument_lookup_from_contracts,
    load_instrument_contracts,
    parse_intraday_decision_times,
)
from quant_platform.application.operator.serialization import _json_default as _json_default
from quant_platform.application.results import ResultPresentation, UseCaseResult, UseCaseStatus

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from quant_platform.config import PlatformSettings

_BACKTEST_WARMUP_DAYS = 270


class SupervisedSamplesResultLike(Protocol):
    """Structural view of a supervised-samples build result.

    Declared here so this pure helper can shape sample payloads without
    importing the concrete ``services`` result type.
    """

    @property
    def samples(self) -> Sequence[object]: ...

    @property
    def requested_points(self) -> int: ...

    @property
    def skipped_missing_features(self) -> int: ...

    @property
    def skipped_stale_features(self) -> int: ...

    @property
    def skipped_missing_bars(self) -> int: ...

    @property
    def skipped_invalid_features(self) -> int: ...


def research_json_result(
    payload: Mapping[str, object],
    *,
    passed: bool = True,
    exit_code: int = 2,
) -> UseCaseResult[dict[str, object]]:
    """Wrap a research payload dict as a JSON-rendered ``UseCaseResult``.

    When ``passed`` is False the result is BLOCKED with ``exit_code`` so the CLI
    presentation layer surfaces a non-zero exit.
    """
    return UseCaseResult(
        status=UseCaseStatus.OK if passed else UseCaseStatus.BLOCKED,
        payload=dict(payload),
        exit_code=0 if passed else exit_code,
        presentation=ResultPresentation.JSON,
    )


def _load_instrument_contracts(path: str) -> dict[uuid.UUID, dict[str, object]]:
    return load_instrument_contracts(path)


def _instrument_lookup_from_contracts(
    contracts: Mapping[uuid.UUID, dict[str, object]],
) -> dict[str, uuid.UUID]:
    return instrument_lookup_from_contracts(contracts)


def _parse_intraday_decision_times(
    raw_values: list[str],
    start: datetime,
    end: datetime,
) -> tuple[datetime, ...]:
    """Parse ISO datetimes or HH:MM daily decision times into UTC datetimes."""
    return parse_intraday_decision_times(raw_values, start, end)


def _latest_calibration_artifact(settings: PlatformSettings) -> Path | None:
    root = Path(settings.storage.object_store_root) / "calibration"
    if not root.is_dir():
        return None
    candidates = sorted(
        (p for p in root.glob("simulator_calibration_*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_calibration_recommendation_bps(
    path: Path | None,
    *,
    max_age_days: float | None = None,
    as_of: datetime | None = None,
) -> tuple[float | None, dict[str, object]]:
    if path is None or not path.is_file():
        return None, {"path": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {"path": str(path), "error": f"unreadable: {exc}"}
    overall = payload.get("overall") or {}
    bps_raw = overall.get("recommended_bps")
    if not isinstance(bps_raw, (int, float)) or bps_raw <= 0:
        return None, {"path": str(path), "error": "no overall.recommended_bps"}
    metadata: dict[str, object] = {
        "path": str(path),
        "sample_count": payload.get("sample_count"),
        "insufficient_data": bool(payload.get("insufficient_data", False)),
        "generated_at": payload.get("generated_at"),
    }
    if metadata["insufficient_data"]:
        metadata["error"] = "insufficient_data"
        return None, metadata
    if max_age_days is not None and as_of is not None:
        generated_at_raw = payload.get("generated_at")
        if isinstance(generated_at_raw, str):
            try:
                generated_at = datetime.fromisoformat(generated_at_raw)
            except ValueError:
                metadata["error"] = "invalid generated_at"
                return None, metadata
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=UTC)
            age_days = (
                as_of.astimezone(UTC) - generated_at.astimezone(UTC)
            ).total_seconds() / 86400.0
            metadata["age_days"] = age_days
            if age_days > max_age_days:
                metadata["error"] = "stale"
                return None, metadata
    return float(bps_raw), metadata


def _require_durable_research_inputs(settings: PlatformSettings) -> None:
    if not settings.storage.postgres_dsn:
        raise OperatorUsageError(
            "features build-samples requires durable historical feature vectors via "
            "QP__STORAGE__POSTGRES_DSN when no feature input file is supplied; "
            "an in-memory session cannot contain historical research vectors."
        )


def _samples_result_payload(
    path: Path,
    result: SupervisedSamplesResultLike,
) -> dict[str, object]:
    return {
        "output": str(path),
        "samples": len(result.samples),
        "date_policy": str(getattr(result, "date_policy", "unknown")),
        "as_of_dates_requested": int(getattr(result, "as_of_dates_requested", 0)),
        "requested_points": result.requested_points,
        "skipped_missing_features": result.skipped_missing_features,
        "skipped_stale_features": result.skipped_stale_features,
        "skipped_missing_bars": result.skipped_missing_bars,
        "skipped_invalid_features": result.skipped_invalid_features,
    }
