"""Supervised alpha sample generation from feature and bar stores."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from quant_platform.services.research_service.sampling.dates import (
    _ensure_utc,
    daily_as_of_dates,
    research_as_of_dates,
)
from quant_platform.services.research_service.sampling.returns import (
    _bar_history_by_instrument,
    _forward_return,
    _forward_return_from_bars,
)

__all__ = [
    "BuildSamplesResult",
    "SupervisedAlphaSample",
    "_ensure_utc",
    "_forward_return",
    "_forward_return_from_bars",
    "build_supervised_samples",
    "daily_as_of_dates",
    "has_realized_returns",
    "research_as_of_dates",
    "write_samples_json",
]

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from quant_platform.core.contracts import FeatureRepository, HistoricalDataStore
    from quant_platform.core.domain.research import FeatureVector


@dataclass(frozen=True)
class SupervisedAlphaSample:
    """One point-in-time feature row with a forward return label.

    ``metadata`` carries optional context tags (e.g. ``sector``, ``regime``)
    used by the walk-forward attribution layer.  Tags are stored as a tuple
    of ``(key, value)`` pairs so the dataclass stays hashable and
    deterministic across processes.

    ``forward_return`` is the model **label** — a multi-day forward return
    used for IC, feature weighting, and fold-mean-IC. It must not be
    compounded as a daily realized P&L; see ``realized_return_1d`` below.

    The optional fields support governed, label-leakage-safe portfolio
    accounting in callers that opt in:

    * ``realized_return_1d`` — one-day **simple** realized return
      ``close[t+1] / close[t] - 1.0``, used for Sharpe, drawdown, and total
      return. The evaluator compounds these via ``equity *= 1 + r``; a
      log return there would be a subtle variant of the bug ADR-003 fixes.
    * ``as_of_index`` / ``label_end_index`` — integer positions on the
      *global* sorted trading-day calendar (not per-instrument row indices)
      so a sample-level purge can drop any train sample whose label window
      reaches the test window.
    * ``label_end_as_of`` — explicit datetime version of ``label_end_index``
      for callers that prefer datetime-based purge.

    All new fields default to ``None`` so existing callers and existing
    sample JSON artifacts continue to work without change.

    See ``docs/architecture/adr-003-return-accounting-separation.md`` for
    the rationale behind the dual-field design.
    """

    as_of: datetime
    instrument_id: uuid.UUID
    features: dict[str, float]
    forward_return: float
    metadata: tuple[tuple[str, str], ...] = ()
    realized_return_1d: float | None = None
    as_of_index: int | None = None
    label_end_index: int | None = None
    label_end_as_of: datetime | None = None

    def __post_init__(self) -> None:
        # Validate optional fields when set. Defaults of ``None`` opt the
        # sample out of realized-mode evaluation, which is fine for legacy
        # callers — but a *set* value must be well-formed, otherwise a NaN
        # would propagate through ``equity *= 1 + r`` and silently corrupt
        # an entire equity curve. Loud failure beats silent infection.
        if self.realized_return_1d is not None and not math.isfinite(
            float(self.realized_return_1d)
        ):
            raise ValueError(
                f"realized_return_1d must be finite when set; got {self.realized_return_1d!r}"
            )
        if self.as_of_index is not None and int(self.as_of_index) < 0:
            raise ValueError(f"as_of_index must be >= 0 when set; got {self.as_of_index!r}")
        if self.label_end_index is not None and int(self.label_end_index) < 0:
            raise ValueError(f"label_end_index must be >= 0 when set; got {self.label_end_index!r}")
        if (
            self.as_of_index is not None
            and self.label_end_index is not None
            and int(self.label_end_index) < int(self.as_of_index)
        ):
            raise ValueError(
                "label_end_index must be >= as_of_index; got "
                f"as_of_index={self.as_of_index}, "
                f"label_end_index={self.label_end_index}"
            )

    def to_json_row(self) -> dict[str, object]:
        row = asdict(self)
        row["as_of"] = self.as_of.astimezone(UTC).isoformat()
        row["instrument_id"] = str(self.instrument_id)
        if self.metadata:
            row["metadata"] = [list(pair) for pair in self.metadata]
        else:
            row.pop("metadata", None)
        # Drop unset optional fields so JSON stays terse for callers that
        # don't populate them; loaders treat absence as ``None``.
        for optional_key in (
            "realized_return_1d",
            "as_of_index",
            "label_end_index",
            "label_end_as_of",
        ):
            if row.get(optional_key) is None:
                row.pop(optional_key, None)
        if self.label_end_as_of is not None:
            row["label_end_as_of"] = self.label_end_as_of.astimezone(UTC).isoformat()
        return row

    def metadata_dict(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.metadata}


def has_realized_returns(
    scored: Sequence[tuple[SupervisedAlphaSample, float]],
) -> bool:
    """Return ``True`` iff every scored sample carries ``realized_return_1d``.

    Used by the signed-rank ``daily_metrics`` and the long-only
    ``evaluate_long_only_portfolio`` to switch between **realized mode**
    (mark-to-market with one-day simple returns) and **legacy mode** (the
    pre-ADR-003 behavior that compounds the multi-day forward-return
    label as if it were a daily realized P&L).

    Co-located with ``SupervisedAlphaSample`` so the realized-mode
    contract lives next to the field that defines it.
    """
    return all(row.realized_return_1d is not None for row, _ in scored)


@dataclass(frozen=True)
class BuildSamplesResult:
    """Summary of a supervised sample build."""

    samples: tuple[SupervisedAlphaSample, ...]
    requested_points: int
    skipped_missing_features: int
    skipped_stale_features: int
    skipped_missing_bars: int
    skipped_invalid_features: int
    as_of_dates_requested: int = 0
    date_policy: str = "custom"


async def build_supervised_samples(
    *,
    feature_repo: FeatureRepository,
    bar_store: HistoricalDataStore,
    instrument_ids: Iterable[uuid.UUID],
    feature_set_version: str,
    as_of_dates: Iterable[datetime],
    horizon_days: int,
    bar_seconds: int = 86400,
    max_feature_age_days: int = 3,
    date_policy: str = "custom",
) -> BuildSamplesResult:
    """Build leakage-aware supervised samples for calibration and boosting.

    For each ``as_of`` timestamp, the latest feature vector at or before
    ``as_of`` is joined to an entry close at or before ``as_of`` and an exit
    close at or after ``as_of + horizon_days``.  Rows are skipped when features
    are stale, bars are unavailable, or required numeric values are non-finite.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be > 0")
    if max_feature_age_days < 0:
        raise ValueError("max_feature_age_days must be >= 0")

    ids = tuple(instrument_ids)
    samples: list[SupervisedAlphaSample] = []
    requested = 0
    skipped_missing_features = 0
    skipped_stale_features = 0
    skipped_missing_bars = 0
    skipped_invalid_features = 0
    requested_dates = tuple(_ensure_utc(as_of) for as_of in as_of_dates)
    feature_history = await _feature_history_by_instrument(
        feature_repo=feature_repo,
        instrument_ids=ids,
        feature_set_version=feature_set_version,
        as_of_dates=requested_dates,
        max_feature_age_days=max_feature_age_days,
    )
    bar_history = await _bar_history_by_instrument(
        bar_store=bar_store,
        instrument_ids=ids,
        as_of_dates=requested_dates,
        horizon_days=horizon_days,
        bar_seconds=bar_seconds,
    )

    for as_of in requested_dates:
        by_instrument = (
            _latest_vectors_from_history(feature_history, as_of=as_of)
            if feature_history is not None
            else {
                vector.instrument_id: vector
                for vector in await feature_repo.get_vectors(list(ids), feature_set_version, as_of)
            }
        )
        for instrument_id in ids:
            requested += 1
            vector = by_instrument.get(instrument_id)
            if vector is None:
                skipped_missing_features += 1
                continue
            if vector.as_of < as_of - timedelta(days=max_feature_age_days):
                skipped_stale_features += 1
                continue
            features = _clean_features(vector.features)
            if features is None:
                skipped_invalid_features += 1
                continue
            forward_return = _forward_return_from_bars(
                bars=bar_history.get(instrument_id, ()),
                as_of=as_of,
                horizon_days=horizon_days,
            )
            if forward_return is None:
                skipped_missing_bars += 1
                continue
            samples.append(
                SupervisedAlphaSample(
                    as_of=as_of,
                    instrument_id=instrument_id,
                    features=features,
                    forward_return=forward_return,
                )
            )

    return BuildSamplesResult(
        samples=tuple(samples),
        requested_points=requested,
        skipped_missing_features=skipped_missing_features,
        skipped_stale_features=skipped_stale_features,
        skipped_missing_bars=skipped_missing_bars,
        skipped_invalid_features=skipped_invalid_features,
        as_of_dates_requested=len(requested_dates),
        date_policy=date_policy,
    )


async def _feature_history_by_instrument(
    *,
    feature_repo: FeatureRepository,
    instrument_ids: Sequence[uuid.UUID],
    feature_set_version: str,
    as_of_dates: Sequence[datetime],
    max_feature_age_days: int,
) -> dict[uuid.UUID, list[FeatureVector]] | None:
    if not as_of_dates:
        return {}
    history_loader = getattr(feature_repo, "get_vector_history", None)
    if not callable(history_loader):
        return None
    start = min(as_of_dates) - timedelta(days=max_feature_age_days)
    end = max(as_of_dates)
    rows = await history_loader(list(instrument_ids), feature_set_version, start, end)
    by_instrument: dict[uuid.UUID, list[FeatureVector]] = {
        instrument_id: [] for instrument_id in instrument_ids
    }
    for vector in sorted(rows, key=lambda row: (_ensure_utc(row.as_of), row.instrument_id)):
        by_instrument.setdefault(vector.instrument_id, []).append(vector)
    return by_instrument


def _latest_vectors_from_history(
    history: Mapping[uuid.UUID, Sequence[FeatureVector]],
    *,
    as_of: datetime,
) -> dict[uuid.UUID, FeatureVector]:
    latest: dict[uuid.UUID, FeatureVector] = {}
    for instrument_id, rows in history.items():
        for vector in rows:
            available_at = _ensure_utc(vector.available_at or vector.as_of)
            vector_as_of = _ensure_utc(vector.as_of)
            if vector_as_of <= as_of and available_at <= as_of:
                latest[instrument_id] = vector
            if vector_as_of > as_of:
                break
    return latest


def write_samples_json(samples: Iterable[SupervisedAlphaSample], path: Path) -> Path:
    """Write samples in the shared calibration/boosting JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.to_json_row() for sample in samples]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _clean_features(raw: Mapping[str, float]) -> dict[str, float] | None:
    features: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(numeric):
            return None
        features[str(key)] = numeric
    return features
