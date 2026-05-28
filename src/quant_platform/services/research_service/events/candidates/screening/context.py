"""Point-in-time SEC-event context feature construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


def _events_by_instrument(
    source_manifest: Mapping[str, object],
) -> dict[UUID, tuple[datetime, ...]]:
    events: dict[UUID, list[datetime]] = {}
    raw_events = source_manifest.get("events", ())
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, str):
        return {}
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            continue
        raw_instrument = raw.get("instrument_id")
        raw_occurred = raw.get("occurred_at")
        if raw_instrument is None or raw_occurred is None:
            continue
        try:
            instrument_id = UUID(str(raw_instrument))
            occurred_at = datetime.fromisoformat(str(raw_occurred))
        except (TypeError, ValueError):
            continue
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        events.setdefault(instrument_id, []).append(occurred_at.astimezone(UTC))
    return {instrument_id: tuple(sorted(rows)) for instrument_id, rows in events.items()}


def events_by_instrument_from_manifest(
    source_manifest: Mapping[str, object],
) -> dict[UUID, tuple[datetime, ...]]:
    """Return point-in-time SEC event timestamps keyed by instrument id."""
    return _events_by_instrument(source_manifest)


def _event_context_features(
    sample: SupervisedAlphaSample,
    events_by_instrument: Mapping[UUID, Sequence[datetime]],
) -> dict[str, float]:
    return event_context_features(
        instrument_id=sample.instrument_id,
        as_of=sample.as_of,
        events_by_instrument=events_by_instrument,
    )


def event_context_features(
    *,
    instrument_id: UUID,
    as_of: datetime,
    events_by_instrument: Mapping[UUID, Sequence[datetime]],
) -> dict[str, float]:
    """Build materializable SEC event-count features for one instrument/date."""
    as_of = as_of.astimezone(UTC)
    event_ages = [
        (as_of - occurred_at).total_seconds() / 86400.0
        for occurred_at in events_by_instrument.get(instrument_id, ())
        if occurred_at <= as_of
    ]
    features: dict[str, float] = {}
    for lookback in (1, 2, 3, 4, 5, 6, 7, 8, 9, 21):
        active = [age for age in event_ages if 0.0 <= age <= float(lookback)]
        decays = [max(0.0, 1.0 - age / float(lookback)) for age in active]
        features[f"sec_event_count_le_{lookback}d_scaled"] = min(float(len(active)), 3.0) / 3.0
        features[f"sec_event_density_{lookback}d"] = sum(decays) / float(lookback)
        features[f"sec_event_recency_{lookback}d"] = max(decays) if decays else 0.0
    features["sec_event_count_21d"] = features["sec_event_count_le_21d_scaled"] * 3.0
    features["sec_event_density_21d"] = features["sec_event_density_21d"]
    features["days_since_sec_event"] = min(event_ages) if event_ages else 999.0
    return features
