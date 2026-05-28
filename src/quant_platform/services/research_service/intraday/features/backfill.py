"""Immutable feature-vector backfill for promoted intraday alpha candidates."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.core.domain.research import FeatureVector
from quant_platform.services.research_service.campaigns.screening.common import ensure_utc
from quant_platform.services.research_service.intraday.candidates.features import (
    build_intraday_candidate_feature_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from quant_platform.core.contracts import FeatureRepository
    from quant_platform.core.domain.market_data import MarketBar
    from quant_platform.services.research_service.intraday.candidates.features import (
        IntradayCandidateFeatureSpec,
    )
    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@dataclass(frozen=True)
class IntradayFeatureBackfillResult:
    feature_set_version: str
    dry_run: bool
    sample_count: int
    feature_row_count: int
    vector_count: int
    skipped_existing_vectors: int
    skipped_no_intraday_session: int
    skipped_no_selected_features: int
    skipped_availability: int
    feature_names: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        vector_key = "vectors_would_store" if self.dry_run else "vectors_stored"
        return {
            "feature_set_version": self.feature_set_version,
            "dry_run": self.dry_run,
            "sample_count": self.sample_count,
            "feature_row_count": self.feature_row_count,
            vector_key: self.vector_count,
            "skipped_existing_vectors": self.skipped_existing_vectors,
            "skipped_no_intraday_session": self.skipped_no_intraday_session,
            "skipped_no_selected_features": self.skipped_no_selected_features,
            "skipped_availability": self.skipped_availability,
            "feature_names": list(self.feature_names),
        }


async def backfill_intraday_feature_vectors(
    *,
    samples: Sequence[SupervisedAlphaSample],
    intraday_bars: Sequence[MarketBar],
    candidates: Sequence[IntradayCandidateFeatureSpec],
    feature_names: Sequence[str],
    repo: FeatureRepository,
    strategy_run_id: uuid.UUID,
    feature_set_version: str,
    artifact_uri: str,
    dry_run: bool = False,
) -> IntradayFeatureBackfillResult:
    selected_names = tuple(sorted({str(name) for name in feature_names}))
    selected = set(selected_names)
    rows = build_intraday_candidate_feature_rows(
        samples=samples,
        intraday_bars=intraday_bars,
        candidates=candidates,
    )
    vector_count = 0
    skipped_existing = 0
    skipped_no_selected = 0
    skipped_availability = 0
    for row in rows:
        as_of = ensure_utc(row.as_of)
        available_at = ensure_utc(row.available_at)
        if available_at > as_of:
            skipped_availability += 1
            continue
        features = {name: float(value) for name, value in row.features.items() if name in selected}
        if not features:
            skipped_no_selected += 1
            continue
        if await _has_existing_vector(
            repo=repo,
            instrument_id=row.instrument_id,
            feature_set_version=feature_set_version,
            as_of=as_of,
        ):
            skipped_existing += 1
            continue
        vector_count += 1
        if dry_run:
            continue
        await repo.store_vector(
            FeatureVector(
                vector_id=_intraday_vector_id(
                    feature_set_version,
                    row.instrument_id,
                    as_of,
                ),
                instrument_id=row.instrument_id,
                as_of=as_of,
                feature_set_version=feature_set_version,
                features=features,
                strategy_run_id=strategy_run_id,
                artifact_uri=artifact_uri,
                available_at=available_at,
                metadata={"source": "intraday_microstructure"},
            )
        )
    return IntradayFeatureBackfillResult(
        feature_set_version=feature_set_version,
        dry_run=dry_run,
        sample_count=len(samples),
        feature_row_count=len(rows),
        vector_count=vector_count,
        skipped_existing_vectors=skipped_existing,
        skipped_no_intraday_session=len(samples) - len(rows),
        skipped_no_selected_features=skipped_no_selected,
        skipped_availability=skipped_availability,
        feature_names=selected_names,
    )


async def _has_existing_vector(
    *,
    repo: FeatureRepository,
    instrument_id: uuid.UUID,
    feature_set_version: str,
    as_of: datetime,
) -> bool:
    checked_as_of = ensure_utc(as_of)
    rows = await repo.get_vectors([instrument_id], feature_set_version, checked_as_of)
    return any(
        row.instrument_id == instrument_id
        and row.feature_set_version == feature_set_version
        and row.as_of == checked_as_of
        for row in rows
    )


def feature_names_from_family_file(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"feature family file must be a JSON object: {path}")
    families = payload.get("families")
    if not isinstance(families, Mapping):
        raise ValueError(f"feature family file missing families object: {path}")
    names: set[str] = set()
    for raw_members in families.values():
        if isinstance(raw_members, list):
            names.update(str(member) for member in raw_members)
    return tuple(sorted(names))


def _intraday_vector_id(
    feature_set_version: str,
    instrument_id: uuid.UUID,
    as_of: object,
) -> uuid.UUID:
    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"intraday:{feature_set_version}:{instrument_id}:{as_of}",
    )


__all__ = [
    "IntradayFeatureBackfillResult",
    "backfill_intraday_feature_vectors",
    "feature_names_from_family_file",
]
