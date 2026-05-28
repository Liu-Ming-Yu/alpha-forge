"""In-memory FeatureRepository implementation.

Satisfies the ``FeatureRepository`` protocol from ``core.contracts`` for
testing, paper trading, and early-stage production before a Postgres- or
Parquet-backed store is needed.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import structlog

from quant_platform.core.exceptions import FeatureValidationError

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.research import FeatureVector

log = structlog.get_logger(__name__)


class InMemoryFeatureRepository:
    """Write-once, query-by-as_of feature vector store.

    Args:
        max_vectors: Maximum number of feature vectors to retain in memory.
            When the cap is reached, the oldest 20% of vectors (by as_of) are
            pruned before each new insertion.  Default: 100_000.
    """

    def __init__(self, max_vectors: int = 100_000) -> None:
        self._vectors: dict[tuple[uuid.UUID, str, datetime], FeatureVector] = {}
        self._max_vectors = max_vectors
        self._lock = asyncio.Lock()

    async def store_vector(self, vector: FeatureVector) -> None:
        for name, value in vector.features.items():
            if not math.isfinite(value):
                raise FeatureValidationError(
                    f"Non-finite value {value!r} for feature {name!r} on "
                    f"instrument {vector.instrument_id}"
                )
        async with self._lock:
            if len(self._vectors) >= self._max_vectors:
                self._evict()
            key = (vector.instrument_id, vector.feature_set_version, vector.as_of)
            if key in self._vectors:
                raise ValueError(
                    f"Duplicate FeatureVector: instrument={vector.instrument_id}, "
                    f"version={vector.feature_set_version}, as_of={vector.as_of}"
                )
            self._vectors[key] = vector

    def _evict(self) -> None:
        evict_count = max(1, self._max_vectors // 5)
        sorted_keys = sorted(self._vectors, key=lambda k: self._vectors[k].as_of)
        for key in sorted_keys[:evict_count]:
            del self._vectors[key]
        log.debug("feature_repository.evicted", count=evict_count)

    async def get_vectors(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        as_of: datetime,
    ) -> list[FeatureVector]:
        async with self._lock:
            result: list[FeatureVector] = []
            ids = set(instrument_ids)
            for key, vec in self._vectors.items():
                iid, ver, ts = key
                available_at = vec.available_at or vec.as_of
                if (
                    iid in ids
                    and ver == feature_set_version
                    and ts <= as_of
                    and available_at <= as_of
                ):
                    result.append(vec)
            best: dict[uuid.UUID, FeatureVector] = {}
            for vec in result:
                prev = best.get(vec.instrument_id)
                if prev is None or (vec.as_of, vec.available_at or vec.as_of) > (
                    prev.as_of,
                    prev.available_at or prev.as_of,
                ):
                    best[vec.instrument_id] = vec
            return list(best.values())

    async def get_vector_history(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        start: datetime,
        end: datetime,
    ) -> list[FeatureVector]:
        """Return all visible vectors in a closed point-in-time window."""
        async with self._lock:
            ids = set(instrument_ids)
            rows: list[FeatureVector] = []
            for (instrument_id, version, as_of), vector in sorted(
                self._vectors.items(),
                key=lambda item: item[1].as_of,
            ):
                if (
                    instrument_id in ids
                    and version == feature_set_version
                    and start <= as_of <= end
                    and (vector.available_at or vector.as_of) <= end
                ):
                    rows.append(vector)
            return rows

    async def prune(self, older_than: datetime) -> int:
        """Drop in-memory vectors with ``as_of < older_than``.

        Mirrors :meth:`PostgresFeatureRepository.prune` so the CLI has
        the same surface regardless of backend.  Returns the number of
        keys removed.
        """
        async with self._lock:
            stale = [key for key, vec in self._vectors.items() if vec.as_of < older_than]
            for key in stale:
                del self._vectors[key]
            return len(stale)
