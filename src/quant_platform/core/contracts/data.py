"""Data-layer contracts: market data, historical bars, feature persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncIterator
    from datetime import date, datetime

    from quant_platform.core.domain.instruments import (
        CorporateAction,
        SecurityMasterRecord,
        SymbolHistory,
        UniverseSnapshot,
    )
    from quant_platform.core.domain.market_data import (
        BarDataset,
        DatasetQuorumEvidence,
        MarketBar,
        VendorBarBatch,
    )
    from quant_platform.core.domain.research import FeatureDataset, FeatureVector


@runtime_checkable
class MarketDataProvider(Protocol):
    """Real-time and delayed market data subscription interface.

    Must never:
        Cache data beyond the declared TTL; return stale bars as current.
    """

    def subscribe_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> AsyncIterator[MarketBar]:
        """Yield new bars as they arrive for the given instrument and resolution."""
        ...

    async def get_last_bar(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
    ) -> MarketBar | None:
        """Return the most recently completed bar, or None if unavailable."""
        ...


@runtime_checkable
class HistoricalDataStore(Protocol):
    """Read/write interface for the Parquet-backed bar store.

    Must never:
        Return unadjusted bars to downstream consumers.  Adjustment is
        applied at read time using the corporate action log.
    """

    async def get_bars(
        self,
        instrument_id: uuid.UUID,
        bar_seconds: int,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        """Return all adjusted bars for the given instrument and window."""
        ...

    async def store_bars(self, bars: list[MarketBar]) -> None:
        """Persist bars to the object store; idempotent on bar_id."""
        ...

    async def get_corporate_actions(
        self,
        instrument_id: uuid.UUID,
        since: date,
    ) -> list[CorporateAction]:
        """Return corporate actions affecting this instrument since the given date."""
        ...


@runtime_checkable
class HistoricalBarVendorAdapter(Protocol):
    """Vendor-neutral historical bar fetch/import interface.

    Implementations may wrap IBKR, a paid market-data API, or immutable local
    files.  Production promotion code treats all adapters the same and relies
    on the returned :class:`VendorBarBatch` for lineage and coverage evidence.
    """

    async def fetch_bars(
        self,
        instruments: list[uuid.UUID],
        start: datetime,
        end: datetime,
        bar_seconds: int,
        *,
        as_of: datetime,
    ) -> VendorBarBatch:
        """Return canonical bars plus vendor/source coverage metadata."""
        ...


@runtime_checkable
class FeatureRepository(Protocol):
    """Persistence interface for computed FeatureVectors.

    Must never:
        Overwrite an existing vector for the same (instrument_id, as_of,
        feature_set_version) tuple.  Vectors are immutable once stored.
        Return vectors whose ``as_of`` or ``available_at`` is after the
        requested decision timestamp.
    """

    async def store_vector(self, vector: FeatureVector) -> None:
        """Persist a feature vector; raises if a duplicate key already exists."""
        ...

    async def get_vectors(
        self,
        instrument_ids: list[uuid.UUID],
        feature_set_version: str,
        as_of: datetime,
    ) -> list[FeatureVector]:
        """Return latest vectors point-in-time visible at or before ``as_of``."""
        ...

    async def prune(self, older_than: datetime) -> int:
        """Delete vectors with ``as_of < older_than`` and return the count.

        Added in Phase 4.4 to retire R-DAT-03.  Implementations that
        cannot safely delete (e.g. read-only replicas) may return 0.
        """
        ...


@runtime_checkable
class LiquidityProfileSnapshot(Protocol):
    """Minimal liquidity profile shape consumed across services."""

    @property
    def adv_shares_20d(self) -> float: ...


@runtime_checkable
class LiquidityProfileProvider(Protocol):
    """Read-only access to per-instrument liquidity profiles."""

    def get_profile(self, instrument_id: uuid.UUID) -> LiquidityProfileSnapshot | None:
        """Return the latest liquidity profile when available."""
        ...


@runtime_checkable
class SecurityMaster(Protocol):
    """Point-in-time instrument metadata source for live/research parity."""

    async def get_record(
        self,
        instrument_id: uuid.UUID,
        *,
        as_of: datetime,
    ) -> SecurityMasterRecord | None:
        """Return approved metadata known at ``as_of``."""
        ...

    async def require_record(
        self,
        instrument_id: uuid.UUID,
        *,
        as_of: datetime,
    ) -> SecurityMasterRecord:
        """Return metadata or raise when a live-required row is missing."""
        ...


@runtime_checkable
class InstrumentRepository(SecurityMaster, Protocol):
    """Durable security master, symbol history, and universe snapshots."""

    async def upsert_security_master_record(self, record: SecurityMasterRecord) -> None:
        """Insert or update one point-in-time security-master record."""
        ...

    async def add_symbol_history(self, history: SymbolHistory) -> None:
        """Persist one symbol-history interval."""
        ...

    async def resolve_symbol(self, symbol: str, *, as_of: datetime) -> uuid.UUID | None:
        """Resolve a symbol to the instrument active at ``as_of``."""
        ...

    async def save_universe_snapshot(self, snapshot: UniverseSnapshot) -> None:
        """Persist an immutable universe snapshot."""
        ...

    async def latest_universe_snapshot(
        self,
        universe_name: str,
        *,
        as_of: datetime,
    ) -> UniverseSnapshot | None:
        """Return latest universe snapshot available at or before ``as_of``."""
        ...


@runtime_checkable
class DatasetCatalog(Protocol):
    """Registry for immutable bar and feature datasets."""

    async def register_bar_dataset(self, dataset: BarDataset) -> None:
        """Persist a versioned market-data dataset manifest."""
        ...

    async def register_feature_dataset(self, dataset: FeatureDataset) -> None:
        """Persist a versioned feature dataset manifest."""
        ...

    async def latest_feature_dataset(
        self,
        feature_set_version: str,
        *,
        as_of: datetime,
        min_quality: str = "approved",
    ) -> FeatureDataset | None:
        """Return latest feature dataset available for live scoring."""
        ...

    async def record_quorum_evidence(self, evidence: DatasetQuorumEvidence) -> None:
        """Persist vendor-quorum evidence for a dataset family."""
        ...

    async def latest_quorum_evidence(
        self,
        dataset_kind: str,
        *,
        as_of: datetime,
    ) -> DatasetQuorumEvidence | None:
        """Return latest vendor-quorum evidence available at or before as_of."""
        ...
