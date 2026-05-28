"""Typed request DTOs for operator-facing application use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import uuid
    from datetime import datetime
    from decimal import Decimal
    from pathlib import Path


@dataclass(frozen=True)
class NoInputRequest:
    """Use-case request with no command-specific input."""


@dataclass(frozen=True)
class RunCycleRequest:
    initial_cash: Decimal


@dataclass(frozen=True)
class SuperviseRequest:
    initial_cash: Decimal
    interval_seconds: float
    mode: str = "paper"
    max_cycles: int | None = None
    contracts_file: str | None = None
    engine_name: str = "cross_sectional_equity"
    execution_backend: str = "simulated"


@dataclass(frozen=True)
class ServeApiRequest:
    initial_cash: Decimal
    host: str
    port: int


@dataclass(frozen=True)
class BrokerContractsRequest:
    contracts_file: str


@dataclass(frozen=True)
class PaperLifecycleRequest:
    contracts_file: str
    instrument_id: uuid.UUID
    max_notional_usd: Decimal


@dataclass(frozen=True)
class PassiveRepriceRequest:
    mode: str
    contracts_file: str
    initial_cash: Decimal


@dataclass(frozen=True)
class RunEngineRequest:
    mode: str
    initial_cash: Decimal
    cycles: int
    contracts_file: str | None
    engine_name: str
    execution_backend: str


@dataclass(frozen=True)
class RunMultiEngineRequest:
    mode: str
    engine_names: tuple[str, ...]
    budgets_file: str | None
    cycles: int
    initial_cash: Decimal
    contracts_file: str | None


@dataclass(frozen=True)
class EventBusSweepRequest:
    stream: str


@dataclass(frozen=True)
class PreflightRequest:
    profile: str
    contracts_file: str | None


@dataclass(frozen=True)
class FactorsCalibrateRequest:
    samples_path: Path
    output_dir: Path
    horizon_days: int
    l2_lambda: float
    momentum_scale: float


@dataclass(frozen=True)
class TearsheetRequest:
    run_id: uuid.UUID
    root: Path


@dataclass(frozen=True)
class IntradayValidateRequest:
    """Validate a vendor intraday bar file against the canonical schema."""

    input: Path
    vendor: str
    contracts_file: str
    as_of: datetime


@dataclass(frozen=True)
class IntradayImportRequest:
    """Validate and store a vendor intraday bar file."""

    input: Path
    vendor: str
    contracts_file: str
    as_of: datetime
    allow_quarantined: bool


@dataclass(frozen=True)
class IntradayFetchRequest:
    """Fetch intraday bars from a configured external vendor and store them."""

    vendor: str
    contracts_file: str
    start: datetime
    end: datetime
    as_of: datetime
    output_file: Path | None
    allow_quarantined: bool


@dataclass(frozen=True)
class IntradayQuorumRequest:
    """Compute multi-vendor intraday quorum evidence."""

    vendor_file: tuple[str, ...]
    contracts_file: str
    as_of: datetime
    required_vendor_count: int
    max_disagreement_bps: Decimal


IntradayCommandRequest: TypeAlias = (
    IntradayValidateRequest | IntradayImportRequest | IntradayFetchRequest | IntradayQuorumRequest
)


@dataclass(frozen=True)
class IngestSecTextEventsRequest:
    """Download and store SEC filings as governed text events."""

    contracts_file: str
    start: datetime
    end: datetime
    cik_map_file: Path
    sec_user_agent: str
    forms: tuple[str, ...]
    timeout_seconds: float
    limit_per_symbol: int | None
    include_exhibits: bool
    artifact_root: Path | None


@dataclass(frozen=True)
class IngestNewsTextEventsRequest:
    """Download and store vendor news as governed text events."""

    vendor: str
    contracts_file: str
    start: datetime
    end: datetime
    provider_codes: tuple[str, ...]
    total_results_per_symbol: int
    include_article_text: bool
    artifact_root: Path | None


@dataclass(frozen=True)
class ExtractTextFeaturesRequest:
    """Extract event-level LLM text features into the feature repository."""

    start: datetime
    end: datetime
    prompt_version: str
    document_role: str
    source_data_manifest: Path | None
    artifact_root: Path | None
    concurrency: int = 1
    status_file: Path | None = None
    per_call_timeout_seconds: float = 180.0


TextEventsRequest: TypeAlias = (
    IngestSecTextEventsRequest | IngestNewsTextEventsRequest | ExtractTextFeaturesRequest
)


__all__ = [
    "BrokerContractsRequest",
    "EventBusSweepRequest",
    "ExtractTextFeaturesRequest",
    "FactorsCalibrateRequest",
    "IngestSecTextEventsRequest",
    "IntradayCommandRequest",
    "IntradayFetchRequest",
    "IntradayImportRequest",
    "IntradayQuorumRequest",
    "IntradayValidateRequest",
    "IngestNewsTextEventsRequest",
    "NoInputRequest",
    "PaperLifecycleRequest",
    "PassiveRepriceRequest",
    "PreflightRequest",
    "RunCycleRequest",
    "RunEngineRequest",
    "RunMultiEngineRequest",
    "ServeApiRequest",
    "SuperviseRequest",
    "TearsheetRequest",
    "TextEventsRequest",
]
