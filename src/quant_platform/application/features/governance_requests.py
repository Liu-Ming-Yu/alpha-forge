"""Request and result DTOs for feature-governance use cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@dataclass(frozen=True)
class FeatureAuditRunRequest:
    feature_card: Path
    samples: Path | None
    contracts_file: str | None
    start: datetime | None
    end: datetime | None
    feature_set_version: str
    horizon_days: int
    bar_seconds: int
    max_feature_age_days: int
    output_root: Path | None
    baseline_features: str
    slippage_bps_per_turnover: float
    min_daily_groups: int
    min_coverage: float
    min_oos_ic: float
    min_icir: float
    max_negative_ic_streak: int
    max_turnover: float
    persist: bool


@dataclass(frozen=True)
class FeatureAuditStatusRequest:
    feature_name: str | None
    feature_version: str | None
    limit: int
    output_root: Path | None


@dataclass(frozen=True)
class FeatureAuditAssertRequest:
    manifest: Path | None
    feature_name: str | None
    feature_version: str | None
    minimum_state: str


@dataclass(frozen=True)
class FeatureAuditRetireRequest:
    feature_name: str
    feature_version: str
    feature_set_version: str
    reason: str


@dataclass(frozen=True)
class CampaignFeatureAuditRequest:
    samples: Sequence[SupervisedAlphaSample]
    feature_set_version: str
    horizon_days: int
    slippage_bps_per_turnover: float
    mode: str
    feature_card_dir: Path | None
    candidate_feature_names: Sequence[str] | None = None


@dataclass(frozen=True)
class FeatureAuditCommandResult:
    payload: dict[str, object]
    passed: bool = True
