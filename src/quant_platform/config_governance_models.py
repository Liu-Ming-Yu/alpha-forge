"""Production, V2, backtest, and regime-governance settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ProductionSettings(BaseModel):
    """Production-readiness gate thresholds."""

    profile_preset: Literal["dev", "industrial"] = Field(
        default="dev",
        description=(
            "Use 'industrial' to make live-grade production settings mandatory "
            "at PlatformSettings validation time."
        ),
    )
    data_health_min_coverage_pct: float = 1.0
    data_health_min_liquidity_coverage_pct: float = 1.0
    data_health_stale_after_days: int = 3
    text_gate_min_observations: int = 20
    text_gate_min_ic: float = 0.05
    text_gate_max_negative_streak: int = 3
    signal_gate_max_drawdown: float = -0.10
    signal_gate_max_turnover: float = 1.0
    heartbeat_stale_after_minutes: int = 10
    prediction_evidence_stale_after_hours: int = Field(
        default=24,
        ge=1,
        description="Maximum age for prediction evidence used by promotion gates.",
    )
    prediction_evidence_min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum mean confidence for promoted forecast evidence.",
    )


class V2Settings(BaseModel):
    """V2 institutional-production gate switches."""

    enabled: bool = False
    account_orchestrator_enabled: bool = False
    require_security_master: bool = False
    require_feature_datasets: bool = False
    require_event_sourced_oms: bool = False
    require_dataset_quorum: bool = False
    third_eod_vendor: str = ""
    readiness_snapshot_required: bool = False
    max_feature_age_seconds: int = 86_400


class BacktestSettings(BaseModel):
    """Research-to-production parity controls for the backtest engine."""

    require_market_regime: bool = True
    require_intraday_evidence: bool = Field(
        default=False,
        description=(
            "When True, production-candidate gates require a passing "
            "backtest_evidence_manifest.json with dual-engine intraday "
            "reconciliation. Keep False for daily/dev workflows; "
            "set QP__BACKTEST__REQUIRE_INTRADAY_EVIDENCE=true for "
            "industrial intraday promotion."
        ),
    )


class RegimeThresholdsSettings(BaseModel):
    """Calibration thresholds for ``MarketRegimeDetector``."""

    crisis_vol: float = 0.35
    risk_off_vol: float = 0.25
    low_vol: float = 0.20
    downtrend_z: float = -0.05
    uptrend_z: float = 0.02
    weak_breadth: float = 0.40
    strong_breadth: float = 0.55


class RegimeSettings(BaseModel):
    """Market-regime detector parameters."""

    enabled: bool = True
    market_proxy_instrument_id: str = ""
    trend_window: int = 200
    vol_window: int = 21
    breadth_window: int = 50
    bar_seconds: int = 86400
    lookback_days: int = 380
    thresholds: RegimeThresholdsSettings = RegimeThresholdsSettings()
    require_seed_on_cycle: bool = Field(
        default=False,
        description=(
            "When True together with ``enabled=True``, ``run_strategy_cycle`` "
            "fails the cycle if the bar store cannot produce MarketStats "
            "(proxy unconfigured, empty bars, or fetch failure). Otherwise "
            "the detector warns and returns TRANSITION with confidence=0. "
            "Flip to True in production once the proxy is seeded; leave "
            "False in paper/tests that run without a proxy."
        ),
    )
    disagree_confidence_haircut: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Multiplier applied to regime confidence when the candidate "
            "label differs from the stable label, so brief regime "
            "disagreements scale risk down without flipping outright. "
            "Default 0.75 preserves the original hard-coded haircut; "
            "lower values penalise disagreement more aggressively."
        ),
    )

    @field_validator("market_proxy_instrument_id")
    @classmethod
    def _validate_proxy_uuid(cls, v: str) -> str:
        if v:
            import uuid as _uuid

            try:
                _uuid.UUID(v)
            except ValueError as exc:
                raise ValueError(
                    f"market_proxy_instrument_id must be a valid UUID, got {v!r}"
                ) from exc
        return v

    @model_validator(mode="after")
    def _require_proxy_when_seed_enforced(self) -> RegimeSettings:
        if self.enabled and self.require_seed_on_cycle and not self.market_proxy_instrument_id:
            raise ValueError(
                "market_proxy_instrument_id must be set when both "
                "enabled=True and require_seed_on_cycle=True. "
                "Set QP__REGIME__MARKET_PROXY_INSTRUMENT_ID."
            )
        return self
