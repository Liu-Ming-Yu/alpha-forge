"""Risk limit settings."""

from __future__ import annotations

import uuid  # noqa: TC003 - Pydantic resolves postponed UUID annotations at runtime.
import warnings
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator


class RiskSettings(BaseModel):
    """Hard risk limits loaded at session start.

    These feed into ``RiskLimits`` construction.  Change them in .env or
    via env vars; never modify at runtime without an operator action.
    """

    max_single_name_weight: Decimal = Decimal("0.05")
    max_sector_weight: Decimal = Decimal("0.20")
    # Cash-account-safe default. Production deployments that need higher
    # gross exposure must set ``QP__RISK__MAX_GROSS_EXPOSURE`` explicitly;
    # values above 0.85 emit a ``UserWarning`` so a misconfiguration is
    # visible at session startup.
    max_gross_exposure: Decimal = Decimal("0.60")
    max_daily_turnover: Decimal = Decimal("0.20")
    min_cash_buffer: Decimal = Decimal("0.05")
    max_drawdown_halt: Decimal = Decimal("-0.15")
    vol_target_annualised: Decimal | None = None
    auto_correct_threshold: int = Field(
        default=1,
        description="Position mismatch (shares) at or below which reconciliation auto-corrects",
    )
    require_sector_mapping: bool = Field(
        default=False,
        description=(
            "Fail closed at session start when any tradable instrument is "
            "missing a sector mapping.  Enabling this flag converts what was "
            "previously a silent 'unknown-sector concentration' risk into a "
            "startup error.  Flip to True in production via "
            "QP__RISK__REQUIRE_SECTOR_MAPPING=true."
        ),
    )
    require_registered_model_match: bool = Field(
        default=False,
        description=(
            "Strict mode for the session-start model-registry preflight "
            "(Phase 3.4).  When True, a live session refuses to start "
            "unless the active RegisteredModel for the strategy has an "
            "engine_version matching the running engine.  When False, a "
            "mismatch only logs a ``session.model.preflight.mismatch`` "
            "warning.  Flip to True in production once a registered "
            "model is promoted; leave False for paper and CI."
        ),
    )
    halt_on_stale_features: bool = Field(
        default=False,
        description=(
            "When True, a feature maintenance job that runs but produces zero "
            "features causes the engine to raise DataStalenessError and abort "
            "the cycle.  When False (default), the error is logged at ERROR level "
            "and the cycle continues with the previously computed features. "
            "Configured via QP__RISK__HALT_ON_STALE_FEATURES."
        ),
    )
    etf_correlation_groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Map of group_name -> list of instrument symbols. ETFs in the same group "
            "share a combined weight cap of max_single_name_weight x etf_group_cap_multiplier. "
            'Example: {"us_broad": ["SPY", "QQQ", "IWM"]}. '
            "Configured via QP__RISK__ETF_CORRELATION_GROUPS as a JSON string."
        ),
    )
    etf_group_cap_multiplier: float = Field(
        default=1.5,
        description=(
            "Combined weight cap for an ETF correlation group = "
            "max_single_name_weight x etf_group_cap_multiplier. "
            "Default 1.5 means a group of ETFs can hold up to 1.5x the single-name limit combined. "
            "Configured via QP__RISK__ETF_GROUP_CAP_MULTIPLIER."
        ),
    )
    pdt_enabled: bool = Field(
        default=True,
        description=(
            "Enable Pattern Day Trader detection.  When True, a warning is emitted "
            "(not a hard block) if day_trades_today >= 3 and account NAV < $25,000. "
            "Configured via QP__RISK__PDT_ENABLED."
        ),
    )
    halted_instruments: set[uuid.UUID] = Field(
        default_factory=set,
        description=(
            "Set of instrument UUIDs that are currently halted.  Orders for any "
            "instrument in this set are blocked with BLOCK severity.  "
            "Configure via QP__RISK__HALTED_INSTRUMENTS as a JSON array of UUID strings."
        ),
    )
    wash_sale_lookback_days: int = Field(
        default=30,
        description=(
            "Number of calendar days to look back when checking for wash-sale risk. "
            "A WARN (not BLOCK) violation is emitted if the same instrument was sold "
            "within this window.  Configured via QP__RISK__WASH_SALE_LOOKBACK_DAYS."
        ),
    )
    max_data_age_minutes: int = Field(
        default=60,
        description=(
            "Maximum acceptable age (minutes) for market data during preflight checks. "
            "Preflight fails if the reference instrument's last bar is older than this. "
            "Configured via QP__RISK__MAX_DATA_AGE_MINUTES."
        ),
    )
    max_model_age_hours: float = Field(
        default=48.0,
        description=(
            "Maximum acceptable model age in hours.  When a registered model's "
            "created_at is older than this threshold at initialize() time, "
            "DataStalenessError is raised so the engine refuses to run on a stale model. "
            "Set to 0 to disable the check.  Configured via QP__RISK__MAX_MODEL_AGE_HOURS."
        ),
    )
    mean_shift_alert_threshold: float = Field(
        default=2.0,
        description=(
            "Z-score threshold for feature distribution mean-shift detection.  "
            "When a feature's current mean deviates from its EMA by more than this many "
            "standard deviations, a WARNING is emitted.  "
            "Configured via QP__RISK__MEAN_SHIFT_ALERT_THRESHOLD."
        ),
    )

    @field_validator("max_gross_exposure")
    @classmethod
    def _warn_high_gross_exposure(cls, v: Decimal) -> Decimal:
        if v > Decimal("0.85"):
            warnings.warn(
                f"max_gross_exposure={v} is above the 0.85 cash-account guidance; "
                "ensure margin/leverage controls are intentional and audited.",
                UserWarning,
                stacklevel=2,
            )
        return v
