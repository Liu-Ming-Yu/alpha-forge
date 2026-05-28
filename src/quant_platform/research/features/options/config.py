"""Configuration for the ``options-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

#: Family version. Bump on formula change or schema change. v1 ships
#: 6 features against the scaffolded OptionsSnapshot input contract;
#: the real options-chain vendor feed wiring is a separate (deferred)
#: PR. Future v2 would add per-contract greeks (gamma exposure, vanna,
#: charm) once raw-chain data lands.
FEATURE_SET_VERSION: str = "options-v1"

#: Default expiry tenor (calendar days) for the ATM IV feature.
#: 30-day ATM is the CBOE-standard reference point — same expiry the
#: VIX is calculated against. Appears in feature column names.
DEFAULT_ATM_TENOR_DAYS: int = 30

#: Default expiry tenor used as the "long" end of the IV term-slope
#: feature. Appears in the column name.
DEFAULT_TERM_LONG_TENOR_DAYS: int = 60

#: Default realized-vol window for the IV-realized premium feature.
#: 21 trading days matches the typical ATM IV expiry on a 30-cal-day
#: basis (21 trading days ≈ 30 calendar days). Appears in the column
#: name.
DEFAULT_REALIZED_VOL_WINDOW_DAYS: int = 21


@dataclass(frozen=True)
class OptionsConfig(BaseFamilyConfig):
    """Frozen config for the options feature factory.

    Attributes
    ----------
    version:
        Family-version string stamped into every emitted FeatureSpec.
    atm_tenor_days:
        ATM IV expiry tenor (calendar days). 30 is the CBOE-standard
        reference. Appears in column names.
    term_long_tenor_days:
        Long-end expiry tenor for the IV term-slope feature. Must
        be strictly greater than ``atm_tenor_days``.
    realized_vol_window_days:
        Trailing realized-vol window in trading days. Appears in the
        ``iv_realized_premium_<N>d`` column name.
    """

    version: str = FEATURE_SET_VERSION
    atm_tenor_days: int = DEFAULT_ATM_TENOR_DAYS
    term_long_tenor_days: int = DEFAULT_TERM_LONG_TENOR_DAYS
    realized_vol_window_days: int = DEFAULT_REALIZED_VOL_WINDOW_DAYS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.atm_tenor_days < 1:
            raise ValueError("OptionsConfig.atm_tenor_days must be >= 1")
        if self.term_long_tenor_days < 1:
            raise ValueError("OptionsConfig.term_long_tenor_days must be >= 1")
        if self.term_long_tenor_days <= self.atm_tenor_days:
            raise ValueError(
                "OptionsConfig.term_long_tenor_days must be strictly greater than atm_tenor_days"
            )
        if self.realized_vol_window_days < 2:
            raise ValueError("OptionsConfig.realized_vol_window_days must be >= 2")


DEFAULT_CONFIG: OptionsConfig = OptionsConfig()


__all__ = [
    "DEFAULT_ATM_TENOR_DAYS",
    "DEFAULT_CONFIG",
    "DEFAULT_REALIZED_VOL_WINDOW_DAYS",
    "DEFAULT_TERM_LONG_TENOR_DAYS",
    "FEATURE_SET_VERSION",
    "OptionsConfig",
]
