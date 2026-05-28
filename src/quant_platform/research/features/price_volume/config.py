"""Configuration for the ``price-volume-starter-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quant_platform.research.features.contracts import BaseFamilyConfig

if TYPE_CHECKING:
    from quant_platform.research.features.transforms import MinPeriodsPolicy

FEATURE_SET_VERSION: str = "price-volume-starter-v1"

LOOKBACKS_RETURN: tuple[int, ...] = (1, 5, 10, 21, 63, 126, 252)
LOOKBACKS_VOL: tuple[int, ...] = (21, 63, 126)
LOOKBACK_AMIHUD: int = 20
LOOKBACK_DOLLAR_VOLUME: int = 20
LOOKBACK_VOLUME_ZSCORE: int = 20
LOOKBACK_HIGH_LOW_RANGE: int = 20
LOOKBACK_52W_HIGH: int = 252


@dataclass(frozen=True)
class PriceVolumeConfig(BaseFamilyConfig):
    """Frozen config for the price-volume starter feature factory."""

    min_periods_policy: MinPeriodsPolicy = "full"
    version: str = FEATURE_SET_VERSION

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_periods_policy not in ("full", "partial"):
            raise ValueError(f"unknown min_periods_policy: {self.min_periods_policy!r}")


DEFAULT_CONFIG: PriceVolumeConfig = PriceVolumeConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_SET_VERSION",
    "LOOKBACK_52W_HIGH",
    "LOOKBACK_AMIHUD",
    "LOOKBACK_DOLLAR_VOLUME",
    "LOOKBACK_HIGH_LOW_RANGE",
    "LOOKBACK_VOLUME_ZSCORE",
    "LOOKBACKS_RETURN",
    "LOOKBACKS_VOL",
    "PriceVolumeConfig",
]
