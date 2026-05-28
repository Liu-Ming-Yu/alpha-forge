"""Configuration for the ``text-event-v2`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.research.features.contracts import BaseFamilyConfig

#: Feature-set version. ``v2`` covers all three document types
#: (news + filings + earnings calls) — 27 features total. ``v1`` is
#: still readable from old extraction JSONL but its 5 features are a
#: strict subset of v2.
FEATURE_SET_VERSION: str = "text-event-v2"

#: Default rolling window for the volume-z-score feature, in
#: trading days.
DEFAULT_VOLUME_ZSCORE_WINDOW: int = 20

#: Default rolling window for the multi-day sentiment-mean feature
#: (``news_sentiment_5d``), in trading days.
DEFAULT_SENTIMENT_WINDOW: int = 5

#: Default rolling window for the management-tone-change feature, in
#: trading days. Compares the average call/filing tone over the most
#: recent ``window`` days to the average over the prior ``window``
#: days. 90d ≈ one calendar quarter.
DEFAULT_TONE_CHANGE_WINDOW: int = 90


@dataclass(frozen=True)
class TextEventConfig(BaseFamilyConfig):
    """Frozen config for the text-event feature factory.

    Attributes
    ----------
    version:
        Feature-set version. Defaults to :data:`FEATURE_SET_VERSION`.
    volume_zscore_window:
        Rolling window for the ``news_volume_zscore_*d`` feature, in
        trading days. Bumping this requires a feature-set version
        bump because the rolling-window appears in the feature
        column name.
    sentiment_window:
        Rolling window for ``news_sentiment_*d``. Same versioning
        rule as :attr:`volume_zscore_window`.
    tone_change_window:
        Rolling window for ``management_tone_change``. Same versioning
        rule as :attr:`volume_zscore_window`.
    """

    version: str = FEATURE_SET_VERSION
    volume_zscore_window: int = DEFAULT_VOLUME_ZSCORE_WINDOW
    sentiment_window: int = DEFAULT_SENTIMENT_WINDOW
    tone_change_window: int = DEFAULT_TONE_CHANGE_WINDOW

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.volume_zscore_window < 2:
            raise ValueError("TextEventConfig.volume_zscore_window must be >= 2")
        if self.sentiment_window < 2:
            raise ValueError("TextEventConfig.sentiment_window must be >= 2")
        if self.tone_change_window < 2:
            raise ValueError("TextEventConfig.tone_change_window must be >= 2")


DEFAULT_CONFIG: TextEventConfig = TextEventConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SENTIMENT_WINDOW",
    "DEFAULT_TONE_CHANGE_WINDOW",
    "DEFAULT_VOLUME_ZSCORE_WINDOW",
    "FEATURE_SET_VERSION",
    "TextEventConfig",
]
