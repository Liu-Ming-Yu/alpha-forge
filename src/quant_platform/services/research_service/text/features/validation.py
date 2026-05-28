"""Pure validation helpers for LLM text feature payloads."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from quant_platform.services.research_service.text.features.errors import (
    FeatureExtractionError,
)

FEATURE_KEYS_V1 = (
    "text_sentiment",
    "guidance_direction",
    "revenue_revision_magnitude",
    "macro_sentiment",
)

FEATURE_KEYS_V2 = (
    *FEATURE_KEYS_V1,
    "catalyst_sentiment",
    "earnings_quality",
    "forward_outlook",
)

FEATURE_KEYS_V3 = (
    *FEATURE_KEYS_V2,
    "event_surprise",
    "guidance_specificity",
    "risk_pressure",
    "revision_clarity",
)

FEATURE_KEYS_V4 = (
    *FEATURE_KEYS_V3,
    "operating_quality",
    "demand_outlook",
    "margin_resilience",
    "disclosure_specificity",
)

FEATURE_KEYS = FEATURE_KEYS_V1
FEATURE_KEYS_BY_PROMPT_VERSION = {
    "v1": FEATURE_KEYS_V1,
    "v2": FEATURE_KEYS_V2,
    "v3": FEATURE_KEYS_V3,
    "v4": FEATURE_KEYS_V4,
    "v5": FEATURE_KEYS_V4,
}

RANGES: dict[str, tuple[float, float]] = {
    "text_sentiment": (-1.0, 1.0),
    "guidance_direction": (-1.0, 1.0),
    "revenue_revision_magnitude": (0.0, 1.0),
    "macro_sentiment": (-1.0, 1.0),
    "catalyst_sentiment": (-1.0, 1.0),
    "earnings_quality": (-1.0, 1.0),
    "forward_outlook": (-1.0, 1.0),
    "event_surprise": (-1.0, 1.0),
    "guidance_specificity": (0.0, 1.0),
    "risk_pressure": (0.0, 1.0),
    "revision_clarity": (0.0, 1.0),
    "operating_quality": (-1.0, 1.0),
    "demand_outlook": (-1.0, 1.0),
    "margin_resilience": (-1.0, 1.0),
    "disclosure_specificity": (0.0, 1.0),
}


def first_text_block(content: object) -> str:
    """Return the first text block from Anthropic-compatible message content."""
    if not content:
        return ""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, Iterable):
        return ""
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return text
        if isinstance(block, dict):
            raw = block.get("text")
            if isinstance(raw, str) and raw.strip():
                return raw
    return ""


def validate_text_features(raw: dict[str, Any], prompt_version: str = "v1") -> dict[str, float]:
    keys = FEATURE_KEYS_BY_PROMPT_VERSION.get(prompt_version)
    if keys is None:
        raise FeatureExtractionError(f"Unknown text prompt version {prompt_version!r}")
    features: dict[str, float] = {}
    for key in keys:
        if key not in raw:
            raise FeatureExtractionError(f"Missing required feature key {key!r} in API response")
        try:
            value = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise FeatureExtractionError(f"Feature {key!r} is not numeric: {raw[key]!r}") from exc
        lo, hi = RANGES[key]
        if not (lo <= value <= hi):
            raise FeatureExtractionError(
                f"Feature {key!r}={value} is outside allowed range [{lo}, {hi}]"
            )
        features[key] = value
    return features
