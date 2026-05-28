"""Text feature extraction errors."""

from __future__ import annotations

from quant_platform.core.exceptions import QuantPlatformError


class FeatureExtractionError(QuantPlatformError):
    """Raised when the LLM API call fails or returns an unparseable response."""


class TextFeatureCacheMissError(FeatureExtractionError):
    """Raised when replay-only extraction cannot find a cached artifact."""


class TextFeatureProviderError(FeatureExtractionError):
    """Raised when a retryable provider call ultimately fails."""


class TextFeatureBudgetError(FeatureExtractionError):
    """Raised when extraction would exceed provider call or cost budgets."""


class TextFeatureLatencyError(FeatureExtractionError):
    """Raised when a provider response exceeds the configured latency budget."""
