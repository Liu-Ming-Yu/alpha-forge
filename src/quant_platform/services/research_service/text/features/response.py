"""Response parsing for Anthropic-compatible text feature calls."""

from __future__ import annotations

import json
from typing import Any

from quant_platform.services.research_service.text.features.errors import (
    FeatureExtractionError,
)
from quant_platform.services.research_service.text.features.validation import first_text_block


def parse_message_response(message: object, *, provider_label: str) -> tuple[dict[str, Any], str]:
    content = getattr(message, "content", None)
    if not content:
        raise FeatureExtractionError(f"{provider_label} API returned empty content")

    raw_text = first_text_block(content).strip()
    if not raw_text:
        raise FeatureExtractionError(f"{provider_label} API returned no text content")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise FeatureExtractionError(f"API response is not valid JSON: {raw_text!r}") from exc

    if not isinstance(parsed, dict):
        raise FeatureExtractionError(f"API response is not a JSON object: {type(parsed)}")
    return parsed, raw_text


__all__ = ["parse_message_response"]
