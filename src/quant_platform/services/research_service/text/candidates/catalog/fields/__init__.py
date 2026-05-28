"""Text-candidate field lists and promoted feature names."""

from __future__ import annotations

TEXT_AGGREGATE_LOOKBACK_DAYS = 21
TEXT_AGGREGATE_SUM_CLIP = 3.0
TEXT_AGGREGATE_WINDOWS = (7, 21, 42)

TEXT_CATALYST_V10_PROMOTED_CANDIDATES = (
    "v10_stability_abs_text_specificity_event_surprise_21d",
    "v10_stability_abs_text_tone_cov40_minus_vol_tone_21d",
    "v10_stability_abs_text_specificity_forward_outlook_21d",
)
