"""Governed LLM/text-alpha settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    """Configuration for the governed text feature extractor."""

    provider: Literal["anthropic", "deepseek"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    deepseek_base_url: str = "https://api.deepseek.com/anthropic"
    text_prompt_version: str = "v1"
    max_tokens: int = Field(
        default=3000,
        ge=256,
        description=(
            "Max output tokens per LLM call. Must comfortably fit the configured "
            "extraction JSON (the v4 catalyst prompt emits 15 features ~700-1500 "
            "tokens); a tight cap truncates responses mid-stream and yields "
            "'API response is not valid JSON' failures."
        ),
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=300,
        description=(
            "HTTP timeout for one LLM API request (seconds). Default raised "
            "from 30 to 120 because larger max_tokens (3000) routinely makes a "
            "single call take 60-90s on DeepSeek; a 30s timeout caused retry "
            "storms whose cumulative latency tripped max_request_latency_seconds."
        ),
    )
    shadow_mode_enabled: bool = Field(
        default=False,
        description=(
            "Enable shadow text scoring: compute text features in parallel "
            "without including them in the live portfolio target. Must run "
            "for at least 20 trading days before promotion. Flip via "
            "QP__LLM__SHADOW_MODE_ENABLED=true."
        ),
    )
    live_mode_enabled: bool = Field(
        default=False,
        description=(
            "Enable live text-feature blending only after the live LLM startup "
            "assertion has passed. Set via QP__LLM__LIVE_MODE_ENABLED=true."
        ),
    )
    live_rehearsal_enabled: bool = Field(
        default=False,
        description=(
            "Allow QP__LLM__LIVE_MODE_ENABLED=true only for the non-capital "
            "LLM live rehearsal gate. Requires paper broker routing."
        ),
    )
    text_model_manifest: str = Field(
        default="",
        description=(
            "Explicit path to a governed text model manifest. Required when "
            "QP__LLM__LIVE_MODE_ENABLED=true."
        ),
    )
    text_feature_card_dir: str = Field(
        default="",
        description=(
            "Optional override for hash-pinned text feature cards. Defaults to "
            "infra/config/feature_cards/<text_feature_set_version>."
        ),
    )
    extraction_artifact_root: str = Field(
        default="",
        description=(
            "Optional text extraction artifact cache root. Defaults under "
            "QP__STORAGE__OBJECT_STORE_ROOT/research/text_events/extractions."
        ),
    )
    live_startup_assertion_path: str = Field(
        default="",
        description=(
            "Optional override for the live LLM startup assertion token. Defaults "
            "under QP__STORAGE__OBJECT_STORE_ROOT/governance/llm_live_startup_assertion.json."
        ),
    )
    live_startup_assertion_stale_after_hours: int = Field(
        default=24,
        ge=1,
        description="Maximum age of the live LLM startup assertion token.",
    )
    live_evidence_stale_after_days: int = Field(
        default=3,
        ge=1,
        description="Maximum age for live LLM manifest/evidence freshness checks.",
    )
    max_request_latency_seconds: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Maximum allowed latency for one LLM provider request. Platform "
            "invariant: must be <= timeout_seconds (enforced by the live preflight "
            "gate, runtime_limits._provider_budget_check). Set equal to timeout so "
            "a successful call within the HTTP timeout passes; lower to surface "
            "early-warn telemetry on slow-but-completing calls."
        ),
    )
    max_daily_calls: int = Field(
        default=1_000,
        ge=1,
        description="Daily LLM provider call budget for governed extraction.",
    )
    max_daily_estimated_cost_usd: float = Field(
        default=25.0,
        gt=0,
        description="Daily estimated LLM provider spend budget in USD.",
    )
    estimated_cost_per_call_usd: float = Field(
        default=0.01,
        gt=0,
        description="Conservative per-call cost estimate used for budget gating.",
    )
    replay_only_live: bool = Field(
        default=True,
        description=(
            "When live LLM mode is enabled, fail on extraction cache misses "
            "instead of making provider calls."
        ),
    )
    text_feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "v10_stability_abs_text_specificity_event_surprise_21d": 0.40,
            "v10_stability_abs_text_specificity_forward_outlook_21d": 0.35,
            "v10_stability_abs_text_tone_cov40_minus_vol_tone_21d": 0.25,
        },
        description=(
            "Weights for text features in LinearWeightSignalModel when the governed text "
            "source has positive paper/live ensemble weight."
        ),
    )
    text_feature_set_version: str = Field(
        default="paper-alpha-catalyst-v10",
        description="Audited paper/live feature_set_version for promoted text feature weights.",
    )
    text_feature_versions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional feature-version map for promoted text feature admission. When empty, "
            "each configured text feature weight uses text_feature_set_version."
        ),
    )
    ic_gate_min_ic: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum rolling Spearman IC required for text feature promotion.",
    )
    ic_gate_min_observations: int = Field(
        default=20,
        ge=1,
        description="Minimum number of daily IC observations required for promotion.",
    )
