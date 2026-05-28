"""LLM text feature extractor runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from quant_platform.services.research_service.text.features.artifacts import (
    write_extraction_artifact,
)
from quant_platform.services.research_service.text.features.budget import TextFeatureBudget
from quant_platform.services.research_service.text.features.client import (
    AnthropicClient,
    build_llm_client,
)
from quant_platform.services.research_service.text.features.errors import (
    FeatureExtractionError,
    TextFeatureCacheMissError,
    TextFeatureLatencyError,
    TextFeatureProviderError,
)
from quant_platform.services.research_service.text.features.prompts import (
    INPUT_SEPARATOR,
    INPUT_SUFFIX,
    MAX_TEXT_CHARS,
    PROMPTS,
)
from quant_platform.services.research_service.text.features.provider_errors import (
    is_retryable_provider_error,
)
from quant_platform.services.research_service.text.features.replay_cache import (
    load_cached_extraction_vector,
)
from quant_platform.services.research_service.text.features.response import (
    parse_message_response,
)
from quant_platform.services.research_service.text.features.validation import (
    validate_text_features,
)
from quant_platform.services.research_service.text.features.vectors import (
    build_text_feature_vector,
    prepare_text_content,
)

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from quant_platform.core.domain.market_data.text_events import TextEvent
    from quant_platform.core.domain.research import FeatureVector

log = structlog.get_logger(__name__)

_LLM_API_ATTEMPTS = 3
_LLM_RETRY_BACKOFF_SECONDS = 2.0


class LLMTextFeatureExtractor:
    def __init__(
        self,
        *,
        provider: Literal["anthropic", "deepseek"] = "anthropic",
        model: str = "claude-sonnet-4-6",
        prompt_version: str = "v1",
        max_tokens: int = 512,
        timeout_seconds: float = 30.0,
        deepseek_base_url: str = "https://api.deepseek.com/anthropic",
        cache: dict[tuple[uuid.UUID, str, str, str, str], FeatureVector] | None = None,
        artifact_root: Path | str | None = None,
        replay_only: bool = False,
        max_request_latency_seconds: float | None = None,
        max_daily_calls: int | None = None,
        max_daily_estimated_cost_usd: float | None = None,
        estimated_cost_per_call_usd: float = 0.0,
    ) -> None:
        if prompt_version not in PROMPTS:
            raise ValueError(
                f"Unknown prompt_version={prompt_version!r}. Available: {sorted(PROMPTS)}"
            )
        if provider not in {"anthropic", "deepseek"}:
            raise ValueError("provider must be 'anthropic' or 'deepseek'")
        self._provider = provider
        self._model = model
        self._prompt_version = prompt_version
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._deepseek_base_url = deepseek_base_url.rstrip("/")
        self._cache: dict[tuple[uuid.UUID, str, str, str, str], FeatureVector] = (
            cache if cache is not None else {}
        )
        self._artifact_root = Path(artifact_root) if artifact_root is not None else None
        self._replay_only = replay_only
        self._max_request_latency_seconds = max_request_latency_seconds
        self._budget = TextFeatureBudget(
            provider=self._provider,
            model=self._model,
            prompt_version=self._prompt_version,
            replay_only=self._replay_only,
            max_daily_calls=max_daily_calls,
            max_daily_estimated_cost_usd=max_daily_estimated_cost_usd,
            estimated_cost_per_call_usd=float(estimated_cost_per_call_usd),
        )
        self._client: AnthropicClient | None = None

    def extract(
        self,
        event: TextEvent,
        text_content: str,
        strategy_run_id: uuid.UUID,
        *,
        as_of: datetime | None = None,
    ) -> FeatureVector:
        """Extract structured features from raw text."""
        prepared_content, content_digest, lineage = prepare_text_content(
            event,
            text_content,
            prompt_version=self._prompt_version,
        )
        cache_key = (
            event.event_id,
            self._prompt_version,
            self._model,
            self._provider,
            content_digest,
        )
        if cache_key in self._cache:
            log.debug(
                "text_extractor.cache_hit",
                event_id=str(event.event_id),
                prompt_version=self._prompt_version,
            )
            return self._cache[cache_key]

        cached_vector = load_cached_extraction_vector(
            artifact_root=self._artifact_root,
            provider=self._provider,
            model=self._model,
            prompt_version=self._prompt_version,
            event=event,
            content_digest=content_digest,
            strategy_run_id=strategy_run_id,
            as_of=as_of,
            cache_key=cache_key,
            cache=self._cache,
            safe_log=self._safe_log,
        )
        if cached_vector is not None:
            return cached_vector
        if self._replay_only:
            raise TextFeatureCacheMissError(
                "LLM replay-only mode cache miss: "
                f"event_id={event.event_id} provider={self._provider} "
                f"model={self._model} prompt_version={self._prompt_version}"
            )

        raw_features, raw_response = self._call_api_with_raw(prepared_content)
        features = validate_text_features(raw_features, prompt_version=self._prompt_version)
        extraction_artifact = write_extraction_artifact(
            artifact_root=self._artifact_root,
            provider=self._provider,
            model=self._model,
            prompt_version=self._prompt_version,
            event=event,
            content_digest=content_digest,
            source_artifact_uri=event.artifact_uri,
            features=features,
            raw_response=raw_response,
            lineage=lineage,
            runtime_metadata=self._budget.metadata,
        )

        artifact_uri = (
            f"{extraction_artifact}#prompt={self._prompt_version}"
            if extraction_artifact
            else f"{event.artifact_uri}#prompt={self._prompt_version}"
        )
        vector = build_text_feature_vector(
            event=event,
            strategy_run_id=strategy_run_id,
            as_of=as_of,
            features=features,
            feature_set_version=f"text-{self._prompt_version}",
            artifact_uri=artifact_uri,
        )
        self._cache[cache_key] = vector
        self._safe_log(
            "info",
            "text_extractor.extracted",
            event_id=str(event.event_id),
            event_type=event.event_type.value,
            instrument_id=str(event.instrument_id) if event.instrument_id else None,
            sentiment=features.get("text_sentiment"),
            guidance=features.get("guidance_direction"),
        )
        return vector

    def _get_client(self) -> AnthropicClient:
        if self._client is None:
            self._client = build_llm_client(
                provider=self._provider,
                timeout_seconds=self._timeout,
                deepseek_base_url=self._deepseek_base_url,
            )
        return self._client

    def _call_api_with_raw(self, text_content: str) -> tuple[dict[str, Any], str]:
        if len(text_content) > MAX_TEXT_CHARS:
            log.warning(
                "text_extractor.truncated",
                original_chars=len(text_content),
                truncated_to=MAX_TEXT_CHARS,
            )
            text_content = text_content[:MAX_TEXT_CHARS]
        system_prompt = PROMPTS[self._prompt_version]
        bounded_content = INPUT_SEPARATOR + text_content + INPUT_SUFFIX
        return self._call_messages_api(system_prompt, bounded_content)

    def _call_messages_api(
        self,
        system_prompt: str,
        bounded_content: str,
    ) -> tuple[dict[str, Any], str]:
        client = self._get_client()
        provider_label = "DeepSeek" if self._provider == "deepseek" else "Anthropic"
        for attempt in range(_LLM_API_ATTEMPTS):
            started = time.monotonic()
            self._budget.assert_available()
            self._budget.record_provider_attempt()
            try:
                message = client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": bounded_content}],
                )
                parsed, raw_text = parse_message_response(
                    message,
                    provider_label=provider_label,
                )
            except FeatureExtractionError:
                self._observe_latency("error", started)
                raise
            except Exception as exc:
                self._observe_latency("error", started)
                if is_retryable_provider_error(exc) and attempt < _LLM_API_ATTEMPTS - 1:
                    self._safe_log(
                        "warning",
                        "text_extractor.api_retry",
                        provider=self._provider,
                        model=self._model,
                        prompt_version=self._prompt_version,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                    time.sleep(_LLM_RETRY_BACKOFF_SECONDS * (attempt + 1))
                    continue
                raise TextFeatureProviderError(f"{provider_label} API call failed: {exc}") from exc
            latency_seconds = self._observe_latency("ok", started)
            within_latency_limit = (
                self._max_request_latency_seconds is None
                or latency_seconds <= self._max_request_latency_seconds
            )
            self._budget.metadata.update(
                {
                    "latency_seconds": latency_seconds,
                    "max_request_latency_seconds": self._max_request_latency_seconds,
                    "within_latency_limit": within_latency_limit,
                }
            )
            if not within_latency_limit:
                raise TextFeatureLatencyError(
                    "LLM provider latency budget breached: "
                    f"latency_seconds={latency_seconds:.6f} "
                    f"max_request_latency_seconds={self._max_request_latency_seconds}"
                )
            return parsed, raw_text

        raise FeatureExtractionError(f"{provider_label} API returned no message")

    def _safe_log(self, level: str, event: str, **kwargs: object) -> None:
        try:
            getattr(log, level)(event, **kwargs)
        except OSError:
            return

    def _observe_latency(self, outcome: str, started: float) -> float:
        elapsed = time.monotonic() - started
        try:
            from quant_platform.telemetry.metrics import (
                observe_alpha_text_extraction_latency,
            )

            observe_alpha_text_extraction_latency(self._model, outcome, elapsed)
        except Exception as exc:
            self._safe_log("debug", "text_extractor.metric_observe_failed", error=str(exc))
        return elapsed
