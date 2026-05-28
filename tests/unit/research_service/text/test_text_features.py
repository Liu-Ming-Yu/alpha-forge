"""Unit tests for LLMTextFeatureExtractor and InMemoryTextEventStore."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from quant_platform.core.domain.market_data.text_events import TextEvent, TextEventType
from quant_platform.core.domain.research import FeatureVector
from quant_platform.infrastructure.repositories.feature_repository import InMemoryFeatureRepository
from quant_platform.services.data_service.text.text_event_store import (
    InMemoryTextEventStore,
    PostgresTextEventStore,
)
from quant_platform.services.research_service.text.extraction.sec_primary_compaction import (
    MAX_COMPACTED_TEXT_CHARS,
    SEC_PRIMARY_COMPACTION_POLICY,
    compact_sec_primary_text,
)
from quant_platform.services.research_service.text.extraction.text_event_extraction import (
    TextEventExtractionTarget,
    extract_text_event_features,
)
from quant_platform.services.research_service.text.features import (
    FeatureExtractionError,
    LLMTextFeatureExtractor,
    TextFeatureBudgetError,
    TextFeatureCacheMissError,
    TextFeatureLatencyError,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 9, 15, 14, 0, 0, tzinfo=UTC)
_RUN_ID = uuid.uuid4()


def _make_event(
    *,
    instrument_id: uuid.UUID | None = None,
    event_type: TextEventType = TextEventType.EARNINGS_TRANSCRIPT,
) -> TextEvent:
    return TextEvent(
        event_id=uuid.uuid4(),
        event_type=event_type,
        occurred_at=_NOW,
        source_uri="s3://bucket/raw.txt",
        artifact_uri="s3://bucket/artifact.txt",
        instrument_id=instrument_id,
    )


def _mock_api_response(payload: dict) -> MagicMock:
    """Build a minimal Anthropic message response mock."""
    content_block = SimpleNamespace(text=json.dumps(payload))
    return SimpleNamespace(content=[content_block])


_VALID_PAYLOAD = {
    "text_sentiment": 0.7,
    "guidance_direction": 1.0,
    "revenue_revision_magnitude": 0.4,
    "macro_sentiment": 0.2,
}

_VALID_PAYLOAD_V2 = {
    **_VALID_PAYLOAD,
    "catalyst_sentiment": 0.8,
    "earnings_quality": 0.5,
    "forward_outlook": 0.6,
}

_VALID_PAYLOAD_V3 = {
    **_VALID_PAYLOAD_V2,
    "event_surprise": 0.9,
    "guidance_specificity": 0.75,
    "risk_pressure": 0.2,
    "revision_clarity": 0.8,
}

_VALID_PAYLOAD_V4 = {
    **_VALID_PAYLOAD_V3,
    "operating_quality": 0.55,
    "demand_outlook": 0.65,
    "margin_resilience": -0.15,
    "disclosure_specificity": 0.85,
}

_VALID_PAYLOAD_V5 = dict(_VALID_PAYLOAD_V4)


# ---------------------------------------------------------------------------
# LLMTextFeatureExtractor
# ---------------------------------------------------------------------------


def test_sec_primary_compaction_selects_10k_sections_and_respects_bound() -> None:
    raw_text = "\n".join(
        [
            "dei:DocumentType us-gaap:Assets xbrli:shares iso4217:USD",
            "Item 1A. Risk Factors",
            "Risk pressure language. " * 500,
            "Item 7. Management's Discussion and Analysis",
            "Operating quality, revenue, margin, and liquidity discussion. " * 800,
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            "Market risk discussion. " * 300,
            "Item 8. Financial Statements and Supplementary Data",
            "Revenue and gross margin notes. " * 700,
        ]
    )

    compacted = compact_sec_primary_text(raw_text, form_type="10-K")

    assert compacted.policy_name == SEC_PRIMARY_COMPACTION_POLICY
    assert compacted.original_chars == len(raw_text)
    assert compacted.compacted_chars <= MAX_COMPACTED_TEXT_CHARS
    assert "item_7_mda" in compacted.selected_section_labels
    assert "item_1a_risk_factors" in compacted.selected_section_labels
    assert "dei:DocumentType" not in compacted.text
    assert "omitted" in compacted.text


def test_sec_primary_compaction_falls_back_when_no_anchors_exist() -> None:
    raw_text = "unstructured filing text " * 2_000

    compacted = compact_sec_primary_text(raw_text, form_type="UNKNOWN", max_chars=4_000)

    assert compacted.compacted_chars <= 4_000
    assert compacted.selected_section_labels == (
        "fallback_head",
        "fallback_middle",
        "fallback_tail",
    )


class TestLLMTextFeatureExtractor:
    def _extractor(self, **kwargs) -> LLMTextFeatureExtractor:
        return LLMTextFeatureExtractor(**kwargs)

    def test_extract_returns_feature_vector_with_correct_fields(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        vector = extractor.extract(event, "Q3 results beat estimates.", _RUN_ID, as_of=_NOW)

        assert vector.instrument_id == event.instrument_id
        assert vector.strategy_run_id == _RUN_ID
        assert vector.as_of == _NOW
        assert vector.feature_set_version == "text-v1"
        assert vector.features["text_sentiment"] == pytest.approx(0.7)
        assert vector.features["guidance_direction"] == pytest.approx(1.0)
        assert vector.features["revenue_revision_magnitude"] == pytest.approx(0.4)
        assert vector.features["macro_sentiment"] == pytest.approx(0.2)

    def test_artifact_uri_embeds_prompt_version(self) -> None:
        extractor = self._extractor(prompt_version="v1")
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        vector = extractor.extract(event, "text", _RUN_ID)
        assert "#prompt=v1" in vector.artifact_uri

    def test_extraction_artifact_caches_prompt_and_response(self, tmp_path) -> None:
        extractor = self._extractor(prompt_version="v1", artifact_root=tmp_path)
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        vector = extractor.extract(event, "text", _RUN_ID)

        artifact_path = tmp_path / "anthropic" / "claude-sonnet-4-6" / "v1"
        payload = json.loads(next(artifact_path.glob("*.json")).read_text(encoding="utf-8"))
        assert vector.artifact_uri.startswith(str(artifact_path))
        assert payload["system_prompt"]
        assert json.loads(payload["raw_response"]) == _VALID_PAYLOAD

    def test_extraction_artifact_records_runtime_metadata(self, tmp_path) -> None:
        extractor = self._extractor(
            prompt_version="v1",
            artifact_root=tmp_path,
            max_request_latency_seconds=30.0,
            max_daily_calls=10,
            max_daily_estimated_cost_usd=1.0,
            estimated_cost_per_call_usd=0.01,
        )
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        extractor.extract(event, "text", _RUN_ID)

        artifact_path = tmp_path / "anthropic" / "claude-sonnet-4-6" / "v1"
        payload = json.loads(next(artifact_path.glob("*.json")).read_text(encoding="utf-8"))
        metadata = payload["runtime_metadata"]
        assert metadata["provider"] == "anthropic"
        assert metadata["daily_calls_used"] == 1
        assert metadata["estimated_cost_usd"] == pytest.approx(0.01)
        assert metadata["within_latency_limit"] is True

    def test_replay_only_cache_miss_blocks_provider_call(self, tmp_path) -> None:
        extractor = self._extractor(
            prompt_version="v1",
            artifact_root=tmp_path,
            replay_only=True,
        )
        event = _make_event(instrument_id=uuid.uuid4())
        mock_client = MagicMock()
        extractor._client = mock_client

        with pytest.raises(TextFeatureCacheMissError, match="replay-only mode cache miss"):
            extractor.extract(event, "text", _RUN_ID)

        mock_client.messages.create.assert_not_called()

    def test_extract_v2_returns_catalyst_fields(self) -> None:
        extractor = self._extractor(prompt_version="v2")
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V2)
        extractor._client = mock_client

        vector = extractor.extract(event, "earnings release exhibit", _RUN_ID)

        assert vector.feature_set_version == "text-v2"
        assert vector.features["catalyst_sentiment"] == pytest.approx(0.8)
        assert vector.features["earnings_quality"] == pytest.approx(0.5)
        assert vector.features["forward_outlook"] == pytest.approx(0.6)

    def test_extract_v3_returns_surprise_specificity_fields(self) -> None:
        extractor = self._extractor(prompt_version="v3")
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V3)
        extractor._client = mock_client

        vector = extractor.extract(event, "earnings release exhibit", _RUN_ID)

        assert vector.feature_set_version == "text-v3"
        assert vector.features["event_surprise"] == pytest.approx(0.9)
        assert vector.features["guidance_specificity"] == pytest.approx(0.75)
        assert vector.features["risk_pressure"] == pytest.approx(0.2)
        assert vector.features["revision_clarity"] == pytest.approx(0.8)

    def test_extract_v4_returns_primary_sec_fields(self) -> None:
        extractor = self._extractor(prompt_version="v4")
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V4)
        extractor._client = mock_client

        vector = extractor.extract(event, "primary 10-Q filing", _RUN_ID)

        assert vector.feature_set_version == "text-v4"
        assert vector.features["operating_quality"] == pytest.approx(0.55)
        assert vector.features["demand_outlook"] == pytest.approx(0.65)
        assert vector.features["margin_resilience"] == pytest.approx(-0.15)
        assert vector.features["disclosure_specificity"] == pytest.approx(0.85)

    def test_extract_v5_compacts_primary_sec_text_and_writes_lineage(self, tmp_path) -> None:
        extractor = self._extractor(prompt_version="v5", artifact_root=tmp_path)
        event = _make_event(instrument_id=uuid.uuid4(), event_type=TextEventType.SEC_FILING)
        event.metadata["form_type"] = "10-Q"
        event.metadata["is_primary_document"] = "true"

        raw_text = "\n".join(
            [
                "dei:DocumentType us-gaap:Assets xbrli:shares iso4217:USD",
                "Item 1. Financial Statements",
                "A" * 12_000,
                "Item 2. Management's Discussion and Analysis",
                "Results of operations showed stronger revenue and margin resilience. " * 200,
                "Liquidity and capital resources remained adequate. " * 150,
                "Item 3. Quantitative and Qualitative Disclosures About Market Risk",
                "Market risk disclosure was specific. " * 120,
                "Item 4. Controls and Procedures",
                "B" * 20_000,
            ]
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V5)
        extractor._client = mock_client

        vector = extractor.extract(event, raw_text, _RUN_ID)

        assert vector.feature_set_version == "text-v5"
        payload = mock_client.messages.create.call_args.kwargs
        submitted = payload["messages"][0]["content"]
        assert len(submitted) < len(raw_text)
        assert "Results of operations" in submitted
        assert "omitted" in submitted
        artifact_path = tmp_path / "anthropic" / "claude-sonnet-4-6" / "v5"
        artifact = json.loads(next(artifact_path.glob("*.json")).read_text(encoding="utf-8"))
        assert artifact["lineage"]["policy_name"] == SEC_PRIMARY_COMPACTION_POLICY
        assert (
            artifact["lineage"]["raw_content_digest"]
            != artifact["lineage"]["compacted_content_digest"]
        )
        assert "item_2_mda" in artifact["lineage"]["selected_section_labels"]

    def test_v5_disclosure_specificity_above_one_raises(self) -> None:
        extractor = self._extractor(prompt_version="v5")
        event = _make_event(instrument_id=uuid.uuid4(), event_type=TextEventType.SEC_FILING)

        bad_payload = dict(_VALID_PAYLOAD_V5)
        bad_payload["disclosure_specificity"] = 1.1

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(bad_payload)
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="outside allowed range"):
            extractor.extract(event, "Item 2. Management discussion", _RUN_ID)

    def test_extraction_artifact_resume_skips_api_call(self, tmp_path) -> None:
        event = _make_event(instrument_id=uuid.uuid4())
        priming = self._extractor(prompt_version="v2", artifact_root=tmp_path)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V2)
        priming._client = mock_client
        priming.extract(event, "earnings release exhibit", _RUN_ID)

        resumed = self._extractor(prompt_version="v2", artifact_root=tmp_path)
        resumed._client = MagicMock()
        vector = resumed.extract(event, "earnings release exhibit", _RUN_ID)

        assert vector.feature_set_version == "text-v2"
        assert vector.features["catalyst_sentiment"] == pytest.approx(0.8)
        resumed._client.messages.create.assert_not_called()

    def test_v3_extraction_artifact_resume_skips_api_call(self, tmp_path) -> None:
        event = _make_event(instrument_id=uuid.uuid4())
        priming = self._extractor(prompt_version="v3", artifact_root=tmp_path)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V3)
        priming._client = mock_client
        priming.extract(event, "earnings release exhibit", _RUN_ID)

        resumed = self._extractor(prompt_version="v3", artifact_root=tmp_path)
        resumed._client = MagicMock()
        vector = resumed.extract(event, "earnings release exhibit", _RUN_ID)

        assert vector.feature_set_version == "text-v3"
        assert vector.features["revision_clarity"] == pytest.approx(0.8)
        resumed._client.messages.create.assert_not_called()

    def test_v4_extraction_artifact_resume_skips_api_call(self, tmp_path) -> None:
        event = _make_event(instrument_id=uuid.uuid4())
        priming = self._extractor(prompt_version="v4", artifact_root=tmp_path)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V4)
        priming._client = mock_client
        priming.extract(event, "primary 10-Q filing", _RUN_ID)

        resumed = self._extractor(prompt_version="v4", artifact_root=tmp_path)
        resumed._client = MagicMock()
        vector = resumed.extract(event, "primary 10-Q filing", _RUN_ID)

        assert vector.feature_set_version == "text-v4"
        assert vector.features["disclosure_specificity"] == pytest.approx(0.85)
        resumed._client.messages.create.assert_not_called()

    def test_v5_extraction_artifact_resume_uses_compacted_digest(self, tmp_path) -> None:
        event = _make_event(instrument_id=uuid.uuid4(), event_type=TextEventType.SEC_FILING)
        event.metadata["form_type"] = "10-K"
        raw_text = "Item 7. Management Discussion\n" + ("Operating quality improved. " * 400)
        priming = self._extractor(prompt_version="v5", artifact_root=tmp_path)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD_V5)
        priming._client = mock_client
        priming.extract(event, raw_text, _RUN_ID)

        resumed = self._extractor(prompt_version="v5", artifact_root=tmp_path)
        resumed._client = MagicMock()
        vector = resumed.extract(event, raw_text, _RUN_ID)

        assert vector.feature_set_version == "text-v5"
        assert vector.features["operating_quality"] == pytest.approx(0.55)
        resumed._client.messages.create.assert_not_called()

    def test_cache_hit_does_not_call_api(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        v1 = extractor.extract(event, "text", _RUN_ID)
        v2 = extractor.extract(event, "text", _RUN_ID)

        assert v1 is v2
        assert mock_client.messages.create.call_count == 1

    def test_cache_miss_for_different_event_id(self) -> None:
        extractor = self._extractor()
        event_a = _make_event(instrument_id=uuid.uuid4())
        event_b = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        extractor.extract(event_a, "text", _RUN_ID)
        extractor.extract(event_b, "text", _RUN_ID)

        assert mock_client.messages.create.call_count == 2

    def test_cache_miss_for_changed_text_content(self) -> None:
        # Regression: cache key omitted text_content, so a corrected
        # transcript on the same event_id silently returned the stale
        # vector. Cache key now includes a content digest.
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        extractor.extract(event, "original transcript", _RUN_ID)
        extractor.extract(event, "corrected transcript", _RUN_ID)

        assert mock_client.messages.create.call_count == 2

    def test_api_exception_raises_feature_extraction_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())
        monkeypatch.setattr(
            "quant_platform.services.research_service.text.features.time.sleep",
            lambda _seconds: None,
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TimeoutError("connection timed out")
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="Anthropic API call failed"):
            extractor.extract(event, "text", _RUN_ID)

    def test_provider_latency_budget_breach_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        extractor = self._extractor(max_request_latency_seconds=0.05)
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client
        ticks = iter([10.0, 10.25])
        monkeypatch.setattr(
            "quant_platform.services.research_service.text.features.time.monotonic",
            lambda: next(ticks),
        )

        with pytest.raises(TextFeatureLatencyError, match="latency budget breached"):
            extractor.extract(event, "text", _RUN_ID)

        mock_client.messages.create.assert_called_once()

    def test_daily_call_budget_blocks_provider_call(self) -> None:
        extractor = self._extractor(max_daily_calls=0)
        event = _make_event(instrument_id=uuid.uuid4())
        mock_client = MagicMock()
        extractor._client = mock_client

        with pytest.raises(TextFeatureBudgetError, match="daily call budget exceeded"):
            extractor.extract(event, "text", _RUN_ID)

        mock_client.messages.create.assert_not_called()

    def test_estimated_cost_budget_blocks_provider_call(self) -> None:
        extractor = self._extractor(
            max_daily_estimated_cost_usd=0.005,
            estimated_cost_per_call_usd=0.01,
        )
        event = _make_event(instrument_id=uuid.uuid4())
        mock_client = MagicMock()
        extractor._client = mock_client

        with pytest.raises(TextFeatureBudgetError, match="estimated cost budget exceeded"):
            extractor.extract(event, "text", _RUN_ID)

        mock_client.messages.create.assert_not_called()

    def test_non_json_response_raises_feature_extraction_error(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="not json at all")]
        )
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="not valid JSON"):
            extractor.extract(event, "text", _RUN_ID)
        assert mock_client.messages.create.call_count == 1

    def test_retryable_provider_failure_retries_to_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())
        monkeypatch.setattr(
            "quant_platform.services.research_service.text.features.time.sleep",
            lambda _seconds: None,
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            TimeoutError("temporary timeout"),
            _mock_api_response(_VALID_PAYLOAD),
        ]
        extractor._client = mock_client

        vector = extractor.extract(event, "text", _RUN_ID)

        assert vector.features["text_sentiment"] == pytest.approx(0.7)
        assert mock_client.messages.create.call_count == 2

    def test_missing_feature_key_raises_feature_extraction_error(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        incomplete = dict(_VALID_PAYLOAD)
        del incomplete["macro_sentiment"]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(incomplete)
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="Missing required feature key"):
            extractor.extract(event, "text", _RUN_ID)

    def test_out_of_range_value_raises_feature_extraction_error(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        bad_payload = dict(_VALID_PAYLOAD)
        bad_payload["text_sentiment"] = 2.5  # outside [-1, 1]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(bad_payload)
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="outside allowed range"):
            extractor.extract(event, "text", _RUN_ID)

    def test_revenue_revision_magnitude_below_zero_raises(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        bad_payload = dict(_VALID_PAYLOAD)
        bad_payload["revenue_revision_magnitude"] = -0.1  # outside [0, 1]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(bad_payload)
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="outside allowed range"):
            extractor.extract(event, "text", _RUN_ID)

    def test_v3_risk_pressure_above_one_raises(self) -> None:
        extractor = self._extractor(prompt_version="v3")
        event = _make_event(instrument_id=uuid.uuid4())

        bad_payload = dict(_VALID_PAYLOAD_V3)
        bad_payload["risk_pressure"] = 1.1

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(bad_payload)
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="outside allowed range"):
            extractor.extract(event, "text", _RUN_ID)

    def test_v4_disclosure_specificity_above_one_raises(self) -> None:
        extractor = self._extractor(prompt_version="v4")
        event = _make_event(instrument_id=uuid.uuid4())

        bad_payload = dict(_VALID_PAYLOAD_V4)
        bad_payload["disclosure_specificity"] = 1.1

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(bad_payload)
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="outside allowed range"):
            extractor.extract(event, "text", _RUN_ID)

    def test_unknown_prompt_version_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown prompt_version"):
            LLMTextFeatureExtractor(prompt_version="v999")

    def test_missing_anthropic_package_raises_feature_extraction_error(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        with patch.dict("sys.modules", {"anthropic": None}):
            extractor._client = None
            with pytest.raises((FeatureExtractionError, ImportError)):
                extractor._get_client()

    def test_deepseek_provider_uses_anthropic_compatible_messages_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extractor = self._extractor(
            provider="deepseek",
            model="deepseek-v4-flash",
            deepseek_base_url="https://api.deepseek.com/anthropic",
        )
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        vector = extractor.extract(event, "Q3 results beat estimates.", _RUN_ID)

        assert vector.features["text_sentiment"] == pytest.approx(0.7)
        mock_client.messages.create.assert_called_once()
        payload = mock_client.messages.create.call_args.kwargs
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["system"]
        assert payload["messages"][0]["role"] == "user"

    def test_deepseek_provider_configures_anthropic_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeAnthropic:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        extractor = self._extractor(
            provider="deepseek",
            deepseek_base_url="https://api.deepseek.com/anthropic",
        )

        client = extractor._get_client()

        assert isinstance(client, FakeAnthropic)
        assert calls == [
            {
                "api_key": "sk-test",
                "base_url": "https://api.deepseek.com/anthropic",
                "timeout": 30.0,
            }
        ]

    def test_deepseek_provider_reads_api_key_from_dotenv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeAnthropic:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        import anthropic

        monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-dotenv\n", encoding="utf-8")
        extractor = self._extractor(
            provider="deepseek",
            deepseek_base_url="https://api.deepseek.com/anthropic",
        )

        client = extractor._get_client()

        assert isinstance(client, FakeAnthropic)
        assert calls == [
            {
                "api_key": "sk-dotenv",
                "base_url": "https://api.deepseek.com/anthropic",
                "timeout": 30.0,
            }
        ]

    def test_deepseek_provider_requires_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        extractor = self._extractor(provider="deepseek")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.chdir(tmp_path)
        extractor._client = None

        with pytest.raises(FeatureExtractionError, match="DEEPSEEK_API_KEY"):
            extractor._get_client()

    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            LLMTextFeatureExtractor(provider="other")  # type: ignore[arg-type]

    def test_defaults_as_of_to_event_occurred_at(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_api_response(_VALID_PAYLOAD)
        extractor._client = mock_client

        vector = extractor.extract(event, "text", _RUN_ID)
        assert vector.as_of == event.occurred_at

    def test_empty_api_content_raises_feature_extraction_error(self) -> None:
        extractor = self._extractor()
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = SimpleNamespace(content=[])
        extractor._client = mock_client

        with pytest.raises(FeatureExtractionError, match="empty content"):
            extractor.extract(event, "text", _RUN_ID)

    def test_anthropic_compatible_response_skips_non_text_blocks(self) -> None:
        extractor = self._extractor(provider="deepseek")
        event = _make_event(instrument_id=uuid.uuid4())

        mock_client = MagicMock()
        mock_client.messages.create.return_value = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="intermediate reasoning"),
                SimpleNamespace(type="text", text=json.dumps(_VALID_PAYLOAD)),
            ]
        )
        extractor._client = mock_client

        vector = extractor.extract(event, "text", _RUN_ID)

        assert vector.features["text_sentiment"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# InMemoryTextEventStore
# ---------------------------------------------------------------------------


class TestInMemoryTextEventStore:
    pytestmark = pytest.mark.asyncio

    async def test_store_and_retrieve_event(self) -> None:
        store = InMemoryTextEventStore()
        iid = uuid.uuid4()
        event = _make_event(instrument_id=iid)
        await store.store_event(event)

        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
        )
        assert len(results) == 1
        assert results[0].event_id == event.event_id

    async def test_store_event_is_idempotent(self) -> None:
        store = InMemoryTextEventStore()
        event = _make_event(instrument_id=uuid.uuid4())
        await store.store_event(event)
        await store.store_event(event)  # duplicate
        await store.store_event(event)

        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
        )
        assert len(results) == 1

    async def test_get_events_filters_by_time_window(self) -> None:
        store = InMemoryTextEventStore()
        early = TextEvent(
            event_id=uuid.uuid4(),
            event_type=TextEventType.NEWS_HEADLINE,
            occurred_at=datetime(2025, 9, 14, 10, 0, tzinfo=UTC),
            source_uri="s3://b/a.txt",
            artifact_uri="s3://b/a.txt",
        )
        late = TextEvent(
            event_id=uuid.uuid4(),
            event_type=TextEventType.NEWS_HEADLINE,
            occurred_at=datetime(2025, 9, 15, 10, 0, tzinfo=UTC),
            source_uri="s3://b/b.txt",
            artifact_uri="s3://b/b.txt",
        )
        await store.store_event(early)
        await store.store_event(late)

        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
        )
        assert len(results) == 1
        assert results[0].event_id == late.event_id

    async def test_get_events_filters_by_instrument_id(self) -> None:
        store = InMemoryTextEventStore()
        iid_a = uuid.uuid4()
        iid_b = uuid.uuid4()
        ev_a = _make_event(instrument_id=iid_a)
        ev_b = _make_event(instrument_id=iid_b)
        await store.store_event(ev_a)
        await store.store_event(ev_b)

        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
            instrument_ids=[iid_a],
        )
        assert len(results) == 1
        assert results[0].instrument_id == iid_a

    async def test_get_events_filters_by_event_type(self) -> None:
        store = InMemoryTextEventStore()
        ev_earning = _make_event(event_type=TextEventType.EARNINGS_TRANSCRIPT)
        ev_news = _make_event(event_type=TextEventType.NEWS_HEADLINE)
        await store.store_event(ev_earning)
        await store.store_event(ev_news)

        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
            event_types=[TextEventType.EARNINGS_TRANSCRIPT],
        )
        assert len(results) == 1
        assert results[0].event_type == TextEventType.EARNINGS_TRANSCRIPT

    async def test_get_events_sorted_by_occurred_at(self) -> None:
        store = InMemoryTextEventStore()
        for hour in (12, 8, 16):
            await store.store_event(
                TextEvent(
                    event_id=uuid.uuid4(),
                    event_type=TextEventType.NEWS_HEADLINE,
                    occurred_at=datetime(2025, 9, 15, hour, 0, tzinfo=UTC),
                    source_uri="s3://b/x.txt",
                    artifact_uri="s3://b/x.txt",
                )
            )

        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
        )
        assert [r.occurred_at.hour for r in results] == [8, 12, 16]

    async def test_get_events_raises_on_naive_start(self) -> None:
        store = InMemoryTextEventStore()
        with pytest.raises(ValueError, match="UTC-aware"):
            await store.get_events(
                start=datetime(2025, 9, 15),
                end=datetime(2025, 9, 16, tzinfo=UTC),
            )

    async def test_get_events_raises_when_start_equals_end(self) -> None:
        store = InMemoryTextEventStore()
        t = datetime(2025, 9, 15, 12, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="before end"):
            await store.get_events(start=t, end=t)

    async def test_end_boundary_is_exclusive(self) -> None:
        store = InMemoryTextEventStore()
        t = datetime(2025, 9, 15, 12, 0, tzinfo=UTC)
        await store.store_event(
            TextEvent(
                event_id=uuid.uuid4(),
                event_type=TextEventType.NEWS_HEADLINE,
                occurred_at=t,
                source_uri="s3://b/x.txt",
                artifact_uri="s3://b/x.txt",
            )
        )
        # end == occurred_at: event should NOT appear.
        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=t,
        )
        assert results == []


class _RecordingExtractor:
    def __init__(self) -> None:
        self.event_ids: list[uuid.UUID] = []

    def extract(
        self,
        event: TextEvent,
        text_content: str,
        strategy_run_id: uuid.UUID,
        *,
        as_of: datetime | None = None,
    ) -> FeatureVector:
        del text_content
        self.event_ids.append(event.event_id)
        vector_as_of = as_of if as_of is not None else event.occurred_at
        return FeatureVector(
            vector_id=uuid.uuid4(),
            instrument_id=event.instrument_id or uuid.UUID(int=0),
            strategy_run_id=strategy_run_id,
            as_of=vector_as_of,
            features={"text_sentiment": 0.5},
            feature_set_version="text-v2",
            artifact_uri=event.artifact_uri,
            available_at=vector_as_of,
        )


class TestTextEventExtraction:
    pytestmark = pytest.mark.asyncio

    async def test_manifest_scoped_extraction_ignores_other_durable_events(
        self,
        tmp_path,
    ) -> None:
        store = InMemoryTextEventStore()
        repo = InMemoryFeatureRepository()
        target_artifact = tmp_path / "target.txt"
        other_artifact = tmp_path / "other.txt"
        target_artifact.write_text("target exhibit", encoding="utf-8")
        other_artifact.write_text("other exhibit", encoding="utf-8")
        instrument_id = uuid.uuid4()
        target_event = TextEvent(
            event_id=uuid.uuid4(),
            event_type=TextEventType.SEC_FILING,
            occurred_at=_NOW,
            source_uri="https://sec.test/target",
            artifact_uri=str(target_artifact),
            instrument_id=instrument_id,
            metadata={"is_primary_document": "false", "symbol": "AAPL"},
        )
        other_event = TextEvent(
            event_id=uuid.uuid4(),
            event_type=TextEventType.SEC_FILING,
            occurred_at=_NOW,
            source_uri="https://sec.test/other",
            artifact_uri=str(other_artifact),
            instrument_id=uuid.uuid4(),
            metadata={"is_primary_document": "false", "symbol": "MSFT"},
        )
        await store.store_event(target_event)
        await store.store_event(other_event)
        extractor = _RecordingExtractor()

        result = await extract_text_event_features(
            text_event_store=store,
            feature_repo=repo,
            extractor=extractor,  # type: ignore[arg-type]
            strategy_run_id=_RUN_ID,
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
            document_role="exhibit",
            source_targets=(
                TextEventExtractionTarget(
                    event_id=target_event.event_id,
                    symbol="AAPL",
                    instrument_id=instrument_id,
                    is_primary_document=False,
                ),
            ),
        )

        assert result.passed
        assert result.events_seen == 1
        assert result.vectors_stored == 1
        assert extractor.event_ids == [target_event.event_id]

    async def test_manifest_scoped_extraction_fails_for_missing_durable_event(self) -> None:
        store = InMemoryTextEventStore()
        repo = InMemoryFeatureRepository()
        missing_id = uuid.uuid4()

        result = await extract_text_event_features(
            text_event_store=store,
            feature_repo=repo,
            extractor=_RecordingExtractor(),  # type: ignore[arg-type]
            strategy_run_id=_RUN_ID,
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
            document_role="exhibit",
            source_targets=(TextEventExtractionTarget(event_id=missing_id, symbol="AAPL"),),
        )

        assert not result.passed
        assert result.events_seen == 1
        assert result.failed_events == 1
        assert result.failed_event_details[0]["event_id"] == str(missing_id)
        assert result.failed_event_details[0]["error_class"] == "MissingDurableTextEvent"


class _FakeMappings:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, object]]:
        return self._rows


class _FakeResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self._rows)


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.execute_calls: list[tuple[str, dict[str, object]]] = []

    async def execute(
        self,
        statement: object,
        params: dict[str, object] | None = None,
    ) -> _FakeResult:
        self.execute_calls.append((str(statement), params or {}))
        return _FakeResult(self.rows)


class _FakeAsyncContext:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, *_exc_info: Any) -> bool:
        return False


class _FakeEngine:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def begin(self) -> _FakeAsyncContext:
        return _FakeAsyncContext(self._conn)

    def connect(self) -> _FakeAsyncContext:
        return _FakeAsyncContext(self._conn)


class TestPostgresTextEventStore:
    pytestmark = pytest.mark.asyncio

    async def test_store_event_uses_async_engine_and_jsonb_cast(self) -> None:
        conn = _FakeConnection()
        store = PostgresTextEventStore(_FakeEngine(conn))  # type: ignore[arg-type]
        event = _make_event(instrument_id=uuid.uuid4())

        await store.store_event(event)

        sql, params = conn.execute_calls[0]
        assert "INSERT INTO text_events" in sql
        assert "CAST(:metadata AS JSONB)" in sql
        assert params["id"] == event.event_id
        assert params["instrument_id"] == event.instrument_id
        assert json.loads(params["metadata"]) == {}

    async def test_get_events_builds_filtered_query_and_maps_rows(self) -> None:
        iid = uuid.uuid4()
        event_id = uuid.uuid4()
        conn = _FakeConnection(
            rows=[
                {
                    "id": event_id,
                    "instrument_id": iid,
                    "event_type": TextEventType.NEWS_HEADLINE.value,
                    "occurred_at": _NOW,
                    "source_uri": "https://example.test/news",
                    "artifact_uri": "/tmp/news.txt",
                    "metadata": {"ticker": "AAPL"},
                }
            ]
        )
        store = PostgresTextEventStore(_FakeEngine(conn))  # type: ignore[arg-type]

        results = await store.get_events(
            start=datetime(2025, 9, 15, 0, 0, tzinfo=UTC),
            end=datetime(2025, 9, 16, 0, 0, tzinfo=UTC),
            instrument_ids=[iid],
            event_types=[TextEventType.NEWS_HEADLINE],
        )

        sql, params = conn.execute_calls[0]
        assert "instrument_id = ANY(:instrument_ids)" in sql
        assert "event_type = ANY(:event_types)" in sql
        assert params["instrument_ids"] == [iid]
        assert params["event_types"] == [TextEventType.NEWS_HEADLINE.value]
        assert len(results) == 1
        assert results[0].event_id == event_id
        assert results[0].instrument_id == iid
        assert results[0].metadata["ticker"] == "AAPL"
