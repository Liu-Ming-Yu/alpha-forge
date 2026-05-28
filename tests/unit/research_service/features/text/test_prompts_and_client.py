"""Unit tests for prompt templates + LLM client adapters."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_platform.research.features.text.client import (
    LLMResponse,
    MockLLMClient,
    hash_raw_response,
)
from quant_platform.research.features.text.prompts import (
    NEWS_PROMPT_VERSION,
    PromptTemplate,
    get_news_prompt,
    render_user_message,
    serialise_output_schema,
)
from quant_platform.research.features.text.schemas import (
    NewsExtraction,
    SourceDocument,
)

# ---------------------------------------------------------------------------
# PromptTemplate validation
# ---------------------------------------------------------------------------


def test_prompt_rejects_template_without_instrument_placeholder() -> None:
    with pytest.raises(ValueError, match="instrument_id"):
        PromptTemplate(
            version="bogus",
            system="sys",
            user_template="Extract from: {text}",
            output_schema={"type": "object"},
        )


def test_prompt_rejects_template_without_text_placeholder() -> None:
    with pytest.raises(ValueError, match="text"):
        PromptTemplate(
            version="bogus",
            system="sys",
            user_template="Extract for {instrument_id}.",
            output_schema={"type": "object"},
        )


def test_prompt_rejects_empty_version() -> None:
    with pytest.raises(ValueError, match="version"):
        PromptTemplate(
            version="   ",
            system="sys",
            user_template="for {instrument_id}: {text}",
            output_schema={},
        )


def test_news_prompt_carries_version_pin() -> None:
    prompt = get_news_prompt()
    assert prompt.version == NEWS_PROMPT_VERSION
    assert prompt.version == "news-prompt-v1"


def test_news_prompt_schema_lists_every_extraction_field() -> None:
    prompt = get_news_prompt()
    properties = prompt.output_schema["properties"]
    expected = set(NewsExtraction.SIGNED_FIELDS) | set(NewsExtraction.UNSIGNED_FIELDS)
    assert set(properties) == expected  # type: ignore[arg-type]


def test_news_prompt_schema_enforces_value_ranges() -> None:
    prompt = get_news_prompt()
    props = prompt.output_schema["properties"]
    # Signed fields: min=-1, max=1.
    for name in NewsExtraction.SIGNED_FIELDS:
        assert props[name]["minimum"] == -1.0  # type: ignore[index]
        assert props[name]["maximum"] == 1.0  # type: ignore[index]
    # Unsigned fields: min=0, max=1.
    for name in NewsExtraction.UNSIGNED_FIELDS:
        assert props[name]["minimum"] == 0.0  # type: ignore[index]
        assert props[name]["maximum"] == 1.0  # type: ignore[index]


def test_render_user_message_fills_placeholders() -> None:
    prompt = get_news_prompt()
    doc = SourceDocument(
        doc_id="d1",
        instrument_id="AAPL",
        published_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
        kind="news",
        text="Apple announced a new product today.",
    )
    rendered = render_user_message(prompt, doc)
    assert "AAPL" in rendered
    assert "Apple announced a new product today." in rendered


def test_serialise_output_schema_is_deterministic() -> None:
    prompt = get_news_prompt()
    a = serialise_output_schema(prompt)
    b = serialise_output_schema(prompt)
    assert a == b
    # Keys must be sorted.
    assert a.startswith('{"')


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------


def _doc(doc_id: str = "d1", instrument_id: str = "AAPL") -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        instrument_id=instrument_id,
        published_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
        kind="news",
        text=f"News about {instrument_id}.",
    )


def test_mock_client_returns_responder_payload() -> None:
    payload = {
        "sentiment": 0.4,
        "materiality": 0.7,
        "demand_signal": 0.0,
        "margin_pressure": 0.0,
        "guidance_signal": 0.0,
        "litigation_risk": 0.0,
        "novelty": 0.5,
    }
    client = MockLLMClient(responder=lambda _p, _d: payload)
    response = client.extract(get_news_prompt(), _doc())
    assert isinstance(response, LLMResponse)
    assert response.payload == payload
    assert response.model_version == "mock-llm-v1"
    assert response.confidence == 1.0


def test_mock_client_returns_stable_raw_text() -> None:
    payload = {"sentiment": 0.5, "materiality": 0.5, "novelty": 0.5}
    client = MockLLMClient(responder=lambda _p, _d: payload)
    a = client.extract(get_news_prompt(), _doc())
    b = client.extract(get_news_prompt(), _doc())
    # Same payload → same raw text (sort_keys=True in the mock).
    assert a.raw_text == b.raw_text


def test_mock_client_custom_model_label() -> None:
    client = MockLLMClient(
        responder=lambda _p, _d: {"sentiment": 0.0, "materiality": 0.5, "novelty": 0.5},
        model_label="fake-gpt-99",
    )
    assert client.model_version == "fake-gpt-99"


# ---------------------------------------------------------------------------
# hash_raw_response
# ---------------------------------------------------------------------------


def test_hash_raw_response_is_deterministic() -> None:
    assert hash_raw_response("hello") == hash_raw_response("hello")


def test_hash_raw_response_changes_with_input() -> None:
    assert hash_raw_response("hello") != hash_raw_response("world")


def test_hash_raw_response_is_short_hex() -> None:
    h = hash_raw_response("hello")
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
