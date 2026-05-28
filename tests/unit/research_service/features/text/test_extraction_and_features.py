"""End-to-end tests: extraction pipeline + storage + features."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from quant_platform.research.features.text.aggregator import build_text_panel
from quant_platform.research.features.text.client import MockLLMClient
from quant_platform.research.features.text.extraction import (
    ExtractionConfig,
    LLMTransientError,
    extract_documents,
)
from quant_platform.research.features.text.features import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    compute_text_features,
)
from quant_platform.research.features.text.prompts import get_news_prompt
from quant_platform.research.features.text.schemas import (
    SourceDocument,
)
from quant_platform.research.features.text.storage import (
    append_extracted_records,
    load_extracted_records,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(
    doc_id: str,
    instrument_id: str,
    *,
    day_offset: int = 0,
    text: str = "...",
) -> SourceDocument:
    published = datetime(2024, 1, 2 + day_offset, 15, tzinfo=UTC)
    return SourceDocument(
        doc_id=doc_id,
        instrument_id=instrument_id,
        published_at=published,
        kind="news",
        text=text,
    )


def _payload(sentiment: float = 0.5, materiality: float = 0.8) -> dict:
    return {
        "sentiment": sentiment,
        "materiality": materiality,
        "demand_signal": 0.0,
        "margin_pressure": 0.0,
        "guidance_signal": 0.0,
        "litigation_risk": 0.0,
        "novelty": 0.5,
    }


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------


def test_extract_documents_succeeds_with_valid_payload() -> None:
    docs = [_doc("d1", "AAPL"), _doc("d2", "MSFT")]
    client = MockLLMClient(responder=lambda _p, _d: _payload())
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    assert len(records) == 2
    assert all(r.succeeded for r in records)
    assert records[0].instrument_id == "AAPL"
    assert records[1].instrument_id == "MSFT"
    # Provenance is populated.
    p = records[0].provenance
    assert p is not None
    assert p.prompt_version == "news-prompt-v1"
    assert p.model_version == "mock-llm-v1"
    assert p.source_doc_id == "d1"
    assert 0 <= p.confidence <= 1


def test_extract_documents_handles_out_of_range_score() -> None:
    docs = [_doc("d1", "AAPL")]
    # Mock returns sentiment outside [-1, 1] — schema validation fails.
    client = MockLLMClient(
        responder=lambda _p, _d: {**_payload(), "sentiment": 2.0},
    )
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    assert len(records) == 1
    assert not records[0].succeeded
    assert records[0].failure is not None
    assert records[0].failure.reason == "score_out_of_range"


def test_extract_documents_handles_missing_required_field() -> None:
    docs = [_doc("d1", "AAPL")]
    # Missing required ``sentiment`` field — no default on the
    # dataclass, so construction raises TypeError and the pipeline
    # records a malformed_payload failure.
    client = MockLLMClient(
        responder=lambda _p, _d: {"materiality": 0.5, "novelty": 0.5},
    )
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    assert not records[0].succeeded
    assert records[0].failure is not None
    assert records[0].failure.reason == "malformed_payload"


def test_extract_documents_fills_defaults_for_optional_fields() -> None:
    """``NewsExtraction.from_dict`` falls back to dataclass defaults
    for absent *optional* fields (those whose schema sets a default);
    only missing *required* fields fail the parse."""
    docs = [_doc("d1", "AAPL")]
    # ``materiality`` is required; everything else has a default.
    client = MockLLMClient(
        responder=lambda _p, _d: {"sentiment": 0.4, "materiality": 0.7},
    )
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    assert records[0].succeeded
    assert records[0].extraction is not None
    # Optional fields get their dataclass defaults.
    assert records[0].extraction.demand_signal == 0.0
    assert records[0].extraction.novelty == 0.5


def test_extract_documents_preserves_input_order() -> None:
    docs = [_doc(f"d{i}", f"I{i}") for i in range(5)]
    client = MockLLMClient(responder=lambda _p, _d: _payload())
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    for i, record in enumerate(records):
        assert record.instrument_id == f"I{i}"


def test_extract_documents_retries_transient_errors() -> None:
    docs = [_doc("d1", "AAPL")]

    attempts = {"count": 0}

    def flaky_responder(prompt, doc):
        del prompt, doc
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise LLMTransientError("retry me")
        return _payload()

    client = MockLLMClient(responder=flaky_responder)
    records = extract_documents(
        client=client,
        prompt=get_news_prompt(),
        documents=docs,
        config=ExtractionConfig(max_attempts=3, initial_delay_seconds=0.0),
        sleep=lambda _delay: None,
    )
    assert records[0].succeeded
    assert attempts["count"] == 3


def test_extract_documents_gives_up_after_retry_budget() -> None:
    docs = [_doc("d1", "AAPL")]

    def always_transient(prompt, doc):
        del prompt, doc
        raise LLMTransientError("dead")

    client = MockLLMClient(responder=always_transient)
    records = extract_documents(
        client=client,
        prompt=get_news_prompt(),
        documents=docs,
        config=ExtractionConfig(max_attempts=2, initial_delay_seconds=0.0),
        sleep=lambda _delay: None,
    )
    assert not records[0].succeeded
    assert records[0].failure is not None
    assert records[0].failure.reason == "client_error"


def test_extract_documents_does_not_retry_semantic_errors() -> None:
    docs = [_doc("d1", "AAPL")]

    attempts = {"count": 0}

    def out_of_range(prompt, doc):
        del prompt, doc
        attempts["count"] += 1
        return {**_payload(), "sentiment": 99.0}

    client = MockLLMClient(responder=out_of_range)
    records = extract_documents(
        client=client,
        prompt=get_news_prompt(),
        documents=docs,
        config=ExtractionConfig(max_attempts=5, initial_delay_seconds=0.0),
        sleep=lambda _delay: None,
    )
    assert not records[0].succeeded
    # Schema failures do NOT retry — exactly one call.
    assert attempts["count"] == 1


def test_extraction_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ExtractionConfig(max_attempts=0)
    with pytest.raises(ValueError, match="backoff_factor"):
        ExtractionConfig(backoff_factor=0.5)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_storage_round_trips_extracted_records(tmp_path: Path) -> None:
    target = tmp_path / "extractions.jsonl"
    docs = [_doc("d1", "AAPL"), _doc("d2", "MSFT")]
    client = MockLLMClient(responder=lambda _p, _d: _payload())
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    resolved, count = append_extracted_records(records, path=target)
    assert resolved == target
    assert count == 2
    loaded = load_extracted_records(path=target)
    assert len(loaded) == 2
    assert {r.instrument_id for r in loaded} == {"AAPL", "MSFT"}


def test_storage_loads_failure_records(tmp_path: Path) -> None:
    target = tmp_path / "extractions.jsonl"
    docs = [_doc("d1", "AAPL")]
    bad_client = MockLLMClient(responder=lambda _p, _d: {"sentiment": 99.0})
    records = extract_documents(client=bad_client, prompt=get_news_prompt(), documents=docs)
    append_extracted_records(records, path=target)
    loaded = load_extracted_records(path=target)
    assert len(loaded) == 1
    assert not loaded[0].succeeded
    assert loaded[0].failure is not None


def test_storage_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_extracted_records(path=tmp_path / "missing.jsonl") == ()


def test_storage_skips_malformed_lines_with_warning(tmp_path: Path) -> None:
    target = tmp_path / "extractions.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    good_docs = [_doc("d1", "AAPL")]
    client = MockLLMClient(responder=lambda _p, _d: _payload())
    good_records = extract_documents(client=client, prompt=get_news_prompt(), documents=good_docs)
    with target.open("w", encoding="utf-8") as fh:
        fh.write("definitely not JSON\n")
        fh.write(good_records[0].to_jsonl_line() + "\n")
        fh.write('{"missing": "fields"}\n')
    with pytest.warns(UserWarning):
        loaded = load_extracted_records(path=target)
    assert len(loaded) == 1
    assert loaded[0].instrument_id == "AAPL"


def test_storage_strict_raises_on_first_bad_line(tmp_path: Path) -> None:
    target = tmp_path / "extractions.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("not JSON\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_extracted_records(path=target, strict=True)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_aggregator_groups_by_instrument_and_date() -> None:
    docs = [
        _doc("d1", "AAPL", day_offset=0),
        _doc("d2", "AAPL", day_offset=0),  # same day
        _doc("d3", "AAPL", day_offset=1),  # next day
        _doc("d4", "MSFT", day_offset=0),
    ]
    client = MockLLMClient(responder=lambda _p, _d: _payload(sentiment=0.5))
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    panel = build_text_panel(records=records, documents=docs)
    # 3 unique (instrument, date) cells: AAPL/d0, AAPL/d1, MSFT/d0.
    assert len(panel.frame) == 3
    aapl_d0 = panel.frame[
        (panel.frame["instrument_id"] == "AAPL")
        & (panel.frame["date"] == pd.Timestamp("2024-01-02"))
    ].iloc[0]
    assert aapl_d0["count"] == 2  # two articles aggregated
    # Materiality-weighted mean: both at sentiment 0.5, materiality 0.8 → mean = 0.5.
    assert aapl_d0["sentiment_mean"] == pytest.approx(0.5)


def test_aggregator_counts_failures_separately() -> None:
    docs = [_doc("d1", "AAPL")]
    bad_client = MockLLMClient(responder=lambda _p, _d: {"sentiment": 99.0})
    records = extract_documents(client=bad_client, prompt=get_news_prompt(), documents=docs)
    panel = build_text_panel(records=records, documents=docs)
    assert panel.n_failures == 1
    # The failure produces a row but with count=0.
    row = panel.frame.iloc[0]
    assert row["count"] == 0
    assert row["failure_count"] == 1


def test_aggregator_positive_negative_counts() -> None:
    # 3 positive, 2 neutral, 1 negative.
    docs = [_doc(f"d{i}", "AAPL", day_offset=0) for i in range(6)]
    payloads = [
        _payload(sentiment=0.8),
        _payload(sentiment=0.5),
        _payload(sentiment=0.4),
        _payload(sentiment=0.0),
        _payload(sentiment=0.1),
        _payload(sentiment=-0.5),
    ]
    client = MockLLMClient(
        responder=lambda _p, doc: payloads[int(doc.doc_id.removeprefix("d"))],
    )
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    panel = build_text_panel(records=records, documents=docs)
    row = panel.frame.iloc[0]
    # Threshold 0.3 → 3 positive (0.8, 0.5, 0.4), 1 negative (-0.5).
    assert row["positive_count"] == 3
    assert row["negative_count"] == 1


# ---------------------------------------------------------------------------
# compute_text_features
# ---------------------------------------------------------------------------


def test_compute_text_features_shape() -> None:
    docs = [_doc(f"d{i}", "AAPL", day_offset=i) for i in range(3)]
    client = MockLLMClient(responder=lambda _p, _d: _payload())
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    ff = compute_text_features(records=records, documents=docs)
    assert set(ff.feature_names) == set(FEATURE_NAMES)
    assert len(ff.frame) == 3
    for spec in FEATURE_SPECS:
        assert ff.feature_specs[spec.name] == spec


def test_compute_text_features_empty_records() -> None:
    ff = compute_text_features(records=[], documents=[])
    assert ff.frame.empty
    assert all(v == 0 for v in ff.coverage.values())


def test_compute_text_features_sentiment_matches_aggregator() -> None:
    docs = [_doc("d1", "AAPL")]
    client = MockLLMClient(responder=lambda _p, _d: _payload(sentiment=0.6))
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    ff = compute_text_features(records=records, documents=docs)
    assert ff.frame.iloc[0]["news_sentiment_1d"] == pytest.approx(0.6)
    assert ff.frame.iloc[0]["news_volume_1d"] == 1.0
    assert ff.frame.iloc[0]["positive_news_shock"] == 1.0
    assert ff.frame.iloc[0]["negative_news_shock"] == 0.0


def test_compute_text_features_volume_zscore_requires_full_window() -> None:
    # Single-day data → not enough for a rolling z-score.
    docs = [_doc("d1", "AAPL")]
    client = MockLLMClient(responder=lambda _p, _d: _payload())
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    ff = compute_text_features(records=records, documents=docs)
    # Coverage zero for the rolling-z column (warm-up not satisfied).
    assert ff.coverage["news_volume_zscore_20d"] == 0


def test_feature_specs_registered_in_global_registry() -> None:
    from quant_platform.research.features import get_global_registry

    registry = get_global_registry()
    for spec in FEATURE_SPECS:
        assert registry.has(spec.name, spec.version)
        assert registry.get(spec.name, spec.version) == spec


def test_specs_are_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


# ---------------------------------------------------------------------------
# Integration: full extraction → storage → features
# ---------------------------------------------------------------------------


def test_full_pipeline_extraction_storage_features(tmp_path: Path) -> None:
    """Round-trip the entire pipeline: extract → persist → reload →
    compute features. Pins that the storage layer doesn't drop data
    the feature compute needs."""
    target = tmp_path / "extractions.jsonl"
    docs = [_doc(f"d{i}", "AAPL", day_offset=i) for i in range(3)] + [
        _doc(f"e{i}", "MSFT", day_offset=i) for i in range(3)
    ]
    client = MockLLMClient(
        responder=lambda _p, doc: _payload(
            sentiment=0.5 if doc.instrument_id == "AAPL" else -0.3,
        )
    )
    fresh_records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    append_extracted_records(fresh_records, path=target)
    reloaded = load_extracted_records(path=target)
    assert len(reloaded) == 6
    ff = compute_text_features(records=list(reloaded), documents=docs)
    # Two instruments × three dates = six rows.
    assert len(ff.frame) == 6
    aapl_rows = ff.frame[ff.frame["instrument_id"] == "AAPL"]
    msft_rows = ff.frame[ff.frame["instrument_id"] == "MSFT"]
    # ``pytest.approx`` doesn't compose with Series == ; iterate.
    for value in aapl_rows["news_sentiment_1d"]:
        assert value == pytest.approx(0.5)
    for value in msft_rows["news_sentiment_1d"]:
        assert value == pytest.approx(-0.3)
