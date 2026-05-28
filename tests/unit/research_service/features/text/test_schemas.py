"""Unit tests for the text-event extraction schemas."""

from __future__ import annotations

import json

import pytest

from quant_platform.research.features.text.schemas import (
    SCHEMA_VERSION,
    ExtractedRecord,
    ExtractionProvenance,
    FailedExtraction,
    NewsExtraction,
    SourceDocument,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# NewsExtraction
# ---------------------------------------------------------------------------


def test_news_extraction_defaults_are_neutral() -> None:
    e = NewsExtraction(sentiment=0.0, materiality=0.5)
    assert e.demand_signal == 0.0
    assert e.margin_pressure == 0.0
    assert e.guidance_signal == 0.0
    assert e.litigation_risk == 0.0
    assert e.novelty == 0.5


def test_news_extraction_rejects_out_of_range_sentiment() -> None:
    with pytest.raises(ValueError, match="sentiment must lie in"):
        NewsExtraction(sentiment=2.0, materiality=0.5)
    with pytest.raises(ValueError, match="sentiment must lie in"):
        NewsExtraction(sentiment=-1.1, materiality=0.5)


def test_news_extraction_rejects_out_of_range_unsigned() -> None:
    with pytest.raises(ValueError, match="materiality must lie in"):
        NewsExtraction(sentiment=0.0, materiality=1.5)
    with pytest.raises(ValueError, match="novelty must lie in"):
        NewsExtraction(sentiment=0.0, materiality=0.5, novelty=-0.1)


def test_news_extraction_round_trips_dict() -> None:
    original = NewsExtraction(
        sentiment=0.6,
        materiality=0.8,
        demand_signal=0.4,
        margin_pressure=-0.2,
        guidance_signal=0.1,
        litigation_risk=-0.3,
        novelty=0.9,
    )
    back = NewsExtraction.from_dict(original.to_dict())
    assert back == original


def test_news_extraction_field_descriptions_cover_all_fields() -> None:
    descriptions = NewsExtraction.field_descriptions()
    all_fields = (*NewsExtraction.SIGNED_FIELDS, *NewsExtraction.UNSIGNED_FIELDS)
    assert set(descriptions) == set(all_fields)


# ---------------------------------------------------------------------------
# ExtractionProvenance
# ---------------------------------------------------------------------------


def _sample_provenance() -> ExtractionProvenance:
    return ExtractionProvenance(
        prompt_version="news-prompt-v1",
        model_version="claude-sonnet-4-5",
        source_kind="news",
        source_doc_id="doc-001",
        extracted_at=utc_now_iso(),
        confidence=0.95,
        raw_response_hash="abcdef1234567890",  # pragma: allowlist secret
    )


def test_provenance_requires_non_empty_strings() -> None:
    with pytest.raises(ValueError, match="prompt_version"):
        ExtractionProvenance(
            prompt_version="   ",
            model_version="m",
            source_kind="news",
            source_doc_id="d",
            extracted_at="t",
            confidence=1.0,
            raw_response_hash="h",
        )
    with pytest.raises(ValueError, match="source_doc_id"):
        ExtractionProvenance(
            prompt_version="p",
            model_version="m",
            source_kind="news",
            source_doc_id="",
            extracted_at="t",
            confidence=1.0,
            raw_response_hash="h",
        )


def test_provenance_validates_confidence_range() -> None:
    with pytest.raises(ValueError, match="confidence must lie in"):
        ExtractionProvenance(
            prompt_version="p",
            model_version="m",
            source_kind="news",
            source_doc_id="d",
            extracted_at="t",
            confidence=1.5,
            raw_response_hash="h",
        )


def test_provenance_round_trips_dict() -> None:
    original = _sample_provenance()
    back = ExtractionProvenance.from_dict(original.to_dict())
    assert back == original


# ---------------------------------------------------------------------------
# FailedExtraction
# ---------------------------------------------------------------------------


def _sample_failure() -> FailedExtraction:
    return FailedExtraction(
        source_doc_id="doc-bad",
        source_kind="news",
        failed_at=utc_now_iso(),
        reason="malformed_json",
        detail="json.JSONDecodeError at line 1",
        prompt_version="news-prompt-v1",
        model_version="claude-sonnet-4-5",
    )


def test_failed_extraction_rejects_empty_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        FailedExtraction(
            source_doc_id="d",
            source_kind="news",
            failed_at="t",
            reason="   ",
            detail="x",
            prompt_version="p",
            model_version="m",
        )


def test_failed_extraction_round_trips_dict() -> None:
    original = _sample_failure()
    back = FailedExtraction.from_dict(original.to_dict())
    assert back == original


# ---------------------------------------------------------------------------
# ExtractedRecord
# ---------------------------------------------------------------------------


def test_extracted_record_success_round_trips_jsonl() -> None:
    record = ExtractedRecord(
        instrument_id="AAPL",
        extraction=NewsExtraction(sentiment=0.5, materiality=0.8),
        provenance=_sample_provenance(),
    )
    line = record.to_jsonl_line()
    payload = json.loads(line)
    back = ExtractedRecord.from_payload(payload)
    assert back == record


def test_extracted_record_failure_round_trips_jsonl() -> None:
    record = ExtractedRecord(
        instrument_id="AAPL",
        failure=_sample_failure(),
    )
    payload = json.loads(record.to_jsonl_line())
    back = ExtractedRecord.from_payload(payload)
    assert back == record


def test_extracted_record_rejects_both_success_and_failure() -> None:
    with pytest.raises(ValueError, match="cannot carry both"):
        ExtractedRecord(
            instrument_id="AAPL",
            extraction=NewsExtraction(sentiment=0.0, materiality=0.5),
            provenance=_sample_provenance(),
            failure=_sample_failure(),
        )


def test_extracted_record_rejects_neither() -> None:
    with pytest.raises(ValueError, match="must carry either"):
        ExtractedRecord(instrument_id="AAPL")


def test_extracted_record_rejects_lopsided_success() -> None:
    with pytest.raises(ValueError, match="extraction and provenance must be set"):
        ExtractedRecord(
            instrument_id="AAPL",
            extraction=NewsExtraction(sentiment=0.0, materiality=0.5),
            provenance=None,
        )


def test_extracted_record_rejects_unsupported_schema_version() -> None:
    record = ExtractedRecord(
        instrument_id="AAPL",
        failure=_sample_failure(),
    )
    payload = json.loads(record.to_jsonl_line())
    payload["schema_version"] = "v9999"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        ExtractedRecord.from_payload(payload)


def test_extracted_record_succeeded_property() -> None:
    success = ExtractedRecord(
        instrument_id="AAPL",
        extraction=NewsExtraction(sentiment=0.0, materiality=0.5),
        provenance=_sample_provenance(),
    )
    failure = ExtractedRecord(instrument_id="AAPL", failure=_sample_failure())
    assert success.succeeded is True
    assert failure.succeeded is False


# ---------------------------------------------------------------------------
# SourceDocument
# ---------------------------------------------------------------------------


def test_source_document_requires_tz_aware_published_at() -> None:
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        SourceDocument(
            doc_id="d1",
            instrument_id="AAPL",
            published_at=datetime(2024, 1, 1),  # naive
            kind="news",
            text="...",
        )


def test_source_document_rejects_empty_doc_id() -> None:
    from datetime import UTC, datetime

    with pytest.raises(ValueError, match="doc_id"):
        SourceDocument(
            doc_id="",
            instrument_id="AAPL",
            published_at=datetime(2024, 1, 1, tzinfo=UTC),
            kind="news",
            text="...",
        )


# ---------------------------------------------------------------------------
# Module-level
# ---------------------------------------------------------------------------


def test_schema_version_is_v2() -> None:
    assert SCHEMA_VERSION == "v2"


def test_utc_now_iso_returns_string_with_timezone() -> None:
    iso = utc_now_iso()
    assert isinstance(iso, str)
    assert iso.endswith("+00:00")
