"""Tests for the text-event-v2 expansion: filings + earnings calls + 5 new news features."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from quant_platform.research.features.text.aggregator import (
    build_earnings_call_panel,
    build_filing_panel,
    build_text_panel,
)
from quant_platform.research.features.text.client import MockLLMClient
from quant_platform.research.features.text.extraction import extract_documents
from quant_platform.research.features.text.features import (
    FEATURE_NAMES,
    FEATURE_SPECS,
    compute_text_features,
)
from quant_platform.research.features.text.prompts import (
    EARNINGS_CALL_PROMPT_VERSION,
    FILING_PROMPT_VERSION,
    get_earnings_call_prompt,
    get_filing_prompt,
    get_news_prompt,
    get_prompt_for_kind,
)
from quant_platform.research.features.text.schemas import (
    EarningsCallExtraction,
    ExtractedRecord,
    ExtractionProvenance,
    FilingExtraction,
    NewsExtraction,
    SourceDocument,
    extraction_from_dict_for_kind,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BASE_DATE = datetime(2024, 1, 2, 15, tzinfo=UTC)


def _date_with_offset(day_offset: int) -> datetime:
    """Compose a datetime that can stride days without month-overflow errors."""
    from datetime import timedelta

    return _BASE_DATE + timedelta(days=day_offset)


def _make_news_doc(doc_id: str, instrument_id: str, day_offset: int = 0) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        instrument_id=instrument_id,
        published_at=_date_with_offset(day_offset),
        kind="news",
        text="News text.",
    )


def _make_filing_doc(
    doc_id: str, instrument_id: str, day_offset: int = 0, kind: str = "filing-10q"
) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        instrument_id=instrument_id,
        published_at=_date_with_offset(day_offset),
        kind=kind,
        text="Filing text.",
    )


def _make_call_doc(doc_id: str, instrument_id: str, day_offset: int = 0) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        instrument_id=instrument_id,
        published_at=_date_with_offset(day_offset),
        kind="earnings-call",
        text="Call text.",
    )


def _filing_payload(**overrides: float) -> dict[str, float]:
    base = {
        "risk_sentiment": 0.2,
        "uncertainty_score": 0.3,
        "management_tone": 0.4,
        "litigation_risk": 0.1,
        "guidance_sentiment": 0.5,
        "supply_chain_risk": 0.0,
        "inventory_risk": 0.0,
        "margin_pressure": 0.2,
        "demand_weakness": 0.1,
        "financing_stress": 0.3,
    }
    base.update(overrides)
    return base


def _call_payload(**overrides: float) -> dict[str, float]:
    base = {
        "management_confidence": 0.5,
        "analyst_pushback": 0.4,
        "guidance_quality": 0.2,
        "margin_pressure": 0.1,
        "demand_signal": 0.3,
        "capex_intent": 0.0,
        "inventory_problem": -0.2,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# FilingExtraction
# ---------------------------------------------------------------------------


def test_filing_extraction_defaults_are_neutral() -> None:
    e = FilingExtraction(risk_sentiment=0.0, uncertainty_score=0.5)
    assert e.management_tone == 0.0
    assert e.guidance_sentiment == 0.0
    assert e.supply_chain_risk == 0.0
    assert e.inventory_risk == 0.0
    assert e.margin_pressure == 0.0
    assert e.demand_weakness == 0.0
    assert e.financing_stress == 0.0


def test_filing_extraction_rejects_out_of_range_signed() -> None:
    with pytest.raises(ValueError, match="risk_sentiment must lie in"):
        FilingExtraction(risk_sentiment=2.0, uncertainty_score=0.5)
    with pytest.raises(ValueError, match="management_tone must lie in"):
        FilingExtraction(risk_sentiment=0.0, uncertainty_score=0.5, management_tone=1.5)


def test_filing_extraction_rejects_out_of_range_unsigned() -> None:
    with pytest.raises(ValueError, match="uncertainty_score must lie in"):
        FilingExtraction(risk_sentiment=0.0, uncertainty_score=1.5)


def test_filing_extraction_round_trips_dict() -> None:
    original = FilingExtraction(**_filing_payload())  # type: ignore[arg-type]
    back = FilingExtraction.from_dict(original.to_dict())
    assert back == original


def test_filing_extraction_field_descriptions_cover_all_fields() -> None:
    descriptions = FilingExtraction.field_descriptions()
    all_fields = (*FilingExtraction.SIGNED_FIELDS, *FilingExtraction.UNSIGNED_FIELDS)
    assert set(descriptions) == set(all_fields)


# ---------------------------------------------------------------------------
# EarningsCallExtraction
# ---------------------------------------------------------------------------


def test_earnings_call_defaults_are_neutral() -> None:
    e = EarningsCallExtraction(management_confidence=0.0, analyst_pushback=0.0)
    assert e.guidance_quality == 0.0
    assert e.margin_pressure == 0.0
    assert e.demand_signal == 0.0
    assert e.capex_intent == 0.0
    assert e.inventory_problem == 0.0


def test_earnings_call_rejects_out_of_range_signed() -> None:
    with pytest.raises(ValueError, match="management_confidence must lie in"):
        EarningsCallExtraction(management_confidence=2.0, analyst_pushback=0.0)


def test_earnings_call_rejects_out_of_range_unsigned() -> None:
    with pytest.raises(ValueError, match="analyst_pushback must lie in"):
        EarningsCallExtraction(management_confidence=0.0, analyst_pushback=1.5)


def test_earnings_call_round_trips_dict() -> None:
    original = EarningsCallExtraction(**_call_payload())  # type: ignore[arg-type]
    back = EarningsCallExtraction.from_dict(original.to_dict())
    assert back == original


def test_earnings_call_field_descriptions_cover_all_fields() -> None:
    descriptions = EarningsCallExtraction.field_descriptions()
    all_fields = (
        *EarningsCallExtraction.SIGNED_FIELDS,
        *EarningsCallExtraction.UNSIGNED_FIELDS,
    )
    assert set(descriptions) == set(all_fields)


# ---------------------------------------------------------------------------
# extraction_from_dict_for_kind
# ---------------------------------------------------------------------------


def test_dispatch_news_to_news_extraction() -> None:
    result = extraction_from_dict_for_kind("news", {"sentiment": 0.4, "materiality": 0.5})
    assert isinstance(result, NewsExtraction)


def test_dispatch_filing_prefix_to_filing_extraction() -> None:
    for kind in ("filing-10k", "filing-10q", "filing-8k"):
        result = extraction_from_dict_for_kind(kind, _filing_payload())
        assert isinstance(result, FilingExtraction), kind


def test_dispatch_earnings_call_to_earnings_call_extraction() -> None:
    result = extraction_from_dict_for_kind("earnings-call", _call_payload())
    assert isinstance(result, EarningsCallExtraction)


def test_dispatch_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported source_kind"):
        extraction_from_dict_for_kind("podcast", {})


# ---------------------------------------------------------------------------
# ExtractedRecord round-trip across kinds
# ---------------------------------------------------------------------------


def _provenance_for(kind: str, doc_id: str = "d1") -> ExtractionProvenance:
    return ExtractionProvenance(
        prompt_version="x-prompt-v1",
        model_version="mock",
        source_kind=kind,
        source_doc_id=doc_id,
        extracted_at=utc_now_iso(),
        confidence=1.0,
        raw_response_hash="abcdef1234567890",  # pragma: allowlist secret
    )


def test_extracted_record_round_trips_filing_jsonl() -> None:
    record = ExtractedRecord(
        instrument_id="AAPL",
        extraction=FilingExtraction(**_filing_payload()),  # type: ignore[arg-type]
        provenance=_provenance_for("filing-10q"),
    )
    import json

    line = record.to_jsonl_line()
    back = ExtractedRecord.from_payload(json.loads(line))
    assert back == record
    assert isinstance(back.extraction, FilingExtraction)


def test_extracted_record_round_trips_earnings_call_jsonl() -> None:
    record = ExtractedRecord(
        instrument_id="AAPL",
        extraction=EarningsCallExtraction(**_call_payload()),  # type: ignore[arg-type]
        provenance=_provenance_for("earnings-call"),
    )
    import json

    back = ExtractedRecord.from_payload(json.loads(record.to_jsonl_line()))
    assert back == record
    assert isinstance(back.extraction, EarningsCallExtraction)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_filing_prompt_carries_version_pin() -> None:
    prompt = get_filing_prompt()
    assert prompt.version == FILING_PROMPT_VERSION
    assert prompt.version == "filing-prompt-v1"


def test_filing_prompt_schema_covers_all_filing_fields() -> None:
    prompt = get_filing_prompt()
    properties = prompt.output_schema["properties"]
    expected = set(FilingExtraction.SIGNED_FIELDS) | set(FilingExtraction.UNSIGNED_FIELDS)
    assert set(properties) == expected  # type: ignore[arg-type]


def test_earnings_call_prompt_carries_version_pin() -> None:
    prompt = get_earnings_call_prompt()
    assert prompt.version == EARNINGS_CALL_PROMPT_VERSION


def test_earnings_call_prompt_schema_covers_all_call_fields() -> None:
    prompt = get_earnings_call_prompt()
    properties = prompt.output_schema["properties"]
    expected = set(EarningsCallExtraction.SIGNED_FIELDS) | set(
        EarningsCallExtraction.UNSIGNED_FIELDS
    )
    assert set(properties) == expected  # type: ignore[arg-type]


def test_get_prompt_for_kind_dispatches() -> None:
    assert get_prompt_for_kind("news").version == get_news_prompt().version
    assert get_prompt_for_kind("filing-10q").version == get_filing_prompt().version
    assert get_prompt_for_kind("earnings-call").version == get_earnings_call_prompt().version


def test_get_prompt_for_kind_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported source_kind"):
        get_prompt_for_kind("podcast")


# ---------------------------------------------------------------------------
# Filing panel aggregator
# ---------------------------------------------------------------------------


def test_build_filing_panel_one_row_per_filing() -> None:
    docs = [
        _make_filing_doc("f1", "AAPL", day_offset=0),
        _make_filing_doc("f2", "AAPL", day_offset=30),
        _make_filing_doc("f3", "MSFT", day_offset=0),
    ]
    client = MockLLMClient(responder=lambda _p, _d: _filing_payload())
    records = extract_documents(client=client, prompt=get_filing_prompt(), documents=docs)
    panel = build_filing_panel(records=records, documents=docs)
    assert len(panel.frame) == 3
    assert panel.n_records_processed == 3
    assert (panel.frame["filing_count"] == 1).all()


def test_build_filing_panel_ignores_non_filing_records() -> None:
    docs = [
        _make_filing_doc("f1", "AAPL"),
        _make_news_doc("n1", "AAPL"),  # ignored
        _make_call_doc("c1", "AAPL"),  # ignored
    ]
    filing_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _filing_payload()),
        prompt=get_filing_prompt(),
        documents=[docs[0]],
    )
    news_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.5, "materiality": 0.5}),
        prompt=get_news_prompt(),
        documents=[docs[1]],
    )
    call_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _call_payload()),
        prompt=get_earnings_call_prompt(),
        documents=[docs[2]],
    )
    all_records = [*filing_records, *news_records, *call_records]
    panel = build_filing_panel(records=all_records, documents=docs)
    assert len(panel.frame) == 1
    assert panel.n_records_processed == 1


def test_build_filing_panel_empty_returns_typed_frame() -> None:
    panel = build_filing_panel(records=[], documents=[])
    assert panel.frame.empty
    assert "filing_count" in panel.frame.columns
    assert "filing_risk_sentiment" in panel.frame.columns


def test_build_filing_panel_failure_records_count_separately() -> None:
    docs = [_make_filing_doc("f1", "AAPL")]
    bad_client = MockLLMClient(
        responder=lambda _p, _d: {"risk_sentiment": 99.0, "uncertainty_score": 0.5}
    )
    records = extract_documents(client=bad_client, prompt=get_filing_prompt(), documents=docs)
    panel = build_filing_panel(records=records, documents=docs)
    assert panel.n_failures == 1
    row = panel.frame.iloc[0]
    assert row["filing_count"] == 0
    assert row["filing_failure_count"] == 1


# ---------------------------------------------------------------------------
# Earnings-call panel aggregator
# ---------------------------------------------------------------------------


def test_build_earnings_call_panel_one_row_per_call() -> None:
    docs = [
        _make_call_doc("c1", "AAPL"),
        _make_call_doc("c2", "MSFT", day_offset=1),
    ]
    client = MockLLMClient(responder=lambda _p, _d: _call_payload())
    records = extract_documents(client=client, prompt=get_earnings_call_prompt(), documents=docs)
    panel = build_earnings_call_panel(records=records, documents=docs)
    assert len(panel.frame) == 2
    assert (panel.frame["call_count"] == 1).all()


def test_build_earnings_call_panel_ignores_non_call_records() -> None:
    docs = [_make_call_doc("c1", "AAPL"), _make_filing_doc("f1", "AAPL")]
    call_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _call_payload()),
        prompt=get_earnings_call_prompt(),
        documents=[docs[0]],
    )
    filing_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _filing_payload()),
        prompt=get_filing_prompt(),
        documents=[docs[1]],
    )
    panel = build_earnings_call_panel(records=[*call_records, *filing_records], documents=docs)
    assert len(panel.frame) == 1


def test_build_earnings_call_panel_empty_returns_typed_frame() -> None:
    panel = build_earnings_call_panel(records=[], documents=[])
    assert panel.frame.empty
    assert "call_count" in panel.frame.columns
    assert "call_management_confidence" in panel.frame.columns


# ---------------------------------------------------------------------------
# News-only aggregator now ignores non-news records
# ---------------------------------------------------------------------------


def test_build_text_panel_ignores_non_news_records() -> None:
    docs = [_make_news_doc("n1", "AAPL"), _make_filing_doc("f1", "AAPL")]
    news_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.5, "materiality": 0.5}),
        prompt=get_news_prompt(),
        documents=[docs[0]],
    )
    filing_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _filing_payload()),
        prompt=get_filing_prompt(),
        documents=[docs[1]],
    )
    panel = build_text_panel(records=[*news_records, *filing_records], documents=docs)
    assert len(panel.frame) == 1
    assert panel.n_records_processed == 1


# ---------------------------------------------------------------------------
# News dispersion / novelty / materiality aggregator columns
# ---------------------------------------------------------------------------


def test_aggregator_sentiment_dispersion_uses_unweighted_population_std() -> None:
    # Two articles same day, sentiments 0.0 and 1.0. Mean = 0.5, var =
    # (0.25 + 0.25)/2 = 0.25, std = 0.5.
    docs = [
        _make_news_doc("n1", "AAPL", day_offset=0),
        _make_news_doc("n2", "AAPL", day_offset=0),
    ]
    sentiments = {"n1": 0.0, "n2": 1.0}
    client = MockLLMClient(
        responder=lambda _p, doc: {
            "sentiment": sentiments[doc.doc_id],
            "materiality": 0.5,
        }
    )
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    panel = build_text_panel(records=records, documents=docs)
    row = panel.frame.iloc[0]
    assert row["count"] == 2
    assert row["sentiment_dispersion"] == pytest.approx(0.5, abs=1e-9)


def test_aggregator_novelty_and_materiality_means() -> None:
    docs = [_make_news_doc("n1", "AAPL"), _make_news_doc("n2", "AAPL")]
    payloads = {
        "n1": {"sentiment": 0.0, "materiality": 0.2, "novelty": 0.4},
        "n2": {"sentiment": 0.0, "materiality": 0.8, "novelty": 0.6},
    }
    client = MockLLMClient(responder=lambda _p, doc: payloads[doc.doc_id])
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    panel = build_text_panel(records=records, documents=docs)
    row = panel.frame.iloc[0]
    assert row["novelty_mean"] == pytest.approx(0.5)
    assert row["materiality_mean"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Feature catalog
# ---------------------------------------------------------------------------


def test_feature_specs_total_twentyseven() -> None:
    assert len(FEATURE_SPECS) == 27
    assert len(FEATURE_NAMES) == 27
    # No duplicate names.
    assert len(set(FEATURE_NAMES)) == 27


def test_feature_specs_all_under_text_event_v2() -> None:
    for spec in FEATURE_SPECS:
        assert spec.version == "text-event-v2", spec.name
        assert spec.family == "text", spec.name


def test_feature_specs_evidence_gated_by_default() -> None:
    for spec in FEATURE_SPECS:
        assert spec.expected_direction == "unknown", spec.name
        assert spec.larger_is_better is False, spec.name


def test_feature_specs_include_all_news_features() -> None:
    expected_news = {
        "news_sentiment_1d",
        "news_sentiment_5d",
        "news_volume_1d",
        "news_volume_zscore_20d",
        "positive_news_shock",
        "negative_news_shock",
        "sentiment_change",
        "sentiment_dispersion",
        "news_novelty",
        "event_materiality",
    }
    assert expected_news.issubset(set(FEATURE_NAMES))


def test_feature_specs_include_all_filing_features() -> None:
    expected_filing = {
        "filing_risk_sentiment",
        "filing_uncertainty_score",
        "management_tone_change",
        "litigation_risk_score",
        "filing_guidance_sentiment",
        "supply_chain_risk_score",
        "inventory_risk_score",
        "margin_pressure_score",
        "demand_weakness_score",
        "financing_stress_score",
    }
    assert expected_filing.issubset(set(FEATURE_NAMES))


def test_feature_specs_include_all_call_features() -> None:
    expected_call = {
        "management_confidence",
        "analyst_pushback",
        "guidance_quality",
        "call_margin_pressure",
        "call_demand_signal",
        "capex_intent",
        "inventory_problem",
    }
    assert expected_call.issubset(set(FEATURE_NAMES))


# ---------------------------------------------------------------------------
# compute_text_features end-to-end (mixed kinds)
# ---------------------------------------------------------------------------


def test_compute_text_features_handles_mixed_kinds() -> None:
    news_docs = [_make_news_doc(f"n{i}", "AAPL", day_offset=i) for i in range(2)]
    filing_docs = [_make_filing_doc("f1", "AAPL", day_offset=5)]
    call_docs = [_make_call_doc("c1", "AAPL", day_offset=10)]
    all_docs = [*news_docs, *filing_docs, *call_docs]

    news_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.4, "materiality": 0.5}),
        prompt=get_news_prompt(),
        documents=news_docs,
    )
    filing_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _filing_payload()),
        prompt=get_filing_prompt(),
        documents=filing_docs,
    )
    call_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _call_payload()),
        prompt=get_earnings_call_prompt(),
        documents=call_docs,
    )
    ff = compute_text_features(
        records=[*news_records, *filing_records, *call_records],
        documents=all_docs,
    )
    # 4 unique (instrument, date) rows: 2 news, 1 filing, 1 call (all AAPL).
    assert len(ff.frame) == 4
    # All 27 columns present.
    for name in FEATURE_NAMES:
        assert name in ff.frame.columns
    # News features non-NaN on news dates only.
    news_rows = ff.frame.iloc[:2]
    assert news_rows["news_sentiment_1d"].notna().all()
    # Filing features non-NaN on filing date.
    filing_row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-01-07")]
    assert len(filing_row) == 1
    assert pd.notna(filing_row.iloc[0]["filing_risk_sentiment"])
    # Call features non-NaN on call date.
    call_row = ff.frame[ff.frame["date"] == pd.Timestamp("2024-01-12")]
    assert len(call_row) == 1
    assert pd.notna(call_row.iloc[0]["management_confidence"])


def test_compute_text_features_management_tone_change_is_filing_diff() -> None:
    docs = [
        _make_filing_doc("q1", "AAPL", day_offset=0),
        _make_filing_doc("q2", "AAPL", day_offset=90),
        _make_filing_doc("q3", "AAPL", day_offset=180),
    ]
    tones = {"q1": 0.2, "q2": 0.5, "q3": 0.1}
    client = MockLLMClient(
        responder=lambda _p, doc: _filing_payload(management_tone=tones[doc.doc_id])
    )
    records = extract_documents(client=client, prompt=get_filing_prompt(), documents=docs)
    ff = compute_text_features(records=records, documents=docs)
    sorted_frame = ff.frame.sort_values("date").reset_index(drop=True)
    # First filing: no prior, so NaN.
    assert pd.isna(sorted_frame.iloc[0]["management_tone_change"])
    # Second filing: 0.5 - 0.2 = 0.3.
    assert sorted_frame.iloc[1]["management_tone_change"] == pytest.approx(0.3)
    # Third filing: 0.1 - 0.5 = -0.4.
    assert sorted_frame.iloc[2]["management_tone_change"] == pytest.approx(-0.4)


def test_compute_text_features_news_sentiment_5d_uses_full_window() -> None:
    # 4 days of data → not enough for a 5-day full-window mean.
    docs = [_make_news_doc(f"n{i}", "AAPL", day_offset=i) for i in range(4)]
    client = MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.5, "materiality": 0.5})
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    ff = compute_text_features(records=records, documents=docs)
    # 5d rolling mean requires 5 observations; with only 4, all NaN.
    assert ff.frame["news_sentiment_5d"].isna().all()


def test_compute_text_features_news_sentiment_5d_emits_when_full_window() -> None:
    # 5 days at sentiment 0.4 → 5d mean = 0.4 on the fifth day.
    docs = [_make_news_doc(f"n{i}", "AAPL", day_offset=i) for i in range(5)]
    client = MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.4, "materiality": 0.5})
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    ff = compute_text_features(records=records, documents=docs)
    sorted_frame = ff.frame.sort_values("date").reset_index(drop=True)
    # Only the 5th row has the full window.
    assert pd.isna(sorted_frame.iloc[3]["news_sentiment_5d"])
    assert sorted_frame.iloc[4]["news_sentiment_5d"] == pytest.approx(0.4)


def test_compute_text_features_sentiment_change_uses_prior_day() -> None:
    # Two days: 0.3 then 0.7. Change on day 2 = 0.4.
    docs = [
        _make_news_doc("n1", "AAPL", day_offset=0),
        _make_news_doc("n2", "AAPL", day_offset=1),
    ]
    sentiments = {"n1": 0.3, "n2": 0.7}
    client = MockLLMClient(
        responder=lambda _p, doc: {
            "sentiment": sentiments[doc.doc_id],
            "materiality": 0.5,
        }
    )
    records = extract_documents(client=client, prompt=get_news_prompt(), documents=docs)
    ff = compute_text_features(records=records, documents=docs)
    sorted_frame = ff.frame.sort_values("date").reset_index(drop=True)
    assert pd.isna(sorted_frame.iloc[0]["sentiment_change"])
    assert sorted_frame.iloc[1]["sentiment_change"] == pytest.approx(0.4)


def test_compute_text_features_empty_records_returns_empty_frame() -> None:
    ff = compute_text_features(records=[], documents=[])
    assert ff.frame.empty
    assert set(ff.feature_names) == set(FEATURE_NAMES)
    assert all(v == 0 for v in ff.coverage.values())


def test_filing_features_are_sparse_outside_filing_dates() -> None:
    """When news + filings coexist, filing columns must be NaN on news-only dates."""
    news_docs = [_make_news_doc("n1", "AAPL", day_offset=0)]
    filing_docs = [_make_filing_doc("f1", "AAPL", day_offset=5)]

    news_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.4, "materiality": 0.5}),
        prompt=get_news_prompt(),
        documents=news_docs,
    )
    filing_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _filing_payload()),
        prompt=get_filing_prompt(),
        documents=filing_docs,
    )
    ff = compute_text_features(
        records=[*news_records, *filing_records],
        documents=[*news_docs, *filing_docs],
    )
    sorted_frame = ff.frame.sort_values("date").reset_index(drop=True)
    # News date → filing features NaN.
    news_row = sorted_frame.iloc[0]
    assert pd.isna(news_row["filing_risk_sentiment"])
    # Filing date → filing features non-NaN, news features NaN.
    filing_row = sorted_frame.iloc[1]
    assert pd.notna(filing_row["filing_risk_sentiment"])
    assert pd.isna(filing_row["news_sentiment_1d"])


# ---------------------------------------------------------------------------
# Edge cases — kind whitelist + record-without-document + trading-dates
# ---------------------------------------------------------------------------


def test_unknown_filing_kind_is_rejected_by_dispatcher() -> None:
    """Permissive ``startswith("filing")`` was replaced with an explicit
    whitelist. ``filing-poetry`` is unknown and must raise rather than
    silently routing to FilingExtraction (where it would fail later
    with a confusing "missing fields" error)."""
    with pytest.raises(ValueError, match="Unsupported source_kind"):
        extraction_from_dict_for_kind("filing-poetry", _filing_payload())
    with pytest.raises(ValueError, match="Unsupported source_kind"):
        get_prompt_for_kind("filing-poetry")


def test_known_filing_kinds_constant_includes_all_supported_forms() -> None:
    from quant_platform.research.features.text.schemas import (
        KNOWN_FILING_KINDS,
        KNOWN_SOURCE_KINDS,
    )

    assert set(KNOWN_FILING_KINDS) == {"filing-10k", "filing-10q", "filing-8k"}
    assert set(KNOWN_SOURCE_KINDS) == {*KNOWN_FILING_KINDS, "news", "earnings-call"}


def test_record_with_missing_source_doc_is_silently_skipped() -> None:
    """When an ``ExtractedRecord`` references a ``source_doc_id`` not
    in the document list, the aggregator must skip the row (no
    ``published_at`` to assign). The skip is silent; the operator
    reads about it via ``n_records_processed`` vs len(records)."""
    docs = [_make_news_doc("n1", "AAPL")]
    extra_provenance = ExtractionProvenance(
        prompt_version="news-prompt-v1",
        model_version="mock",
        source_kind="news",
        source_doc_id="orphan-doc-id-not-in-list",
        extracted_at=utc_now_iso(),
        confidence=1.0,
        raw_response_hash="abcdef1234567890",  # pragma: allowlist secret
    )
    orphan_record = ExtractedRecord(
        instrument_id="AAPL",
        extraction=NewsExtraction(sentiment=0.5, materiality=0.5),
        provenance=extra_provenance,
    )
    legit_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.5, "materiality": 0.5}),
        prompt=get_news_prompt(),
        documents=docs,
    )
    panel = build_text_panel(
        records=[*legit_records, orphan_record],
        documents=docs,
    )
    # Only the legit record contributed a row.
    assert len(panel.frame) == 1
    # But n_records_processed counts BOTH (the orphan was still seen).
    assert panel.n_records_processed == 2


def test_compute_text_features_with_trading_dates_keeps_filings_sparse() -> None:
    """When ``trading_dates`` densifies the news panel, filing + call
    rows must NOT be back-filled — they remain sparse on their
    publication dates only."""
    news_docs = [_make_news_doc("n1", "AAPL", day_offset=0)]
    filing_docs = [_make_filing_doc("f1", "AAPL", day_offset=2)]
    call_docs = [_make_call_doc("c1", "AAPL", day_offset=4)]
    all_docs = [*news_docs, *filing_docs, *call_docs]

    news_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: {"sentiment": 0.5, "materiality": 0.5}),
        prompt=get_news_prompt(),
        documents=news_docs,
    )
    filing_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _filing_payload()),
        prompt=get_filing_prompt(),
        documents=filing_docs,
    )
    call_records = extract_documents(
        client=MockLLMClient(responder=lambda _p, _d: _call_payload()),
        prompt=get_earnings_call_prompt(),
        documents=call_docs,
    )

    # Densify across 5 dates (only days 0, 2, 4 carry events).
    trading_dates = pd.DatetimeIndex(
        [_date_with_offset(i).replace(hour=0, tzinfo=None) for i in range(5)]
    )
    ff = compute_text_features(
        records=[*news_records, *filing_records, *call_records],
        documents=all_docs,
        trading_dates=trading_dates,
    )
    # News panel densified → 5 rows. Filings + calls join on the same
    # (instrument, date) grid so they still produce 5 rows total.
    sorted_frame = ff.frame.sort_values("date").reset_index(drop=True)
    assert len(sorted_frame) == 5
    # Day 2 (filing date) — filing feature non-NaN.
    assert pd.notna(sorted_frame.iloc[2]["filing_risk_sentiment"])
    # Day 4 (call date) — call feature non-NaN.
    assert pd.notna(sorted_frame.iloc[4]["management_confidence"])
    # All other days — filing/call features NaN (sparse).
    for i in (0, 1, 3):
        assert pd.isna(sorted_frame.iloc[i]["filing_risk_sentiment"])
        assert pd.isna(sorted_frame.iloc[i]["management_confidence"])
    # News volume on day 0 = 1 (event); other days = 0 (densified zero).
    assert sorted_frame.iloc[0]["news_volume_1d"] == 1
    assert sorted_frame.iloc[1]["news_volume_1d"] == 0


def test_v1_legacy_payload_loads_as_news_extraction() -> None:
    """``v1`` JSONL records predate the tagged-union routing — they
    were always news. The v2 loader must accept them so the ~100
    articles already persisted under v1 (during TWS dev) remain
    readable without an external migration step."""
    import json

    legacy_payload = {
        "schema_version": "v1",
        "instrument_id": "AAPL",
        "extraction": {
            "sentiment": 0.4,
            "materiality": 0.7,
            "demand_signal": 0.0,
            "margin_pressure": 0.0,
            "guidance_signal": 0.0,
            "litigation_risk": 0.0,
            "novelty": 0.5,
        },
        "provenance": {
            "prompt_version": "news-prompt-v1",
            "model_version": "claude-sonnet-4-5",
            "source_kind": "news",
            "source_doc_id": "doc-001",
            "extracted_at": "2024-01-01T00:00:00+00:00",
            "confidence": 1.0,
            "raw_response_hash": "abcdef1234567890",  # pragma: allowlist secret
        },
    }
    record = ExtractedRecord.from_payload(legacy_payload)
    assert record.succeeded
    assert isinstance(record.extraction, NewsExtraction)
    assert record.schema_version == "v1"
    # Round-trip preserves the legacy version stamp so re-emitted
    # JSONL is still recognisable as v1.
    assert json.loads(record.to_jsonl_line())["schema_version"] == "v1"


def test_v1_legacy_with_typo_source_kind_still_dispatches_news() -> None:
    """v1 routing is hard-coded to news regardless of what
    ``source_kind`` field says — protects against historical typos
    in the operator's source_kind value."""
    legacy_payload = {
        "schema_version": "v1",
        "instrument_id": "AAPL",
        "extraction": {"sentiment": 0.4, "materiality": 0.7},
        "provenance": {
            "prompt_version": "news-prompt-v1",
            "model_version": "mock",
            # Typo here — v2 dispatch would reject "neews" outright;
            # v1 loader force-routes to news.
            "source_kind": "neews",
            "source_doc_id": "d1",
            "extracted_at": "2024-01-01T00:00:00+00:00",
            "confidence": 1.0,
            "raw_response_hash": "abcdef1234567890",  # pragma: allowlist secret
        },
    }
    record = ExtractedRecord.from_payload(legacy_payload)
    assert isinstance(record.extraction, NewsExtraction)


def test_aggregator_isinstance_guard_in_build_kind_panel() -> None:
    """Internal core ``_build_kind_panel`` raises ``TypeError`` if the
    extraction type doesn't match the kind filter. The public builders
    enforce this via the filter; this is a defense-in-depth test that
    constructs an inconsistent record and pokes the internal core
    directly."""
    from quant_platform.research.features.text.aggregator import _build_kind_panel
    from quant_platform.research.features.text.schemas import (
        EarningsCallExtraction,
        FilingExtraction,
    )

    # Build a record whose source_kind says "filing-10q" but whose
    # extraction is a NewsExtraction (bypassing the public builder's
    # kind filter).
    news_extraction = NewsExtraction(sentiment=0.0, materiality=0.5)
    wrong_provenance = ExtractionProvenance(
        prompt_version="x",
        model_version="m",
        source_kind="filing-10q",
        source_doc_id="d1",
        extracted_at=utc_now_iso(),
        confidence=1.0,
        raw_response_hash="abcdef1234567890",  # pragma: allowlist secret
    )
    # ExtractedRecord doesn't validate kind ↔ extraction-class match,
    # so the inconsistent record constructs fine.
    bad_record = ExtractedRecord(
        instrument_id="AAPL",
        extraction=news_extraction,
        provenance=wrong_provenance,
    )
    bad_doc = SourceDocument(
        doc_id="d1",
        instrument_id="AAPL",
        published_at=_date_with_offset(0),
        kind="filing-10q",
        text="...",
    )
    with pytest.raises(TypeError, match="expected FilingExtraction"):
        _build_kind_panel(
            records=[bad_record],
            document_index={bad_doc.doc_id: bad_doc},
            kind_filter=("filing-10q",),
            column_prefix="filing_",
            mean_field_names=(*FilingExtraction.SIGNED_FIELDS, *FilingExtraction.UNSIGNED_FIELDS),
            extraction_class=FilingExtraction,
        )

    # Same shape, but for an earnings-call mismatch.
    wrong_call_record = ExtractedRecord(
        instrument_id="AAPL",
        extraction=news_extraction,
        provenance=ExtractionProvenance(
            prompt_version="x",
            model_version="m",
            source_kind="earnings-call",
            source_doc_id="d1",
            extracted_at=utc_now_iso(),
            confidence=1.0,
            raw_response_hash="abcdef1234567890",  # pragma: allowlist secret
        ),
    )
    with pytest.raises(TypeError, match="expected EarningsCallExtraction"):
        _build_kind_panel(
            records=[wrong_call_record],
            document_index={bad_doc.doc_id: bad_doc},
            kind_filter="earnings-call",
            column_prefix="call_",
            mean_field_names=(
                *EarningsCallExtraction.SIGNED_FIELDS,
                *EarningsCallExtraction.UNSIGNED_FIELDS,
            ),
            extraction_class=EarningsCallExtraction,
        )
