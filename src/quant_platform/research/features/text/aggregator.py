"""Per-document extraction → (instrument_id, date) panel.

Extracted records arrive at this layer one per source document. For
each document kind (news / filing / earnings-call) we produce a wide
panel keyed by ``(instrument_id, date)`` carrying the metrics the
feature compute layer needs.

The aggregator is **PIT-safe**: each extraction's
``provenance.extracted_at`` is the time the LLM ran, but the panel
date is keyed to the source document's ``published_at`` so the
feature value at date ``d`` only sees documents that became publicly
available on or before ``d``. The extraction-timestamp is metadata,
not a join key — that's the whole point of running the LLM offline
ahead of the next trading day.

Failed extractions are tallied separately into a ``*_failure_count``
column so a downstream coverage report can show how many documents
the LLM couldn't process. Their content is otherwise ignored —
"failure" is the source of truth, not a zero-sentiment phantom.

Three panel builders live here:

* :func:`build_text_panel` — news records (``source_kind == "news"``).
  Materiality-weighted sentiment, volume, positive/negative shock
  counts, novelty/materiality moments, dispersion. Columns are
  un-prefixed for backward compatibility with the v1 layout.
* :func:`build_filing_panel` — SEC-filing records
  (``source_kind`` starts with ``"filing"``). One row per
  ``(instrument_id, published-date)``; columns are namespaced
  ``filing_*``.
* :func:`build_earnings_call_panel` — earnings-call records
  (``source_kind == "earnings-call"``). One row per
  ``(instrument_id, call-date)``; columns namespaced ``call_*``.

Each builder filters input records to its own ``source_kind`` and
silently ignores the others, so a single mixed list of records can be
fanned out to all three without pre-partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.text.schemas import KNOWN_FILING_KINDS, NewsExtraction

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from quant_platform.research.features.text.schemas import (
        ExtractedRecord,
        SourceDocument,
    )


#: Threshold above (below) which an article counts as "positive"
#: ("negative") for the shock features. Conservative default — the
#: aggregator only counts articles whose sentiment is clearly one
#: side or the other, not the noise around zero.
POSITIVE_SENTIMENT_THRESHOLD: float = 0.3
NEGATIVE_SENTIMENT_THRESHOLD: float = -0.3


@dataclass(frozen=True)
class AggregatedTextPanel:
    """Wide-format intermediate the feature compute reads.

    The frame is keyed by ``(instrument_id, date)`` and carries one
    column per aggregated metric. ``compute_text_features`` consumes
    this directly; nothing else should reach into the aggregator
    output, so the dataclass keeps callers honest.
    """

    frame: pd.DataFrame
    n_records_processed: int
    n_failures: int


def _published_date(
    record: ExtractedRecord,
    document_index: Mapping[str, SourceDocument],
) -> pd.Timestamp | None:
    """Look up the source document's ``published_at`` and return it
    normalised to a naive UTC date (matches the rest of the panel
    convention)."""
    doc_id = (
        record.provenance.source_doc_id
        if record.provenance is not None
        else (record.failure.source_doc_id if record.failure is not None else None)
    )
    if doc_id is None:
        return None
    document = document_index.get(doc_id)
    if document is None:
        return None
    return pd.Timestamp(document.published_at).tz_convert("UTC").normalize().tz_localize(None)


def _record_source_kind(record: ExtractedRecord) -> str:
    """Return the ``source_kind`` of a record regardless of whether it
    succeeded or failed. Empty string when neither side carries
    ``source_kind``."""
    if record.provenance is not None:
        return record.provenance.source_kind
    if record.failure is not None:
        return record.failure.source_kind
    return ""


def _empty_panel_frame(
    *,
    extra_columns: tuple[str, ...],
    integer_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build an empty, typed aggregator-panel frame.

    Every per-kind builder returns the same shape on the empty path
    (instrument_id + date + named columns); centralising the dtype
    table prevents the three builders from drifting apart.

    Parameters
    ----------
    extra_columns:
        Column names beyond ``("instrument_id", "date")``. Float
        dtype unless listed in ``integer_columns``.
    integer_columns:
        Subset of ``extra_columns`` to type as ``int64`` (count-style
        columns).
    """
    int_set = set(integer_columns)
    series: dict[str, pd.Series] = {
        "instrument_id": pd.Series(dtype=str),
        "date": pd.Series(dtype="datetime64[ns]"),
    }
    for column in extra_columns:
        series[column] = pd.Series(dtype="int64" if column in int_set else "float64")
    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# News panel
# ---------------------------------------------------------------------------


def build_text_panel(
    *,
    records: Iterable[ExtractedRecord],
    documents: Iterable[SourceDocument],
    positive_threshold: float = POSITIVE_SENTIMENT_THRESHOLD,
    negative_threshold: float = NEGATIVE_SENTIMENT_THRESHOLD,
) -> AggregatedTextPanel:
    """Aggregate news extractions into a per-(instrument, date) frame.

    Non-news records (filings, earnings calls) are silently ignored —
    use :func:`build_filing_panel` / :func:`build_earnings_call_panel`
    for those. This keeps the legacy v1 contract intact while letting
    callers pass a single mixed records list.

    Parameters
    ----------
    records:
        Iterable of :class:`ExtractedRecord`. Successes and failures
        both — successes contribute to all metrics; failures only to
        ``failure_count``. Records whose ``source_kind`` is not
        ``"news"`` are skipped without contributing to
        ``n_records_processed``.
    documents:
        Iterable of the original :class:`SourceDocument` instances.
        Used to look up ``published_at`` per record. Indexed by
        ``doc_id`` internally; documents the records don't reference
        are silently ignored.
    positive_threshold:
        Sentiment cutoff above which an article counts toward
        ``positive_count``.
    negative_threshold:
        Sentiment cutoff below which an article counts toward
        ``negative_count``. Negative number (e.g. ``-0.3``).

    Returns
    -------
    AggregatedTextPanel
        Wide frame keyed by ``(instrument_id, date)``.
    """
    return _build_news_panel_from_index(
        records=records,
        document_index={doc.doc_id: doc for doc in documents},
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )


def _build_news_panel_from_index(
    *,
    records: Iterable[ExtractedRecord],
    document_index: Mapping[str, SourceDocument],
    positive_threshold: float = POSITIVE_SENTIMENT_THRESHOLD,
    negative_threshold: float = NEGATIVE_SENTIMENT_THRESHOLD,
) -> AggregatedTextPanel:
    """News-panel core that takes a pre-built ``document_index``.

    ``compute_text_features`` calls all three per-kind builders in
    sequence; routing through this internal helper lets it build the
    index once and pass it down, instead of three identical O(n)
    rebuilds.
    """
    if positive_threshold <= 0:
        raise ValueError("positive_threshold must be > 0")
    if negative_threshold >= 0:
        raise ValueError("negative_threshold must be < 0")

    news_records = [r for r in records if _record_source_kind(r) == "news"]

    rows: list[dict[str, object]] = []
    n_failures = 0
    for record in news_records:
        published = _published_date(record, document_index)
        if published is None:
            # Document not in the index; skip the row but still count
            # the record so callers can see the discrepancy via
            # ``n_records_processed``.
            continue
        if not record.succeeded:
            # ``ExtractedRecord.__post_init__`` guarantees that
            # ``succeeded == False`` ⇒ ``failure is not None``, so we
            # don't need an explicit narrowing assert here.
            n_failures += 1
            rows.append(
                {
                    "instrument_id": record.instrument_id,
                    "date": published,
                    "sentiment_weighted": 0.0,
                    "count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "novelty_sum": 0.0,
                    "novelty_max": 0.0,
                    "materiality_sum": 0.0,
                    "sentiment_sum": 0.0,
                    "sentiment_squared_sum": 0.0,
                    "failure_count": 1,
                }
            )
            continue
        # ``ExtractedRecord.__post_init__`` guarantees that the
        # ``record.succeeded`` branch carries a non-None ``extraction``,
        # and the news filter above guarantees the extraction is a
        # ``NewsExtraction``. The runtime check narrows the type for
        # mypy and surfaces a clear error if a future contributor
        # breaks the filter invariant — using ``raise`` (not
        # ``assert``) because ``python -O`` strips assertions.
        extraction = record.extraction
        if not isinstance(extraction, NewsExtraction):
            raise TypeError(
                f"build_text_panel: news filter let through {type(extraction).__name__}"
            )
        sentiment_weighted = extraction.sentiment * extraction.materiality
        rows.append(
            {
                "instrument_id": record.instrument_id,
                "date": published,
                "sentiment_weighted": sentiment_weighted,
                "count": 1,
                "positive_count": int(extraction.sentiment >= positive_threshold),
                "negative_count": int(extraction.sentiment <= negative_threshold),
                "novelty_sum": extraction.novelty,
                "novelty_max": extraction.novelty,
                "materiality_sum": extraction.materiality,
                # Unweighted sentiment moments — feed
                # ``sentiment_dispersion`` (per-date std of per-article
                # sentiment). Materiality-weighted ``sentiment_mean`` is
                # better for the central tendency; the dispersion feature
                # captures *disagreement* across the day's articles, so
                # the unweighted variant is the cleaner reading.
                "sentiment_sum": extraction.sentiment,
                "sentiment_squared_sum": extraction.sentiment * extraction.sentiment,
                "failure_count": 0,
            }
        )

    if not rows:
        empty = _empty_panel_frame(
            extra_columns=(
                "sentiment_weighted",
                "count",
                "positive_count",
                "negative_count",
                "novelty_sum",
                "novelty_max",
                "materiality_sum",
                "sentiment_sum",
                "sentiment_squared_sum",
                "failure_count",
                "sentiment_mean",
                "sentiment_dispersion",
                "novelty_mean",
                "materiality_mean",
            ),
            integer_columns=("count", "positive_count", "negative_count", "failure_count"),
        )
        return AggregatedTextPanel(
            frame=empty, n_records_processed=len(news_records), n_failures=n_failures
        )

    df = pd.DataFrame(rows)
    grouped = df.groupby(["instrument_id", "date"], sort=False, as_index=False)
    aggregated = grouped.agg(
        sentiment_weighted=("sentiment_weighted", "sum"),
        count=("count", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        novelty_sum=("novelty_sum", "sum"),
        novelty_max=("novelty_max", "max"),
        materiality_sum=("materiality_sum", "sum"),
        sentiment_sum=("sentiment_sum", "sum"),
        sentiment_squared_sum=("sentiment_squared_sum", "sum"),
        failure_count=("failure_count", "sum"),
    )
    # Materiality-weighted mean: sentiment_weighted / materiality_sum,
    # NaN when no materiality (i.e. only failures on a date).
    aggregated["sentiment_mean"] = np.where(
        aggregated["materiality_sum"] > 0,
        aggregated["sentiment_weighted"] / aggregated["materiality_sum"],
        np.nan,
    )
    # Unweighted per-date sentiment dispersion: sqrt(E[s²] − E[s]²). We
    # use the unweighted variant so the feature captures disagreement
    # across the day's articles independently of materiality weighting.
    # Population-style (divide by N, not N-1) because we treat the day's
    # articles as the population — there's no inference about a wider
    # sample, just a summary of the observed set.
    count_safe = aggregated["count"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        unweighted_mean = np.where(count_safe > 0, aggregated["sentiment_sum"] / count_safe, np.nan)
        unweighted_var = np.where(
            count_safe > 0,
            aggregated["sentiment_squared_sum"] / count_safe - unweighted_mean**2,
            np.nan,
        )
    # Numerical guard: squared-sum minus squared-mean can land at a tiny
    # negative number for nearly-identical floats. Clip to zero before
    # the sqrt so the feature is real-valued.
    aggregated["sentiment_dispersion"] = np.sqrt(np.clip(unweighted_var, 0.0, None))
    aggregated["novelty_mean"] = np.where(
        count_safe > 0, aggregated["novelty_sum"] / count_safe, np.nan
    )
    aggregated["materiality_mean"] = np.where(
        count_safe > 0, aggregated["materiality_sum"] / count_safe, np.nan
    )
    aggregated = aggregated.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return AggregatedTextPanel(
        frame=aggregated, n_records_processed=len(news_records), n_failures=n_failures
    )


# ---------------------------------------------------------------------------
# Filing panel
# ---------------------------------------------------------------------------


def _filing_mean_fields() -> tuple[str, ...]:
    """Filing extraction fields aggregated by per-(instrument, date)
    **mean** — derived from :class:`FilingExtraction`'s field
    catalogues. Lazy import avoids a top-level cycle and keeps a
    single source of truth: adding a field to ``FilingExtraction``
    automatically extends the panel.

    The typical case is one filing per date, so the mean equals the
    single observation, but the formulation degrades gracefully when
    an instrument has multiple filings on the same date (8-K + 10-Q
    on the same day, for example).
    """
    from quant_platform.research.features.text.schemas import FilingExtraction

    return (*FilingExtraction.SIGNED_FIELDS, *FilingExtraction.UNSIGNED_FIELDS)


def _build_kind_panel(
    *,
    records: Iterable[ExtractedRecord],
    document_index: Mapping[str, SourceDocument],
    kind_filter: str | tuple[str, ...] | frozenset[str] | set[str],
    column_prefix: str,
    mean_field_names: tuple[str, ...],
    extraction_class: type,
) -> AggregatedTextPanel:
    """Shared core for the filing + earnings-call aggregators.

    The two panels differ only in:

    * which ``source_kind`` they include,
    * what column prefix they use (``filing_`` vs ``call_``),
    * which extraction class they expect (governs the runtime
      isinstance guard).

    Centralising the logic keeps the two builders' failure-handling,
    empty-frame shape, and group-by-mean semantics in lockstep.

    Parameters
    ----------
    kind_filter:
        Either a single ``source_kind`` string or a ``tuple``/``set``
        of accepted kinds. Records whose ``source_kind`` doesn't
        match are silently skipped.
    column_prefix:
        Prepended to ``mean_field_names`` to name the exported
        columns. Also produces ``<prefix>count`` and
        ``<prefix>failure_count`` count columns.
    mean_field_names:
        Per-extraction field names that get aggregated by mean.
    extraction_class:
        Concrete extraction class (``FilingExtraction`` or
        ``EarningsCallExtraction``). Used for an ``isinstance``
        narrowing assertion — a future contributor who adds a
        non-matching record kind above the filter will hit the
        assertion at runtime instead of producing a silent type
        mismatch.
    """
    accepted_kinds: set[str] = {kind_filter} if isinstance(kind_filter, str) else set(kind_filter)
    kind_records = [r for r in records if _record_source_kind(r) in accepted_kinds]

    count_column = f"{column_prefix}count"
    failure_count_column = f"{column_prefix}failure_count"

    rows: list[dict[str, object]] = []
    n_failures = 0
    for record in kind_records:
        published = _published_date(record, document_index)
        if published is None:
            continue
        if not record.succeeded:
            n_failures += 1
            base: dict[str, object] = {
                "instrument_id": record.instrument_id,
                "date": published,
                count_column: 0,
                failure_count_column: 1,
            }
            for field_name in mean_field_names:
                # Failures don't contribute to the mean — encoded as
                # NaN so the groupby ``.mean()`` skips them, which is
                # exactly the "no signal" outcome we want.
                base[f"{column_prefix}{field_name}"] = np.nan
            rows.append(base)
            continue
        extraction = record.extraction
        # Runtime narrowing — the kind filter above should guarantee
        # this, but if a future contributor adds a branch that
        # bypasses the filter, the check surfaces the bug here
        # instead of producing a wrong-type panel. Using ``raise``
        # (not ``assert``) because ``python -O`` strips assertions.
        if not isinstance(extraction, extraction_class):
            raise TypeError(
                f"_build_kind_panel: expected {extraction_class.__name__} "
                f"for kind in {accepted_kinds!r}, got {type(extraction).__name__}"
            )
        row: dict[str, object] = {
            "instrument_id": record.instrument_id,
            "date": published,
            count_column: 1,
            failure_count_column: 0,
        }
        for field_name in mean_field_names:
            row[f"{column_prefix}{field_name}"] = float(getattr(extraction, field_name))
        rows.append(row)

    extra_columns = (
        count_column,
        failure_count_column,
        *(f"{column_prefix}{name}" for name in mean_field_names),
    )
    if not rows:
        empty = _empty_panel_frame(
            extra_columns=extra_columns,
            integer_columns=(count_column, failure_count_column),
        )
        return AggregatedTextPanel(
            frame=empty,
            n_records_processed=len(kind_records),
            n_failures=n_failures,
        )

    df = pd.DataFrame(rows)
    grouped = df.groupby(["instrument_id", "date"], sort=False, as_index=False)
    agg_spec: dict[str, tuple[str, str]] = {
        count_column: (count_column, "sum"),
        failure_count_column: (failure_count_column, "sum"),
    }
    for field_name in mean_field_names:
        column = f"{column_prefix}{field_name}"
        agg_spec[column] = (column, "mean")
    aggregated = grouped.agg(**agg_spec)
    aggregated = aggregated.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    return AggregatedTextPanel(
        frame=aggregated,
        n_records_processed=len(kind_records),
        n_failures=n_failures,
    )


def build_filing_panel(
    *,
    records: Iterable[ExtractedRecord],
    documents: Iterable[SourceDocument],
) -> AggregatedTextPanel:
    """Aggregate filing extractions into a per-(instrument, date) frame.

    The resulting frame has one row per ``(instrument_id, date)`` and
    columns ``filing_count``, ``filing_failure_count``, and one
    ``filing_<field>`` per filing-extraction field. Non-filing records
    are silently ignored.
    """
    from quant_platform.research.features.text.schemas import FilingExtraction

    return _build_kind_panel(
        records=records,
        document_index={doc.doc_id: doc for doc in documents},
        kind_filter=KNOWN_FILING_KINDS,
        column_prefix="filing_",
        mean_field_names=_filing_mean_fields(),
        extraction_class=FilingExtraction,
    )


# ---------------------------------------------------------------------------
# Earnings-call panel
# ---------------------------------------------------------------------------


def _earnings_call_mean_fields() -> tuple[str, ...]:
    """Earnings-call extraction fields aggregated by per-(instrument,
    date) mean — derived from :class:`EarningsCallExtraction`'s field
    catalogues. Same rationale as :func:`_filing_mean_fields`."""
    from quant_platform.research.features.text.schemas import EarningsCallExtraction

    return (
        *EarningsCallExtraction.SIGNED_FIELDS,
        *EarningsCallExtraction.UNSIGNED_FIELDS,
    )


def build_earnings_call_panel(
    *,
    records: Iterable[ExtractedRecord],
    documents: Iterable[SourceDocument],
) -> AggregatedTextPanel:
    """Aggregate earnings-call extractions into a per-(instrument, date) frame.

    Columns: ``call_count``, ``call_failure_count``, one
    ``call_<field>`` per earnings-call extraction field. Non-call
    records are silently ignored.
    """
    from quant_platform.research.features.text.schemas import EarningsCallExtraction

    return _build_kind_panel(
        records=records,
        document_index={doc.doc_id: doc for doc in documents},
        kind_filter="earnings-call",
        column_prefix="call_",
        mean_field_names=_earnings_call_mean_fields(),
        extraction_class=EarningsCallExtraction,
    )


__all__ = [
    "NEGATIVE_SENTIMENT_THRESHOLD",
    "POSITIVE_SENTIMENT_THRESHOLD",
    "AggregatedTextPanel",
    "build_earnings_call_panel",
    "build_filing_panel",
    "build_text_panel",
]
