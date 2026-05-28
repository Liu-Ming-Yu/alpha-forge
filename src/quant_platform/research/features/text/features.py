"""Top-level compute + FeatureSpec derivation for ``text-event-v2``.

The text family ships **27 features** across three document types:

News (10)
~~~~~~~~~

* ``news_sentiment_1d`` — materiality-weighted mean sentiment for the day.
* ``news_sentiment_5d`` — per-instrument 5-trading-day rolling mean of
  ``news_sentiment_1d`` (smooths single-article noise).
* ``news_volume_1d`` — count of successful extractions for the day.
* ``news_volume_zscore_20d`` — z-score of ``news_volume_1d`` over a
  per-instrument 20-day window.
* ``positive_news_shock`` / ``negative_news_shock`` — tail counters
  using the ±0.3 cutoff.
* ``sentiment_change`` — ``news_sentiment_1d[t] − news_sentiment_1d[t-1]``,
  per-instrument. Captures intraday sentiment reversals when news
  flow is dense.
* ``sentiment_dispersion`` — per-(instrument, date) standard deviation
  of per-article sentiment. Captures *disagreement* across the day's
  articles independently of the materiality-weighted mean.
* ``news_novelty`` — per-(instrument, date) average novelty of the
  day's articles. New information vs. restatement.
* ``event_materiality`` — per-(instrument, date) average materiality
  of the day's articles. Hooks downstream models that want to
  attention-weight by materiality.

Filings (10)
~~~~~~~~~~~~

All filing features are **sparse**: non-NaN only on dates where a
filing actually published. The model trainer is responsible for
forward-fill / decay decisions downstream — sparse-on-filing-date is
the PIT-honest representation.

* ``filing_risk_sentiment`` — Risk Factors section sentiment.
* ``filing_uncertainty_score`` — hedge-word density.
* ``management_tone_change`` — per-instrument
  ``management_tone[this_filing] − management_tone[prior_filing]``.
* ``litigation_risk_score`` — sign-flipped legal/regulatory risk.
* ``filing_guidance_sentiment`` — explicit forward-guidance tone.
* ``supply_chain_risk_score`` — sign-flipped supply-chain health.
* ``inventory_risk_score`` — sign-flipped inventory health.
* ``margin_pressure_score`` — sign-flipped margin trajectory.
* ``demand_weakness_score`` — sign-flipped end-market demand.
* ``financing_stress_score`` — sign-flipped financing-profile health.

Earnings calls (7)
~~~~~~~~~~~~~~~~~~

Same sparse-on-publication-date semantics as filings.

* ``management_confidence`` — prepared-remarks confidence.
* ``analyst_pushback`` — Q&A pushback intensity.
* ``guidance_quality`` — specificity/breadth of forward guidance.
* ``call_margin_pressure`` — sign-flipped margin trajectory from call.
* ``call_demand_signal`` — demand signal from call.
* ``capex_intent`` — forward capex intent.
* ``inventory_problem`` — inventory issue acknowledgement.

Direction conventions
---------------------

All 27 features ship with ``expected_direction="unknown"`` and
``larger_is_better=False``. Text features are too new to ship with
a-priori direction claims; the brief is explicit that they're
evidence-gated. Once walk-forward evidence accumulates, the direction
and ``larger_is_better`` flags should be tightened — and that change
is a version bump, not an in-place edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.features.contracts import FeatureFrame, FeatureSpec
from quant_platform.research.features.text.aggregator import (
    _build_kind_panel,
    _build_news_panel_from_index,
    _earnings_call_mean_fields,
    _filing_mean_fields,
)
from quant_platform.research.features.text.config import (
    DEFAULT_CONFIG,
    TextEventConfig,
)
from quant_platform.research.features.transforms import (
    DEFAULT_KEY_COLUMNS,
    group_by_instrument,
    group_rolling_mean,
    group_rolling_std,
    group_shift,
    safe_div,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from quant_platform.research.features.text.schemas import (
        ExtractedRecord,
        SourceDocument,
    )


REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "date",
)


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------


_FILING_FEATURE_FIELD_MAP: dict[str, str] = {
    "filing_risk_sentiment": "filing_risk_sentiment",
    "filing_uncertainty_score": "filing_uncertainty_score",
    "litigation_risk_score": "filing_litigation_risk",
    "filing_guidance_sentiment": "filing_guidance_sentiment",
    "supply_chain_risk_score": "filing_supply_chain_risk",
    "inventory_risk_score": "filing_inventory_risk",
    "margin_pressure_score": "filing_margin_pressure",
    "demand_weakness_score": "filing_demand_weakness",
    "financing_stress_score": "filing_financing_stress",
}
"""Map exported filing feature name → filing-panel column it reads from.

The naming convention follows the brief and has two exported-name
shapes:

* ``filing_<topic>`` — used when the source field is a neutral score
  (``filing_risk_sentiment``, ``filing_uncertainty_score``,
  ``filing_guidance_sentiment``). Exported name == panel column name.
* ``<topic>_score`` — used when the source field names a *risk* whose
  panel column is already sign-flipped (``litigation_risk_score`` reads
  ``filing_litigation_risk``, which is +1=low/−1=high per the
  schema's reverse-coding convention). The ``_score`` suffix flags
  the export as "already-oriented for forward-return scoring", not
  "this is the raw risk level".

``management_tone_change`` is computed (a per-instrument inter-filing
diff), not a direct column read, so it's handled separately below."""

_CALL_FEATURE_FIELD_MAP: dict[str, str] = {
    "management_confidence": "call_management_confidence",
    "analyst_pushback": "call_analyst_pushback",
    "guidance_quality": "call_guidance_quality",
    "call_margin_pressure": "call_margin_pressure",
    "call_demand_signal": "call_demand_signal",
    "capex_intent": "call_capex_intent",
    "inventory_problem": "call_inventory_problem",
}
"""Map exported earnings-call feature name → call-panel column it reads from.

Earnings-call exports have a simpler rule than filings: most features
are exported by the same name as the underlying schema field
(``management_confidence``, ``capex_intent``, ...). The exceptions
(``call_margin_pressure``, ``call_demand_signal``) disambiguate from
filing features with similar semantics (``margin_pressure_score``,
``demand_weakness_score``) so a downstream catalog never has two
features with the same name."""


def _column_or_nan(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Read ``column`` from ``frame`` as a float ``np.ndarray``, or
    return an all-NaN array of the right length when the column is
    missing.

    The compute layer outer-joins three per-kind panels; a column
    only exists when its source panel was non-empty. This helper
    keeps the per-column read concise and the missing-column
    fallback unambiguous (NaN, never a stray zero).
    """
    if column in frame.columns:
        # pandas' .to_numpy() is typed as Any in this stubs version;
        # the actual return is an ndarray and downstream consumers
        # treat it as such. np.asarray pins the type for mypy
        # without copying when the source is already an ndarray.
        return np.asarray(frame[column].astype(float).to_numpy(), dtype=float)
    return np.full(len(frame), np.nan, dtype=float)


def _build_specs(
    version: str,
    *,
    volume_window: int,
    sentiment_window: int,
) -> tuple[FeatureSpec, ...]:
    volume_z_name = f"news_volume_zscore_{volume_window}d"
    sentiment_5d_name = f"news_sentiment_{sentiment_window}d"
    news_specs: tuple[FeatureSpec, ...] = (
        FeatureSpec(
            name="news_sentiment_1d",
            family="text",
            description=(
                "Materiality-weighted mean LLM-extracted sentiment of all "
                "articles published about the instrument on date d. Range "
                "[-1, 1]."
            ),
            expected_direction="unknown",
            required_inputs=("count", "sentiment_mean"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=sentiment_5d_name,
            family="text",
            description=(
                f"Per-instrument {sentiment_window}-trading-day rolling mean "
                "of news_sentiment_1d. Smooths single-article noise."
            ),
            expected_direction="unknown",
            required_inputs=("count", "sentiment_mean"),
            point_in_time=True,
            lookback_days=sentiment_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="news_volume_1d",
            family="text",
            description=(
                "Count of successful news extractions on date d. Failed "
                "extractions are not counted; the storage layer tracks "
                "them separately for coverage diagnostics."
            ),
            expected_direction="unknown",
            required_inputs=("count",),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name=volume_z_name,
            family="text",
            description=(
                f"Per-instrument rolling z-score of news_volume_1d over "
                f"{volume_window} trading days. Captures unusual coverage "
                "vs. each instrument's own baseline."
            ),
            expected_direction="unknown",
            required_inputs=("count",),
            point_in_time=True,
            lookback_days=volume_window,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="positive_news_shock",
            family="text",
            description=(
                "Count of articles on date d whose extracted sentiment "
                "is >= 0.3. Tail counter — complements the mean."
            ),
            expected_direction="unknown",
            required_inputs=("positive_count",),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="negative_news_shock",
            family="text",
            description=(
                "Count of articles on date d whose extracted sentiment "
                "is <= -0.3. Tail counter — complements the mean."
            ),
            expected_direction="unknown",
            required_inputs=("negative_count",),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="sentiment_change",
            family="text",
            description=(
                "Per-instrument day-over-day change in news_sentiment_1d. "
                "Captures reversals in sentiment flow."
            ),
            expected_direction="unknown",
            required_inputs=("sentiment_mean",),
            point_in_time=True,
            lookback_days=2,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="sentiment_dispersion",
            family="text",
            description=(
                "Per-(instrument, date) standard deviation of per-article "
                "sentiment. Captures disagreement across the day's "
                "articles, independent of the materiality-weighted mean."
            ),
            expected_direction="unknown",
            required_inputs=("count", "sentiment_sum", "sentiment_squared_sum"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="news_novelty",
            family="text",
            description=(
                "Per-(instrument, date) mean novelty score of the day's "
                "articles. 1.0 = breaking news, 0.0 = restatement."
            ),
            expected_direction="unknown",
            required_inputs=("novelty_sum", "count"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
        FeatureSpec(
            name="event_materiality",
            family="text",
            description=(
                "Per-(instrument, date) mean materiality of the day's "
                "articles. Hooks attention-weighted downstream models."
            ),
            expected_direction="unknown",
            required_inputs=("materiality_sum", "count"),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        ),
    )

    filing_descriptions: dict[str, str] = {
        "filing_risk_sentiment": (
            "Risk Factors section sentiment from the most recent SEC "
            "filing (10-K/10-Q/8-K) published on date d. Reverse-coded "
            "so +1 = de-risking language. Sparse — NaN on non-filing days."
        ),
        "filing_uncertainty_score": (
            "Hedge-word density of the SEC filing published on date d, "
            "in [0, 1]. Sparse — NaN on non-filing days."
        ),
        "management_tone_change": (
            "Per-instrument change in MD&A management tone from the "
            "prior filing to the filing published on date d. Sparse — "
            "NaN on non-filing days and on the instrument's first "
            "filing (no prior to diff against)."
        ),
        "litigation_risk_score": (
            "Filing's sign-flipped legal/regulatory risk score. +1 = "
            "low risk, -1 = elevated. Sparse — NaN on non-filing days."
        ),
        "filing_guidance_sentiment": (
            "Filing's explicit forward-guidance tone. 0 when no "
            "guidance is provided. Sparse — NaN on non-filing days."
        ),
        "supply_chain_risk_score": (
            "Filing's sign-flipped supply-chain health score. +1 = "
            "healthy, -1 = stressed. Sparse — NaN on non-filing days."
        ),
        "inventory_risk_score": (
            "Filing's sign-flipped inventory health score. +1 = clean, "
            "-1 = bloated. Sparse — NaN on non-filing days."
        ),
        "margin_pressure_score": (
            "Filing's sign-flipped margin trajectory. +1 = expanding, "
            "-1 = pressure rising. Sparse — NaN on non-filing days."
        ),
        "demand_weakness_score": (
            "Filing's sign-flipped end-market demand signal. +1 = "
            "strengthening, -1 = weakening. Sparse — NaN on non-filing "
            "days."
        ),
        "financing_stress_score": (
            "Filing's sign-flipped financing-profile health. +1 = "
            "healthy, -1 = covenant pressure/going-concern. Sparse — "
            "NaN on non-filing days."
        ),
    }
    filing_specs: tuple[FeatureSpec, ...] = tuple(
        FeatureSpec(
            name=name,
            family="text",
            description=description,
            expected_direction="unknown",
            required_inputs=("filing_count",),
            point_in_time=True,
            lookback_days=(2 if name == "management_tone_change" else 1),
            version=version,
            larger_is_better=False,
        )
        for name, description in filing_descriptions.items()
    )

    call_descriptions: dict[str, str] = {
        "management_confidence": (
            "Prepared-remarks confidence from the earnings call published "
            "on date d. -1 = defensive, +1 = confident. Sparse — NaN on "
            "non-call days."
        ),
        "analyst_pushback": (
            "Intensity of analyst pushback in the call's Q&A, in [0, 1]. "
            "Sparse — NaN on non-call days."
        ),
        "guidance_quality": (
            "Specificity + breadth of forward guidance in the call. +1 = "
            "specific across periods, -1 = withdrawn. Sparse — NaN on "
            "non-call days."
        ),
        "call_margin_pressure": (
            "Sign-flipped margin trajectory from the earnings call. "
            "+1 = expanding, -1 = pressure rising. Sparse — NaN on "
            "non-call days."
        ),
        "call_demand_signal": (
            "Demand signal from the earnings call. +1 = rising demand, "
            "-1 = falling. Sparse — NaN on non-call days."
        ),
        "capex_intent": (
            "Forward capex intent from the earnings call. +1 = "
            "acceleration, -1 = cuts/deferrals. Sparse — NaN on non-call "
            "days."
        ),
        "inventory_problem": (
            "Inventory issue acknowledgement in the call. +1 = explicit "
            "problem, -1 = explicitly clean, 0 = not discussed. Sparse "
            "— NaN on non-call days."
        ),
    }
    call_specs: tuple[FeatureSpec, ...] = tuple(
        FeatureSpec(
            name=name,
            family="text",
            description=description,
            expected_direction="unknown",
            required_inputs=("call_count",),
            point_in_time=True,
            lookback_days=1,
            version=version,
            larger_is_better=False,
        )
        for name, description in call_descriptions.items()
    )

    return news_specs + filing_specs + call_specs


FEATURE_SPECS: tuple[FeatureSpec, ...] = _build_specs(
    DEFAULT_CONFIG.version,
    volume_window=DEFAULT_CONFIG.volume_zscore_window,
    sentiment_window=DEFAULT_CONFIG.sentiment_window,
)
FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
DEFAULT_TRAINING_FEATURE_NAMES: tuple[str, ...] = tuple(
    spec.name for spec in FEATURE_SPECS if not spec.is_alias
)
_SPEC_BY_NAME: Mapping[str, FeatureSpec] = {spec.name: spec for spec in FEATURE_SPECS}


def _specs_for_config(config: TextEventConfig) -> tuple[FeatureSpec, ...]:
    if (
        config.version == DEFAULT_CONFIG.version
        and config.volume_zscore_window == DEFAULT_CONFIG.volume_zscore_window
        and config.sentiment_window == DEFAULT_CONFIG.sentiment_window
    ):
        return FEATURE_SPECS
    return _build_specs(
        config.version,
        volume_window=config.volume_zscore_window,
        sentiment_window=config.sentiment_window,
    )


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_text_features(
    *,
    records: Iterable[ExtractedRecord],
    documents: Iterable[SourceDocument],
    config: TextEventConfig = DEFAULT_CONFIG,
    trading_dates: pd.DatetimeIndex | None = None,
) -> FeatureFrame:
    """Compute the ``text-event-v2`` feature panel.

    Parameters
    ----------
    records:
        Iterable of :class:`ExtractedRecord` from the extraction
        pipeline. May mix news / filing / earnings-call records;
        each builder filters by ``source_kind``.
    documents:
        Source documents the records were extracted from. Used to
        look up each record's ``published_at`` for the panel date.
    config:
        :class:`TextEventConfig`.
    trading_dates:
        Optional explicit list of dates to materialise rows for.
        When omitted, the panel covers exactly the dates that have
        at least one extraction; when supplied, the news panel is
        densified to ``instrument × trading_dates`` (filing + call
        rows remain sparse — they're event-driven, not daily).

    Returns
    -------
    FeatureFrame
    """
    from quant_platform.research.features.text.schemas import (
        KNOWN_FILING_KINDS,
        EarningsCallExtraction,
        FilingExtraction,
    )

    record_list = list(records)
    # Build the document index once and share it across the three
    # per-kind panel builders — the alternative was three identical
    # O(n) dict comprehensions inside each builder.
    document_index = {doc.doc_id: doc for doc in documents}

    specs = _specs_for_config(config)
    feature_names = tuple(spec.name for spec in specs)
    spec_by_name: dict[str, FeatureSpec] = {spec.name: spec for spec in specs}
    volume_z_name = f"news_volume_zscore_{config.volume_zscore_window}d"
    sentiment_5d_name = f"news_sentiment_{config.sentiment_window}d"

    news_panel = _build_news_panel_from_index(
        records=record_list, document_index=document_index
    ).frame
    filing_panel = _build_kind_panel(
        records=record_list,
        document_index=document_index,
        kind_filter=KNOWN_FILING_KINDS,
        column_prefix="filing_",
        mean_field_names=_filing_mean_fields(),
        extraction_class=FilingExtraction,
    ).frame
    call_panel = _build_kind_panel(
        records=record_list,
        document_index=document_index,
        kind_filter="earnings-call",
        column_prefix="call_",
        mean_field_names=_earnings_call_mean_fields(),
        extraction_class=EarningsCallExtraction,
    ).frame

    if trading_dates is not None and not news_panel.empty:
        news_panel = _expand_to_grid(news_panel, trading_dates)

    # ------------------------------------------------------------------
    # Stitch the three panels onto a unified (instrument_id, date) grid.
    # ------------------------------------------------------------------
    panels = [p for p in (news_panel, filing_panel, call_panel) if not p.empty]
    if not panels:
        empty_frame = pd.DataFrame(
            {
                "instrument_id": pd.Series(dtype=str),
                "date": pd.Series(dtype="datetime64[ns]"),
                **{name: pd.Series(dtype=float) for name in feature_names},
            }
        )
        coverage = {name: 0 for name in feature_names}
        return FeatureFrame(
            frame=empty_frame,
            feature_names=feature_names,
            feature_specs=spec_by_name,
            coverage=coverage,
            key_columns=DEFAULT_KEY_COLUMNS,
        )

    # Outer-join the panels on (instrument_id, date) so each (instrument, date)
    # row carries whatever panel data we have for that date. Pandas merge
    # preserves NaN for missing panels — exactly the sparse-on-event-date
    # representation we want.
    joined = panels[0]
    for other in panels[1:]:
        joined = joined.merge(other, on=["instrument_id", "date"], how="outer")
    joined = joined.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # News features
    # ------------------------------------------------------------------
    output: dict[str, object] = {
        "instrument_id": joined["instrument_id"].to_numpy(),
        "date": joined["date"].to_numpy(),
    }

    # News features rely on news panel columns. When the news panel was
    # empty for some instruments, those columns will be NaN — the
    # rolling helpers tolerate this.
    sentiment_mean = joined.get("sentiment_mean")
    count = joined.get("count")
    if sentiment_mean is None:
        sentiment_mean = pd.Series(np.nan, index=joined.index, dtype=float)
    else:
        sentiment_mean = sentiment_mean.astype(float)
    if count is None:
        count = pd.Series(0, index=joined.index, dtype=float)
    else:
        count = count.fillna(0).astype(float)

    # Replace inf with NaN before assignment so downstream consumers don't
    # see infinities masquerading as data.
    output["news_sentiment_1d"] = sentiment_mean.to_numpy()
    output["news_volume_1d"] = count.to_numpy()

    grouped = group_by_instrument(joined.assign(_count=count, _sentiment=sentiment_mean))
    rolling_mean_count = group_rolling_mean(grouped["_count"], config.volume_zscore_window)
    rolling_std_count = group_rolling_std(grouped["_count"], config.volume_zscore_window)
    volume_z = safe_div(
        count - rolling_mean_count, rolling_std_count, require_positive_denom=False
    ).replace([np.inf, -np.inf], np.nan)
    output[volume_z_name] = volume_z.to_numpy()

    rolling_mean_sentiment = group_rolling_mean(
        grouped["_sentiment"], config.sentiment_window, policy="full"
    )
    output[sentiment_5d_name] = rolling_mean_sentiment.to_numpy()

    sentiment_change = sentiment_mean - group_shift(grouped["_sentiment"], 1)
    output["sentiment_change"] = sentiment_change.to_numpy()

    # Positive/negative shocks default to zero (not NaN) on no-news
    # dates so the tail-counters compose cleanly with a densified
    # trading calendar.
    if "positive_count" in joined.columns:
        output["positive_news_shock"] = joined["positive_count"].fillna(0).astype(float).to_numpy()
    else:
        output["positive_news_shock"] = np.zeros(len(joined), dtype=float)
    if "negative_count" in joined.columns:
        output["negative_news_shock"] = joined["negative_count"].fillna(0).astype(float).to_numpy()
    else:
        output["negative_news_shock"] = np.zeros(len(joined), dtype=float)

    output["sentiment_dispersion"] = _column_or_nan(joined, "sentiment_dispersion")
    output["news_novelty"] = _column_or_nan(joined, "novelty_mean")
    output["event_materiality"] = _column_or_nan(joined, "materiality_mean")

    # ------------------------------------------------------------------
    # Filing features — sparse-on-publication-date.
    # ------------------------------------------------------------------
    for exported_name, source_column in _FILING_FEATURE_FIELD_MAP.items():
        output[exported_name] = _column_or_nan(joined, source_column)

    output["management_tone_change"] = _management_tone_change(joined)

    # ------------------------------------------------------------------
    # Earnings-call features — sparse-on-publication-date.
    # ------------------------------------------------------------------
    for exported_name, source_column in _CALL_FEATURE_FIELD_MAP.items():
        output[exported_name] = _column_or_nan(joined, source_column)

    output_frame = pd.DataFrame(output)
    coverage = {name: int(output_frame[name].notna().sum()) for name in feature_names}

    return FeatureFrame(
        frame=output_frame,
        feature_names=feature_names,
        feature_specs=spec_by_name,
        coverage=coverage,
        key_columns=DEFAULT_KEY_COLUMNS,
    )


def _management_tone_change(joined: pd.DataFrame) -> np.ndarray:
    """Per-instrument ``filing_management_tone[t] − filing_management_tone[prior]``.

    The filing panel has rows only on filing dates and contributes
    ``filing_management_tone`` to the joined frame. We want the
    feature to be NaN everywhere except on filing dates, and on the
    first filing for an instrument (no prior to diff against).

    Implementation:

    1. Drop the NaN rows (non-filing dates) per instrument.
    2. Diff vs. the previous filing within each instrument.
    3. Reindex back onto the joined-frame index — the dropped rows
       stay NaN, the kept rows carry the inter-filing diff.

    This avoids ``groupby.apply`` + ``reset_index(level=0, drop=True)``:
    ``groupby(group_keys=False).diff()`` is a vectorised pandas idiom
    that doesn't add a MultiIndex level we'd then have to strip.
    """
    if "filing_management_tone" not in joined.columns:
        return np.full(len(joined), np.nan, dtype=float)

    tone = joined["filing_management_tone"].astype(float)
    # Drop NaN within instruments so .diff() compares each filing to
    # the **previous filing**, not to the NaN row in between. The
    # dropped rows reappear as NaN after reindex back to the joined
    # index, which is the sparse semantics we want.
    filing_only = tone.dropna()
    diffs = filing_only.groupby(
        joined.loc[filing_only.index, "instrument_id"], group_keys=False, sort=False
    ).diff()
    return np.asarray(diffs.reindex(joined.index).to_numpy(), dtype=float)


def _expand_to_grid(
    panel: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Expand the news aggregator output to a full instrument × trading_date grid.

    Missing cells get zero counts + NaN ``sentiment_mean`` so the
    rolling z-score sees "no news" as zero (which is what we want
    for the volume-z calculation) but the sentiment-mean stays
    distinguishable from "extracted mean=0".
    """
    instruments = panel["instrument_id"].unique()
    grid = pd.MultiIndex.from_product(
        [instruments, trading_dates], names=["instrument_id", "date"]
    ).to_frame(index=False)
    merged = grid.merge(panel, on=["instrument_id", "date"], how="left")
    fill_columns = (
        "count",
        "positive_count",
        "negative_count",
        "failure_count",
    )
    for col in fill_columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    return merged


__all__ = [
    "DEFAULT_TRAINING_FEATURE_NAMES",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "REQUIRED_INPUT_COLUMNS",
    "compute_text_features",
]
