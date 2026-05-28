# `text-event-v2` Feature Family

> Definitive reference for the LLM-extracted text feature family registered
> under `family="text"`, `version="text-event-v2"`. Pairs with — but is
> distinct from — the SEC EDGAR ingestion path described in
> [`text-event-alpha-scope.md`](text-event-alpha-scope.md), which feeds the
> older `text-events` CLI / `text_events` Postgres infrastructure.

## At a glance

| Field | Value |
|---|---|
| Family name | `text` |
| Family version | `text-event-v2` |
| Source files | `src/quant_platform/research/features/text/` |
| Public entry point | `compute_text_features(records, documents, config, trading_dates=None)` |
| Document kinds | `news`, `filing-10k`, `filing-10q`, `filing-8k`, `earnings-call` |
| Feature count | **27** (10 news + 10 filings + 7 earnings calls) |
| Schema version | `v2` (loader is read-compatible with `v1`) |
| Tests | `tests/unit/research_service/features/text/` (111 tests) |

The family ships through the canonical `FamilyRegistry` like every other
feature family: the package `__init__.py` builds a `FamilyManifest` and calls
`register_family(MANIFEST)` at import time. Downstream code never imports
`text` private modules — it queries the registry by `(name, version)`.

## Document kinds and extraction schemas

Three structured-extraction dataclasses live in `schemas.py`. Each emits
named floats per source document — never prose, never recommendations.

```python
NewsExtraction(sentiment, materiality, demand_signal, margin_pressure,
               guidance_signal, litigation_risk, novelty)

FilingExtraction(risk_sentiment, uncertainty_score, management_tone,
                 litigation_risk, guidance_sentiment, supply_chain_risk,
                 inventory_risk, margin_pressure, demand_weakness,
                 financing_stress)

EarningsCallExtraction(management_confidence, analyst_pushback, guidance_quality,
                       margin_pressure, demand_signal, capex_intent,
                       inventory_problem)
```

**Scoring convention** (all three schemas):

- Signed fields live in `[-1, 1]`. `0` is the explicit "no signal" sentinel,
  never `None`.
- Unsigned fields live in `[0, 1]`.
- "Risk" fields are **reverse-coded**: `+1` = healthy / low risk, `-1` =
  stressed / elevated risk. This keeps positive-direction always meaning
  "good for forward return" — the platform contract.
- `NewsExtraction.novelty` defaults to `0.5` (not `0.0`) — the unsigned
  "neutral" sentinel.

`ExtractedRecord` wraps any one of these via a tagged union:

```python
record.extraction: NewsExtraction | FilingExtraction | EarningsCallExtraction | None
record.provenance: ExtractionProvenance | None
record.failure: FailedExtraction | None      # tagged-union alternative
```

## Dispatch: `source_kind → schema → prompt`

Three whitelisted source kinds drive every routing decision. The whitelist
lives in `schemas.py` as `KNOWN_SOURCE_KINDS` and `KNOWN_FILING_KINDS`:

| `source_kind` | Extraction class | Prompt |
|---|---|---|
| `news` | `NewsExtraction` | `news-prompt-v1` |
| `filing-10k`, `filing-10q`, `filing-8k` | `FilingExtraction` | `filing-prompt-v1` |
| `earnings-call` | `EarningsCallExtraction` | `earnings-call-prompt-v1` |

Two helpers route on `source_kind`:

```python
from quant_platform.research.features.text.schemas import extraction_from_dict_for_kind
from quant_platform.research.features.text.prompts import get_prompt_for_kind

prompt = get_prompt_for_kind("filing-10q")     # → filing-prompt-v1
extraction = extraction_from_dict_for_kind("filing-10q", payload)
```

Unknown kinds raise `ValueError` rather than silently routing to the wrong
schema. Adding a new SEC form type (6-K, 20-F, etc.) is a one-line addition
to `KNOWN_FILING_KINDS`.

## The 27 features

### News (10) — daily, materiality-weighted

All news features are emitted on dates the instrument had at least one
news article (or, when `trading_dates` is supplied, on every requested
date with zero-count fallback for missing days).

| Feature | Formula | Notes |
|---|---|---|
| `news_sentiment_1d` | Materiality-weighted mean sentiment for the day | Range `[-1, 1]` |
| `news_sentiment_5d` | Per-instrument 5-trading-day rolling mean of `news_sentiment_1d` | Full-window only |
| `news_volume_1d` | Count of successful extractions on date `d` | Failures are tracked separately |
| `news_volume_zscore_20d` | Per-instrument 20-day z-score of `news_volume_1d` | Surfaces unusual coverage |
| `positive_news_shock` | Count of articles with sentiment ≥ `+0.3` | Tail counter |
| `negative_news_shock` | Count of articles with sentiment ≤ `−0.3` | Tail counter |
| `sentiment_change` | `news_sentiment_1d[t] − news_sentiment_1d[t-1]` per-instrument | Reversal detection |
| `sentiment_dispersion` | Per-day population std of per-article sentiment | Disagreement metric |
| `news_novelty` | Per-day mean `novelty` score | New info vs restatement |
| `event_materiality` | Per-day mean `materiality` score | Attention-weight hook |

### Filings (10) — sparse, on-publication-date

Every filing feature is **non-NaN only on dates a filing actually
published** for that instrument. The trainer chooses forward-fill / decay
downstream; sparse-on-event-date is the PIT-honest representation.

| Feature | Source | Reverse-coded |
|---|---|---|
| `filing_risk_sentiment` | `FilingExtraction.risk_sentiment` | ✓ (+1 = de-risking) |
| `filing_uncertainty_score` | `FilingExtraction.uncertainty_score` | — (unsigned `[0, 1]`) |
| `management_tone_change` | Per-instrument `management_tone[this filing] − management_tone[prior filing]` | — |
| `litigation_risk_score` | `FilingExtraction.litigation_risk` | ✓ |
| `filing_guidance_sentiment` | `FilingExtraction.guidance_sentiment` | — |
| `supply_chain_risk_score` | `FilingExtraction.supply_chain_risk` | ✓ |
| `inventory_risk_score` | `FilingExtraction.inventory_risk` | ✓ |
| `margin_pressure_score` | `FilingExtraction.margin_pressure` | ✓ |
| `demand_weakness_score` | `FilingExtraction.demand_weakness` | ✓ |
| `financing_stress_score` | `FilingExtraction.financing_stress` | ✓ |

**Naming convention.** Filing exports come in two shapes:

- `filing_<topic>` — used when the source field is a neutral score
  (`filing_risk_sentiment`, `filing_uncertainty_score`,
  `filing_guidance_sentiment`). Exported name == panel column name.
- `<topic>_score` — used when the source field names a *risk* whose panel
  column is already sign-flipped. `_score` flags the export as
  "already-oriented for forward-return scoring," not "raw risk level."

### Earnings calls (7) — sparse, on-publication-date

Same sparseness semantics as filings.

| Feature | Source | Reverse-coded |
|---|---|---|
| `management_confidence` | `EarningsCallExtraction.management_confidence` | — |
| `analyst_pushback` | `EarningsCallExtraction.analyst_pushback` | — (unsigned) |
| `guidance_quality` | `EarningsCallExtraction.guidance_quality` | — |
| `call_margin_pressure` | `EarningsCallExtraction.margin_pressure` | ✓ |
| `call_demand_signal` | `EarningsCallExtraction.demand_signal` | — |
| `capex_intent` | `EarningsCallExtraction.capex_intent` | — |
| `inventory_problem` | `EarningsCallExtraction.inventory_problem` | — |

The `call_*` prefix appears only where the feature would otherwise collide
with a filing feature with similar semantics (`margin_pressure_score`,
`demand_weakness_score`).

## Direction conventions and evidence gating

**All 27 features ship `expected_direction="unknown"` and `larger_is_better=False`.**

Text features are too new to ship with a-priori direction claims. The brief
is explicit: text features are evidence-gated. Promotion from `unknown` to
`+`/`-` is a **version bump** (`text-event-v3`), not an in-place edit.

The downstream walk-forward + signal-gate pipeline is what decides whether
a text feature earns a direction. The `governance_service` LLM-live-startup
assertion enforces that the prompt version + manifest + audit hashes still
match before any live trading uses the family.

## Compute pipeline

```text
SourceDocument(s) ─→ LLMClient + prompt ─→ ExtractedRecord(s)
                                                │
                                                ▼
                                       build_text_panel   (news)
                                       build_filing_panel (filings)
                                       build_earnings_call_panel (calls)
                                                │
                                                ▼
                                        Three per-kind panels
                                                │
                                                ▼
                              compute_text_features (outer-join + transforms)
                                                │
                                                ▼
                                         FeatureFrame (27 cols)
```

### Aggregator panels

Each per-kind builder produces a wide DataFrame keyed by
`(instrument_id, date)`. The builders filter their input records by
`source_kind`, so callers can pass one mixed list:

- `build_text_panel(records=mixed, documents=docs)` — keeps only
  `source_kind == "news"` records.
- `build_filing_panel(records=mixed, documents=docs)` — keeps records whose
  kind is in `KNOWN_FILING_KINDS`.
- `build_earnings_call_panel(records=mixed, documents=docs)` — keeps
  `source_kind == "earnings-call"` records.

`compute_text_features` calls all three with a shared `document_index`
built once, then outer-joins their frames on `(instrument_id, date)`.

### PIT safety

The panel date is keyed to `SourceDocument.published_at`, not to
`ExtractionProvenance.extracted_at`. The LLM's extraction timestamp is
metadata, not a join key — that's the whole point of running extraction
offline ahead of the next trading day. A feature value at date `d` only
sees documents that became publicly available on or before `d`.

### Failure persistence

When the LLM returns malformed JSON, out-of-range scores, or transient
errors past the retry budget, the pipeline writes a `FailedExtraction`
record rather than dropping the document silently. The aggregator counts
failures into `<prefix>failure_count` per panel so coverage diagnostics
stay honest.

## Configuration

`TextEventConfig` (in `config.py`) carries the family knobs. All have
sensible defaults; pass an instance to `compute_text_features(config=...)`
only when you need a non-default window:

```python
@dataclass(frozen=True)
class TextEventConfig(BaseFamilyConfig):
    version: str = "text-event-v2"
    volume_zscore_window: int = 20      # → news_volume_zscore_20d
    sentiment_window: int = 5           # → news_sentiment_5d
    tone_change_window: int = 90        # reserved for management_tone_change cadence
```

Bumping any window changes the feature column name (the window appears in
it) so it requires a feature-set version bump.

## Schema versioning and v1 compatibility

```python
SCHEMA_VERSION: str = "v2"
LOADABLE_SCHEMA_VERSIONS: frozenset[str] = frozenset({"v1", "v2"})
```

`ExtractedRecord.from_payload` accepts both `v1` and `v2` JSONL rows. `v1`
records predate the tagged-union routing — they were always
`NewsExtraction`, so the loader force-routes them to news dispatch
regardless of any `source_kind` field. Re-emitted JSONL preserves the
original version stamp.

This means the ~100 v1 articles already persisted to Postgres from the
earlier TWS dev run (see PR #47) remain readable without an external
migration step.

## Operator quickstart

```python
from quant_platform.research.features.text import (
    DEFAULT_CONFIG,
    compute_text_features,
)
from quant_platform.research.features.text.client import MockLLMClient
from quant_platform.research.features.text.extraction import extract_documents
from quant_platform.research.features.text.prompts import (
    get_news_prompt,
    get_filing_prompt,
    get_earnings_call_prompt,
)

# Run extraction (offline, retries bounded, failures persisted)
news_records = extract_documents(
    client=MockLLMClient(responder=...),
    prompt=get_news_prompt(),
    documents=news_documents,
)
filing_records = extract_documents(
    client=MockLLMClient(responder=...),
    prompt=get_filing_prompt(),
    documents=filing_documents,
)
call_records = extract_documents(
    client=MockLLMClient(responder=...),
    prompt=get_earnings_call_prompt(),
    documents=call_documents,
)

# Mixed records list — compute_text_features fans out internally
feature_frame = compute_text_features(
    records=[*news_records, *filing_records, *call_records],
    documents=[*news_documents, *filing_documents, *call_documents],
    config=DEFAULT_CONFIG,
    trading_dates=None,          # or a pd.DatetimeIndex to densify news rows
)

feature_frame.frame              # 27-column DataFrame
feature_frame.coverage           # per-feature notna() count
feature_frame.feature_specs      # name → FeatureSpec
```

## Where to look next

- Code: `src/quant_platform/research/features/text/`
- Tests: `tests/unit/research_service/features/text/` (111 tests)
- Promotion runbook: [`runbooks/llm-text-feature-promotion.md`](runbooks/llm-text-feature-promotion.md)
- Related SEC ingestion path: [`text-event-alpha-scope.md`](text-event-alpha-scope.md)
- Phase 5 status: [`architecture/production-roadmap.md`](architecture/production-roadmap.md)
